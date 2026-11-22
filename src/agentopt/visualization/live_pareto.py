"""Fixed-size desktop view for live model-selection snapshots."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread, current_thread, main_thread
from textwrap import wrap


BG = "#f4f6f5"
PANEL = "#ffffff"
EDGE = "#dce5e2"
TEXT = "#192e35"
MUTED = "#657779"
MINT = "#147d70"
BLUE = "#3985a0"


class LiveParetoPlot:
    """Render on the main thread while selection runs in a worker thread."""

    def __init__(self, *, total_questions: int, warm_total: int | None = None,
                 title: str = "Model selection", output: Path | None = None,
                 total_combinations: int | None = None,
                 show_window: bool = True, stop_event: Event | None = None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import fontManager
            from matplotlib.patches import FancyBboxPatch
            from matplotlib.widgets import Button
        except ImportError as exc:
            raise RuntimeError("Live plotting requires matplotlib: pip install agentopt-py[plot]") from exc

        if total_questions < 1 or (warm_total is not None and warm_total < 1):
            raise ValueError("Question and warm-start counts must be positive")
        if show_window and plt.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
            raise RuntimeError("Live plotting requires an interactive Matplotlib backend")
        self.plt = plt
        self.font = "Segoe UI" if any(f.name == "Segoe UI" for f in fontManager.ttflist) else "DejaVu Sans"
        if total_combinations is not None and total_combinations < 1:
            raise ValueError("Combination count must be positive")
        self.total_cells = total_combinations * total_questions if total_combinations is not None else None
        self.total_questions = total_questions
        self.warm_total = warm_total
        self.output = Path(output) if output else None
        self.show_window = show_window
        self.stop_event = stop_event if stop_event is not None else Event()
        self._resume = Event()
        self._resume.set()
        self.paused = False
        self._pause_active = False
        self.finished = False
        self.finish_requested = False
        self._warming_up = True
        self._stop_reason = ""
        self._recommendations = []
        self._legend_offset = 0
        self._last_evaluations = 0
        self._last_spent = "—"
        self._last_status = None
        # Keep this setting scoped to this window, including its toolbar.
        with plt.rc_context({"toolbar": "None", "font.family": self.font}):
            self.figure = plt.figure(figsize=(11.8, 7.2), dpi=100, facecolor=BG)
        self.figure.canvas.manager.set_window_title("AgentOpt · Model selection")
        self.figure.text(.04, .945, "AgentOpt", color=MINT, fontsize=18, weight="bold", fontfamily=self.font)
        self.figure.text(.04, .899, title,
                         color=MUTED, fontsize=10, fontfamily=self.font)
        self.status_text = self.figure.text(.96, .942, "", ha="right", color=MINT, fontsize=12,
                                            fontfamily=self.font, weight="bold")
        self.metric_values = []
        for x, label in ((.04, "EVALUATED CELLS"), (.355, "EVALUATION SPEND"), (.67, "PARETO RECOMMENDATIONS")):
            self.figure.add_artist(FancyBboxPatch((x, .759), .29, .103, transform=self.figure.transFigure,
                boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=PANEL, edgecolor=EDGE, linewidth=.8, zorder=0))
            self.figure.text(x+.018, .832, label, color=MUTED, fontsize=8, fontfamily=self.font)
            self.metric_values.append(self.figure.text(x+.018, .783, "—", color=TEXT, fontsize=21,
                                                       fontfamily=self.font, weight="bold"))
        self.evaluation_count = self.figure.text(.312, .790, "", ha="right", color=MUTED,
                                                 fontsize=9, fontfamily=self.font)
        self.figure.text(.067, .698, "Estimated Pareto frontier", color=TEXT, fontsize=13,
                         fontfamily=self.font, weight="bold")
        self.figure.text(.067, .666, "Quality and deployment cost · conservative estimates", color=MUTED,
                         fontsize=9, fontfamily=self.font)
        self.axis = self.figure.add_axes((.085, .258, .45, .355), facecolor=BG)
        self.recommendation_axis = self.figure.add_axes((.59, .202, .37, .515), facecolor=PANEL)
        self.recommendation_axis.set_axis_off()
        self.detail_text = self.figure.text(.04, .132, "", color=MUTED, fontsize=10, fontfamily=self.font)
        self.figure.text(.04, .082, "Spend = cumulative evaluation cost", color=MUTED, fontsize=9, fontfamily=self.font)
        self.figure.text(.04, .057, "Deployment cost = USD per 1,000 queries", color=MUTED, fontsize=9, fontfamily=self.font)
        self.progress_axis = self.figure.add_axes((.14, .37, .34, .012), facecolor=EDGE)
        self.progress_axis.set_xlim(0, 1)
        self.progress_axis.set_ylim(0, 1)
        self.progress_axis.set_axis_off()
        self.progress_axis.barh(.5, 1, height=1, color=EDGE)
        self.progress_fill = self.progress_axis.barh(.5, 0, height=1, color=MINT)[0]
        self.empty_title = self.figure.text(.31, .49, "", color=TEXT, fontsize=18, ha="center", fontfamily=self.font)
        self.empty_detail = self.figure.text(.31, .435, "", color=MUTED, fontsize=11, ha="center", fontfamily=self.font)
        self.scroll_axis = self.figure.add_axes((.946, .242, .012, .385), zorder=5)
        self.scroll_axis.set(xlim=(0, 1), ylim=(0, 1))
        self.scroll_axis.set_axis_off()
        self.scroll_thumb, = self.scroll_axis.plot([.5, .5], [.75, 1], color="#d4ddda",
                                                  linewidth=3, solid_capstyle="round")
        self.scroll_axis.set_visible(False)
        self._scroll_drag = None
        self._chart_drag = None
        self.buttons = []
        for x, label, callback, color in ((.704, "Finish", self.finish_search, PANEL), (.839, "Pause", self.toggle_pause, MINT)):
            ax = self.figure.add_axes((x, .045, .12, .059))
            button = Button(ax, label, color=color, hovercolor=EDGE if label == "Finish" else "#1b9585")
            button.label.set_fontfamily(self.font)
            button.label.set_fontsize(11)
            button.label.set_color(TEXT if label == "Finish" else BG)
            for spine in ax.spines.values():
                spine.set_color(EDGE)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.finish_button, self.pause_button = self.buttons
        self._zoomed = False
        self.reset_button = None
        if show_window:
            reset_axis = self.figure.add_axes((.445, .677, .09, .036))
            self.reset_button = Button(reset_axis, "Reset view", color=PANEL, hovercolor=EDGE)
            self.reset_button.label.set(fontfamily=self.font, fontsize=9, color=TEXT)
            for spine in reset_axis.spines.values():
                spine.set_color(EDGE)
            self.reset_button.on_clicked(self._reset_chart)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("button_press_event", self._start_drag)
        self.figure.canvas.mpl_connect("motion_notify_event", self._drag)
        self.figure.canvas.mpl_connect("button_release_event", self._end_drag)
        self.figure.canvas.mpl_connect("close_event", self.finish_search)
        self._render_chart()
        self._render_legend()
        self._render_status()
        if show_window:
            manager = self.figure.canvas.manager
            window = getattr(manager, "window", None)
            if hasattr(window, "resizable"):
                window.resizable(False, False)
            elif hasattr(window, "setFixedSize"):
                window.setFixedSize(window.size())
            plt.ion()
            plt.show(block=False)
        self.figure.canvas.draw_idle()

    def _scroll_to(self, offset):
        self._legend_offset = max(0, min(offset, len(self._recommendations)-1))
        self._render_legend()
        self.figure.canvas.draw_idle()

    def _start_drag(self, event):
        if event.button != 1:
            return
        if event.inaxes is self.axis and self._recommendations:
            transform = self.axis.transData.frozen().inverted()
            origin = transform.transform((event.x, event.y))
            self._chart_drag = transform, origin, self.axis.get_xlim(), self.axis.get_ylim()
        elif event.inaxes is self.scroll_axis and self.scroll_axis.get_visible():
            _, y = self.scroll_axis.transAxes.inverted().transform((event.x, event.y))
            low, high = self.scroll_thumb.get_ydata()
            self._scroll_drag = y - (low+high)/2 if low <= y <= high else 0
            self._drag(event)

    def _drag(self, event):
        if self._chart_drag is not None:
            transform, origin, xlim, ylim = self._chart_drag
            dx, dy = transform.transform((event.x, event.y)) - origin
            self.axis.set_xlim(xlim[0]-dx, xlim[1]-dx)
            self.axis.set_ylim(ylim[0]-dy, ylim[1]-dy)
            self._zoomed = True
            self.figure.canvas.draw_idle()
        elif self._scroll_drag is not None:
            _, y = self.scroll_axis.transAxes.inverted().transform((event.x, event.y))
            offset = round((.875-y+self._scroll_drag) / .75 * (len(self._recommendations)-1))
            self._scroll_to(offset)

    def _end_drag(self, event):
        if event.button == 1:
            self._chart_drag = None
            self._scroll_drag = None

    def _reset_chart(self, _event):
        self._zoomed = False
        self._render_chart()
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is self.axis and self._recommendations:
            if event.xdata is None or event.ydata is None or not event.step:
                return
            factor = 1.2 ** (-event.step)
            left, right = self.axis.get_xlim()
            bottom, top = self.axis.get_ylim()
            self.axis.set_xlim(event.xdata + (left-event.xdata)*factor,
                               event.xdata + (right-event.xdata)*factor)
            self.axis.set_ylim(event.ydata + (bottom-event.ydata)*factor,
                               event.ydata + (top-event.ydata)*factor)
            self._zoomed = True
            self.figure.canvas.draw_idle()
            return
        if event.inaxes is not self.recommendation_axis:
            return
        step = 1 if event.button == "down" else -1
        self._scroll_to(self._legend_offset + step)

    def _render_status(self):
        phase = ("Finished" if self.finished else "Finishing…" if self.finish_requested
                 else "Paused" if self._pause_active else "Pausing…" if self.paused
                 else "Initial sampling" if self._warming_up else "Searching")
        status = (phase, self.total_cells, self._last_evaluations, self._last_spent, len(self._recommendations), self._stop_reason)
        if status == self._last_status:
            return
        self._last_status = status
        self.status_text.set_text(phase)
        self.status_text.set_color("#9b6a20" if self.paused and not self.finished else MINT)
        coverage = f"{100 * self._last_evaluations / self.total_cells:.2f}%" if self.total_cells else "—"
        count = f"{self._last_evaluations:,} / {self.total_cells:,}" if self.total_cells else f"{self._last_evaluations:,} completed"
        self.evaluation_count.set_text(count)
        for artist, value in zip(self.metric_values, (coverage, self._last_spent, str(len(self._recommendations)))):
            artist.set_text(value)
        detail = "Recommendations update after each completed evaluation batch."
        if self.finished:
            detail = "Search finished · " + (self._stop_reason or "completed").replace("_", " ")
        elif self.finish_requested:
            detail = "Finishing scheduled evaluations. Results will be saved when they complete."
        elif self._pause_active:
            detail = "Search paused. Continue resumes from this point."
        elif self.paused:
            detail = "Pause requested. Waiting for the current evaluation batch to complete."
        elif self._warming_up:
            detail = "Sampling each model combination before estimating the first frontier."
        self.detail_text.set_text(detail)
        disabled = self.finished or self.finish_requested
        for button in self.buttons:
            button.set_active(not disabled)
            button.ax.set_facecolor(EDGE if disabled else button.color)
            button.label.set_color(MUTED if disabled else TEXT if button is self.finish_button else BG)
        self.pause_button.label.set_text("Pause" if disabled else "Continue" if self.paused else "Pause")
        self.finish_button.label.set_text("Finished" if self.finished else "Finishing…" if self.finish_requested else "Finish")

    def _render_chart(self):
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        ax = self.axis
        has_points = bool(self._recommendations)
        ax.set_visible(has_points)
        if self.reset_button is not None:
            self.reset_button.ax.set_visible(has_points)
        self.empty_title.set_visible(not has_points)
        self.empty_detail.set_visible(not has_points)
        self.progress_axis.set_visible(not has_points and self._warming_up and not self.finished and self.warm_total is not None)
        if not has_points:
            self.empty_title.set_text("Collecting initial samples" if not self.finished else "No recommendation available")
            progress = f"{self._last_evaluations:,} completed"
            if self.warm_total:
                fraction = min(1, self._last_evaluations / self.warm_total)
                self.progress_fill.set_width(fraction)
                progress = f"{self._last_evaluations:,} / {self.warm_total:,} evaluations   ·   {fraction:.0%}"
            self.empty_detail.set_text(progress)
            return
        view = (ax.get_xlim(), ax.get_ylim())
        ax.clear()
        costs = [r.conservative_cost_usd * 1000 for r in self._recommendations]
        scores = [r.conservative_accuracy for r in self._recommendations]
        ax.plot(costs, scores, color=MINT, linewidth=2, zorder=2)
        ax.scatter(costs, scores, color=BLUE, edgecolor=BG, linewidth=1.5, s=70, zorder=3)
        for number, (cost, score) in enumerate(zip(costs, scores), 1):
            ax.annotate(str(number), (cost, score), xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color=TEXT, fontfamily=self.font, annotation_clip=True)
        ax.set_ylim(min(-.03, min(scores)-.08), max(1.03, max(scores)+.08))
        ax.margins(x=.14)
        if self._zoomed:
            ax.set_xlim(view[0])
            ax.set_ylim(view[1])
        ax.grid(color=EDGE, linewidth=.6, alpha=.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(EDGE)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.3g}"))
        ax.set_xlabel("Deployment cost · USD / 1,000 queries", color=MUTED, fontsize=10, labelpad=13, fontfamily=self.font)
        ax.set_ylabel("Estimated quality", color=MUTED, fontsize=10, labelpad=12, fontfamily=self.font)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(self.font)

    def _render_legend(self):
        from matplotlib.patches import FancyBboxPatch

        ax = self.recommendation_axis
        ax.clear()
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((.002, .002), .996, .996, transform=ax.transAxes,
            boxstyle="round,pad=0,rounding_size=0.025", facecolor=PANEL, edgecolor=EDGE))
        ax.text(.055, .913, "Recommended combinations", color=TEXT, fontsize=12,
                weight="bold", fontfamily=self.font, transform=ax.transAxes)
        if not self._recommendations:
            self.scroll_axis.set_visible(False)
            ax.text(.055, .80, "Waiting for initial samples" if not self.finished else "Initial sampling incomplete",
                    color=MINT, fontsize=11, fontfamily=self.font, transform=ax.transAxes)
            ax.text(.055, .69, "Model assignments and their quality–cost\ntrade-offs will appear here.", color=MUTED,
                    fontsize=10, linespacing=1.7, fontfamily=self.font, transform=ax.transAxes)
            return
        top = .85
        end = self._legend_offset
        for index, recommendation in enumerate(self._recommendations[self._legend_offset:], self._legend_offset):
            role_lines = []
            for role, model in recommendation.combo.items():
                role_lines.extend(wrap(f"{role.title()}: {model}", width=39, subsequent_indent="    ", break_on_hyphens=False))
            # Paginate by the actual number of wrapped role lines.
            height = .110 + .050 * len(role_lines)
            if top - height < .075:
                break
            ax.add_patch(FancyBboxPatch((.035, top-height), .93, height-.018,
                boxstyle="round,pad=0,rounding_size=.018", facecolor="#f3f7f5", edgecolor=EDGE, linewidth=.7))
            ax.text(.083, top-.094, f"{index+1}", va="center", color=BLUE,
                    weight="bold", fontsize=11, fontfamily=self.font)
            for line_index, line in enumerate(role_lines):
                ax.text(.135, top-.045-.05*line_index, line, va="top", color=TEXT,
                        fontsize=9.5, fontfamily=self.font)
            ax.text(.065, top-height+.034,
                    f"Quality {recommendation.conservative_accuracy:.3f}  ·  ${recommendation.conservative_cost_usd*1000:.3f} / 1k  ·  {recommendation.observed_samples}/{self.total_questions} tested",
                    color=MINT, fontsize=9, fontfamily=self.font)
            top -= height
            end = index+1
        self.scroll_axis.set_visible(self._legend_offset > 0 or end < len(self._recommendations))
        thumb_top = 1-.75*self._legend_offset/max(1, len(self._recommendations)-1)
        self.scroll_thumb.set_ydata([thumb_top-.25, thumb_top])
        ax.text(.055, .035, f"{self._legend_offset+1}–{end} of {len(self._recommendations)}" +
                ("   ·   Drag scrollbar or scroll" if self._legend_offset or end < len(self._recommendations) else ""),
                color=MUTED, fontsize=8.5, fontfamily=self.font)

    def toggle_pause(self, _event):
        if self.finished or self.finish_requested:
            return
        self.paused = not self.paused
        if self.paused:
            self._resume.clear()
        else:
            self._pause_active = False
            self._resume.set()
        self._render_status()
        self.figure.canvas.draw_idle()

    def finish_search(self, _event):
        if self.finished or self.finish_requested:
            return
        self.finish_requested = True
        self.paused = False
        self._resume.set()
        self.stop_event.set()
        self._render_status()
        self.figure.canvas.draw_idle()

    def update(self, snapshot):
        self._warming_up = snapshot.stop_reason == "warming_up"
        self._stop_reason = snapshot.stop_reason
        self._last_evaluations = snapshot.total_evaluations
        self._last_spent = "Unavailable" if snapshot.selection_cost is None else f"${snapshot.selection_cost:.4f}"
        recommendations = sorted(snapshot.get_recommendations(),
            key=lambda r: (r.conservative_cost_usd, -r.conservative_accuracy, r.model_name))
        redraw_legend = recommendations != self._recommendations or self.finished != snapshot.is_final
        self._recommendations = recommendations
        self._legend_offset = min(self._legend_offset, max(0, len(self._recommendations)-1))
        self.finished = snapshot.is_final
        if self.finished:
            self.paused = False
            self._pause_active = False
        self._render_chart()
        if redraw_legend:
            self._render_legend()
        self._render_status()
        self.figure.canvas.draw_idle()
        if self.finished and self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            # Export through an off-screen canvas. A closed/minimized Tk window
            # can report a zero-sized drawable during its final callback.
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            window_canvas = self.figure.canvas
            size = self.figure.get_size_inches().copy()
            try:
                FigureCanvasAgg(self.figure)
                self.figure.set_size_inches(11.8, 7.2, forward=False)
                self.figure.savefig(self.output, dpi=160, facecolor=BG)
            finally:
                self.figure.set_size_inches(size, forward=False)
                self.figure.set_canvas(window_canvas)

    def pump_events(self):
        if self.show_window and self.plt.fignum_exists(self.figure.number):
            self.figure.canvas.flush_events()
            previous = self._last_status
            self._render_status()
            if previous != self._last_status:
                self.figure.canvas.draw_idle()

    def run(self, selector, *, parallel=True, max_concurrent=4, on_update=None):
        """Run selection in a worker; render snapshots on Matplotlib's main thread.

        Pausing waits at the next selector update, after the scheduled batch.
        Finishing sets the selector's cooperative stop event. Neither action
        cancels or refunds calls that are already scheduled.
        """
        if current_thread() is not main_thread():
            raise RuntimeError("LiveParetoPlot.run must be called on the main thread")
        if getattr(selector, "objective_mode", None) != "multi":
            raise ValueError("Live Pareto plotting requires a multi-objective Gittins selector")
        self.total_questions = len(selector.dataset)
        self.total_cells = len(selector._all_combos()) * self.total_questions
        self._render_status()
        updates = Queue()
        outcome = {}

        def receive(snapshot):
            rendered = Event()
            self._pause_active = self.paused and not snapshot.is_final
            updates.put((snapshot, rendered))
            while not rendered.wait(0.05) and not self.stop_event.is_set():
                pass
            if not snapshot.is_final and not self._resume.is_set():
                self._pause_active = True
                while not self._resume.wait(0.05) and not self.stop_event.is_set():
                    pass
                self._pause_active = False

        def search():
            try:
                outcome["results"] = selector.select_best(
                    parallel=parallel, max_concurrent=max_concurrent,
                    on_update=receive, stop_event=self.stop_event,
                )
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=search, name="agentopt-live-selection")
        worker.start()
        try:
            while worker.is_alive() or not updates.empty():
                try:
                    snapshot, rendered = updates.get(timeout=0.05)
                except Empty:
                    self.pump_events()
                    continue
                if on_update is not None:
                    on_update(snapshot)
                self.update(snapshot)
                self.pump_events()
                rendered.set()
        except BaseException:
            self.stop_event.set()
            self._resume.set()
            worker.join()
            raise
        worker.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome["results"]

    def wait_for_close(self):
        if self.show_window and self.plt.fignum_exists(self.figure.number):
            self.plt.ioff()
            self.plt.show()
