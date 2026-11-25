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
FONT_SMALL = 9
FONT_MEDIUM = 11
FONT_LARGE = 19


class LiveParetoPlot:
    """Render on the main thread while selection runs in a worker thread."""

    def __init__(self, *, total_questions: int, warm_total: int | None = None,
                 title: str = "Model selection", output: Path | None = None,
                 total_combinations: int | None = None,
                 algorithm: str = "CC-Gittins",
                 show_window: bool = True, stop_event: Event | None = None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import fontManager
            from matplotlib.patches import FancyBboxPatch
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
        if self.output is not None and not self.output.suffix:
            self.output = self.output.with_suffix(".png")
        self._output_format = self.output.suffix[1:].lower() if self.output is not None else None
        self.show_window = show_window
        self.stop_event = stop_event if stop_event is not None else Event()
        self._resume = Event()
        self._resume.set()
        self.paused = False
        self._pause_active = False
        self._pending_batch_evaluations = 0
        self._closed = False
        self._closed_canvas = None
        self._card_regions = []
        self._card_press = None
        self.finished = False
        self.finish_requested = False
        self._warming_up = True
        self._stop_reason = ""
        self._recommendations = []
        self._selected_combination = None
        self._chart_labels = []
        self._chart_hovering = False
        self._legend_offset = 0.0
        self._legend_scroll_limit = 0.0
        self._last_evaluations = 0
        self._last_spent = "—"
        self._last_status = None
        # Keep this setting scoped to this window, including its toolbar.
        with plt.rc_context({"toolbar": "None", "font.family": self.font, "font.size": FONT_SMALL}):
            self.figure = plt.figure(figsize=(11.8, 7.2), dpi=100, facecolor=BG)
        self.figure.canvas.manager.set_window_title("AgentOpt · Model selection")
        self.figure.text(.04, .945, "AgentOpt", color=MINT, fontsize=FONT_LARGE, weight="bold", fontfamily=self.font)
        self.figure.text(.04, .899, f"{title} · Algorithm: {algorithm}",
                         color=MUTED, fontsize=FONT_MEDIUM, fontfamily=self.font)
        self.status_text = self.figure.text(.96, .942, "", ha="right", color=MINT, fontsize=FONT_MEDIUM,
                                            fontfamily=self.font, weight="bold")
        self.metric_values = []
        for x, label in ((.04, "EVALUATED CELLS"), (.355, "EVALUATION SPEND"), (.67, "PARETO RECOMMENDATIONS")):
            self.figure.add_artist(FancyBboxPatch((x, .759), .29, .103, transform=self.figure.transFigure,
                boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=PANEL, edgecolor=EDGE, linewidth=.8, zorder=0))
            self.figure.text(x+.018, .832, label, color=MUTED, fontsize=FONT_MEDIUM, fontfamily=self.font)
            self.metric_values.append(self.figure.text(x+.018, .783, "—", color=TEXT, fontsize=FONT_LARGE,
                                                       fontfamily=self.font, weight="bold"))
        self.evaluation_count = self.figure.text(.312, .790, "", ha="right", color=MUTED,
                                                 fontsize=FONT_SMALL, fontfamily=self.font)
        self.figure.text(.067, .698, "Estimated Pareto frontier", color=TEXT, fontsize=15,
                         fontfamily=self.font, weight="bold")
        self.figure.text(.067, .666, "Conservative estimates · Click a point or number to inspect", color=MUTED,
                         fontsize=FONT_SMALL, fontfamily=self.font)
        self.axis = self.figure.add_axes((.085, .258, .45, .355), facecolor=BG)
        self.recommendation_axis = self.figure.add_axes((.59, .202, .37, .515), facecolor=PANEL)
        self.recommendation_axis.set_axis_off()
        self.detail_text = self.figure.text(.59, .192, "", va="top", color=MUTED,
                                            fontsize=FONT_MEDIUM, fontfamily=self.font)
        footer_y, button_height = .065, .059
        self.figure.text(.085, footer_y + button_height / 2,
                         "Spend = cumulative evaluation cost\nDeployment cost = USD per 1,000 queries",
                         va="center", linespacing=1.25, color=MUTED, fontsize=FONT_MEDIUM,
                         fontfamily=self.font)
        self.progress_axis = self.figure.add_axes((.14, .37, .34, .012), facecolor=EDGE)
        self.progress_axis.set_xlim(0, 1)
        self.progress_axis.set_ylim(0, 1)
        self.progress_axis.set_axis_off()
        self.progress_axis.barh(.5, 1, height=1, color=EDGE)
        self.progress_fill = self.progress_axis.barh(.5, 0, height=1, color=MINT)[0]
        self.empty_title = self.figure.text(.31, .49, "", color=TEXT, fontsize=FONT_LARGE, ha="center", fontfamily=self.font)
        self.empty_detail = self.figure.text(.31, .435, "", color=MUTED, fontsize=FONT_MEDIUM, ha="center", fontfamily=self.font)
        self.scroll_axis = self.figure.add_axes((.946, .245, .012, .400), zorder=5)
        self.scroll_axis.set(xlim=(0, 1), ylim=(0, 1))
        self.scroll_axis.set_axis_off()
        self.scroll_thumb, = self.scroll_axis.plot([.5, .5], [0, 1], color="#d4ddda",
                                                  linewidth=3, solid_capstyle="round")
        self._scroll_drag = None
        self._chart_drag = None
        self._chart_press = None
        self.finish_button = self._make_button((.704, footer_y, .12, button_height), "Finish",
                                               self.finish_search)
        self.pause_button = self._make_button((.839, footer_y, .12, button_height), "Pause",
                                              self.toggle_pause, primary=True)
        self.buttons = (self.finish_button, self.pause_button)
        self._zoomed = False
        self.reset_button = None
        if show_window:
            self.reset_button = self._make_button((.445, .677, .09, .036), "Reset view",
                                                  self._reset_chart)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("button_press_event", self._start_drag)
        self.figure.canvas.mpl_connect("motion_notify_event", self._drag)
        self.figure.canvas.mpl_connect("button_release_event", self._end_drag)
        self.figure.canvas.mpl_connect("figure_leave_event", self._update_chart_cursor)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
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
        self._request_draw()

    def _make_button(self, bounds, label, callback, *, primary=False):
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.widgets import Button

        ax = self.figure.add_axes(bounds)
        width, height = self.figure.get_size_inches() * bounds[2:]
        color = MINT if primary else PANEL
        ax.patch = FancyBboxPatch((.008, .025), .984, .95, transform=ax.transAxes,
                                  boxstyle="round,pad=0,rounding_size=.043",
                                  mutation_aspect=width / height, facecolor=color,
                                  edgecolor=color if primary else EDGE, linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        button = Button(ax, label, color=color, hovercolor="#1b9585" if primary else EDGE)
        button.label.set(position=(.5, .5), ha="center", va="center", fontfamily=self.font,
                         fontsize=FONT_MEDIUM, color=BG if primary else TEXT)
        button.on_clicked(callback)
        return button

    def _scroll_to(self, offset):
        self._legend_offset = max(0.0, min(offset, self._legend_scroll_limit))
        self._render_legend()
        self._request_draw()

    def _start_drag(self, event):
        if event.button != 1:
            return
        if event.inaxes is self.axis and self._recommendations:
            transform = self.axis.transData.frozen().inverted()
            origin = transform.transform((event.x, event.y))
            self._chart_drag = transform, origin, self.axis.get_xlim(), self.axis.get_ylim()
            self._chart_press = (event.x, event.y)
            self._update_chart_cursor(event)
        elif event.inaxes is self.recommendation_axis:
            self._card_press = self._card_at(event)
        elif event.inaxes is self.scroll_axis and self._legend_scroll_limit > 0:
            _, y = self.scroll_axis.transAxes.inverted().transform((event.x, event.y))
            low, high = self.scroll_thumb.get_ydata()
            self._scroll_drag = y - (low+high)/2 if low <= y <= high else 0
            self._drag(event)

    def _drag(self, event):
        if self._chart_drag is not None:
            if self._chart_press is not None:
                x, y = self._chart_press
                if (event.x-x)**2 + (event.y-y)**2 < 16:
                    return
                self._chart_press = None
            transform, origin, xlim, ylim = self._chart_drag
            dx, dy = transform.transform((event.x, event.y)) - origin
            self.axis.set_xlim(xlim[0]-dx, xlim[1]-dx)
            self.axis.set_ylim(ylim[0]-dy, ylim[1]-dy)
            self._zoomed = True
            self._layout_chart_labels()
            self._request_draw()
        elif self._scroll_drag is not None and self._legend_scroll_limit > 0:
            _, y = self.scroll_axis.transAxes.inverted().transform((event.x, event.y))
            low, high = self.scroll_thumb.get_ydata()
            thumb_height = high - low
            fraction = (1 - thumb_height / 2 - y + self._scroll_drag) / (1 - thumb_height)
            self._scroll_to(fraction * self._legend_scroll_limit)
        else:
            self._update_chart_cursor(event)

    def _end_drag(self, event):
        if event.button == 1:
            press, card = self._chart_press, self._card_press
            self._chart_press = self._chart_drag = self._scroll_drag = self._card_press = None
            if card is not None and card == self._card_at(event):
                self._selected_combination = card
                self._render_chart()
                self._render_legend()
                self._request_draw()
            if press is not None and event.inaxes is self.axis:
                x, y = press
                if (event.x-x)**2 + (event.y-y)**2 < 16:
                    self._select_chart_combination(event)
            self._update_chart_cursor(event)

    def _update_chart_cursor(self, event):
        from matplotlib.backend_tools import Cursors

        hovering = (event.name != "figure_leave_event" and self._chart_drag is None
                    and (self._chart_combination_at(event) is not None or self._card_at(event) is not None))
        if hovering != self._chart_hovering:
            self.figure.canvas.set_cursor(Cursors.HAND if hovering else Cursors.POINTER)
            self._chart_hovering = hovering

    def _card_at(self, event):
        if event.inaxes is self.recommendation_axis:
            for card, viewport, name in self._card_regions:
                if viewport.contains(event)[0] and card.contains(event)[0]:
                    return name
        return None

    def _chart_combination_at(self, event):
        if event.inaxes is not self.axis:
            return None
        candidates = []
        for point, label, name in self._chart_labels:
            x, y = self.axis.transData.transform(point)
            if not self.axis.bbox.contains(x, y):
                continue
            if label.get_visible() and label.contains(event)[0]:
                return name
            distance = (event.x-x)**2 + (event.y-y)**2
            if distance <= (self.figure.dpi * 8 / 72)**2:
                candidates.append((distance, name))
        if not candidates:
            return None
        shortest = min(distance for distance, _ in candidates)
        tied = [name for distance, name in candidates if abs(distance - shortest) < 1e-6]
        # Repeat clicks on a coincident marker cycle through every combination.
        if self._selected_combination in tied:
            return tied[(tied.index(self._selected_combination) + 1) % len(tied)]
        return tied[0]

    def _select_chart_combination(self, event):
        name = self._chart_combination_at(event)
        if name is not None:
            self._selected_combination = name
            self._render_chart()
            self._render_legend(reveal_selection=True)
            self._request_draw()

    def _reset_chart(self, _event):
        self._zoomed = False
        self._render_chart()
        self._request_draw()

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
            self._layout_chart_labels()
            self._request_draw()
            return
        if self._legend_scroll_limit and event.inaxes in (self.recommendation_axis, self.scroll_axis):
            step = 1 if event.button == "down" else -1
            self._scroll_to(self._legend_offset + .09 * step)

    def _layout_chart_labels(self):
        """Assign visible numbers to distinct screen slots nearest their point."""
        labels = self._chart_labels
        if not labels:
            return
        bbox = self.axis.bbox
        scale = self.figure.dpi / 72
        width = max(18, 8 * len(str(len(labels))) + 6) * scale
        height = (FONT_MEDIUM + 6) * scale
        slots = [(bbox.x0 + 4 + col * width, bbox.y0 + 4 + row * height)
                 for col in range(int((bbox.width - 8) / width))
                 for row in range(int((bbox.height - 8) / height))]
        points = [self.axis.transData.transform(point) for point, _, _ in labels]
        # Leave room for direct point selection, including cycling through ties.
        anchors = set(map(tuple, points))
        slots = [(x, y) for x, y in slots if not any(
            x - 3 <= px <= x + width and y - 3 <= py <= y + height for px, py in anchors)]
        for (px, py), (_, artist, _) in zip(points, labels):
            visible = bool(slots) and bbox.contains(px, py)
            artist.set_visible(visible)
            if visible:
                idx = min(range(len(slots)), key=lambda i:
                          (slots[i][0] - px - 8 * scale)**2 + (slots[i][1] - py - 8 * scale)**2)
                x, y = slots.pop(idx)
                artist.set_position(((x - px) / scale, (y - py) / scale))

    def _render_status(self):
        phase = ("Finished" if self.finished else "Finishing…" if self.finish_requested
                 else "Paused" if self._pause_active else "Pausing…" if self.paused
                 else "Initial sampling" if self._warming_up else "Searching")
        status = (phase, self.total_cells, self._last_evaluations, self._last_spent, len(self._recommendations), self._stop_reason, self._pending_batch_evaluations)
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
        if self._pending_batch_evaluations:
            detail += f" Interrupted batch: {self._pending_batch_evaluations} evaluations pending."
        self.detail_text.set_text("\n".join(wrap(detail, width=54)))
        disabled = self.finished or self.finish_requested
        for button in self.buttons:
            button.set_active(not disabled)
            button.ax.set_facecolor(EDGE if disabled else button.color)
            button.ax.patch.set_edgecolor(EDGE if disabled or button is self.finish_button else button.color)
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
        self._chart_labels.clear()
        if not has_points:
            heading = "No recommendation available" if self.finished else (
                "Collecting initial samples" if self._warming_up else "Evaluating combinations")
            self.empty_title.set_text(heading)
            progress = f"{self._last_evaluations:,} completed"
            if self._warming_up and not self.finished and self.warm_total:
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
        ax.set_ylim(min(-.03, min(scores)-.08), max(1.03, max(scores)+.08))
        ax.margins(x=.14)
        if self._zoomed:
            ax.set_xlim(view[0])
            ax.set_ylim(view[1])
        for number, (cost, score, recommendation) in enumerate(zip(costs, scores, self._recommendations), 1):
            selected = recommendation.model_name == self._selected_combination
            if selected:
                ax.scatter([cost], [score], facecolor=BLUE, edgecolor=MINT, linewidth=2,
                           s=115, zorder=4)
            label = ax.annotate(str(number), (cost, score),
                                xytext=(0, 0), textcoords="offset points", ha="left", va="bottom",
                                fontsize=FONT_MEDIUM, color=MINT if selected else TEXT,
                                weight="bold" if selected else "normal", fontfamily=self.font,
                                annotation_clip=True)
            self._chart_labels.append(((cost, score), label, recommendation.model_name))
        ax.grid(color=EDGE, linewidth=.6, alpha=.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(EDGE)
        ax.tick_params(colors=MUTED, labelsize=FONT_MEDIUM, length=0, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.3g}"))
        ax.set_xlabel("Deployment cost · USD / 1,000 queries", color=MUTED, fontsize=FONT_MEDIUM, labelpad=13, fontfamily=self.font)
        ax.set_ylabel("Estimated quality", color=MUTED, fontsize=FONT_MEDIUM, labelpad=12, fontfamily=self.font)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(self.font)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set(fontfamily=self.font, fontsize=FONT_MEDIUM, color=MUTED)

        # Tick setup above resolves the lazy autoscale limits before we use transData.
        ax.get_xlim()
        self._layout_chart_labels()

    def _render_legend(self, *, reveal_selection=False):
        from matplotlib.patches import FancyBboxPatch, Rectangle

        ax = self.recommendation_axis
        self._card_regions.clear()
        ax.clear()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((.002, .035), .996, .963, transform=ax.transAxes,
            boxstyle="round,pad=0,rounding_size=0.025", facecolor=PANEL, edgecolor=EDGE))
        ax.text(.055, .913, "Recommended combinations", color=TEXT, fontsize=15,
                va="center", weight="bold", fontfamily=self.font, transform=ax.transAxes)
        if not self._recommendations:
            self._legend_scroll_limit = self._legend_offset = 0.0
            self.scroll_thumb.set_ydata([0, 1])
            if self.finished:
                lines = ("No recommendations available", "This run ended before producing", "a recommended combination.")
            elif self._warming_up:
                lines = ("Waiting for initial samples", "Recommended model combinations", "will appear here.")
            else:
                lines = ("Evaluating model combinations", "Recommendations will appear", "when enough results are available.")
            for y, line, color in zip((.56, .51, .46), lines, (MINT, MUTED, MUTED)):
                ax.text(.5, y, line, ha="center", va="center", color=color,
                        fontsize=FONT_MEDIUM, fontfamily=self.font, transform=ax.transAxes)
            return
        from matplotlib.text import Text

        text_left = .135
        role_font_size = 10
        available_width = (.95 - text_left) * ax.bbox.width
        measure = Text(fontfamily=self.font, fontsize=role_font_size)
        measure.set_figure(self.figure)

        def role_lines_for(label):
            lines = []
            while label:
                measure.set_text(label)
                if measure.get_window_extent().width <= available_width:
                    lines.append(label)
                    break
                low, high = 1, len(label)
                while low < high:
                    mid = (low + high + 1) // 2
                    measure.set_text(label[:mid])
                    if measure.get_window_extent().width <= available_width:
                        low = mid
                    else:
                        high = mid - 1
                split = label.rfind(" ", 0, low + 1)
                end = split if split > 0 else low
                lines.append(label[:end])
                label = label[end:].lstrip()
            return lines

        cards = []
        for recommendation in self._recommendations:
            role_lines = []
            for role, model in recommendation.combo.items():
                role_lines.extend(role_lines_for(f"{role.replace('_', ' ').title()}: {model}"))
            cards.append((recommendation, role_lines, .098 + .048 * len(role_lines)))

        viewport_top, viewport_bottom = .86, .084
        viewport_height = viewport_top - viewport_bottom
        content_height = sum(height for _, _, height in cards)
        self._legend_scroll_limit = max(0.0, content_height - viewport_height)
        self._legend_offset = min(self._legend_offset, self._legend_scroll_limit)
        if reveal_selection:
            start = 0.0
            for recommendation, _, height in cards:
                if recommendation.model_name == self._selected_combination:
                    minimum = min(start, start + height - viewport_height)
                    self._legend_offset = min(self._legend_scroll_limit,
                                              max(minimum, min(start, self._legend_offset)))
                    break
                start += height
        viewport = Rectangle((.025, viewport_bottom), .947, viewport_height, transform=ax.transAxes)
        # The selected outline extends past the card box by half its stroke.
        outline_viewport = Rectangle((.025, viewport_bottom-.004), .947, viewport_height+.008,
                                     transform=ax.transAxes)

        top = viewport_top + self._legend_offset
        for index, (recommendation, role_lines, height) in enumerate(cards, 1):
            bottom = top - height
            if top <= viewport_bottom:
                break
            if bottom < viewport_top:
                selected = recommendation.model_name == self._selected_combination
                card = FancyBboxPatch((.035, bottom), .93, height-.012,
                    boxstyle="round,pad=0,rounding_size=.018",
                    facecolor="#e4f3ed" if selected else "#f3f7f5",
                    edgecolor=MINT if selected else EDGE, linewidth=1.3 if selected else .7)
                ax.add_patch(card)
                self._card_regions.append((card, viewport, recommendation.model_name))
                first_text = len(ax.texts)
                rank_y = (bottom + top - .012) / 2
                ax.text(.087, rank_y, str(index), ha="center", va="center",
                        color=BLUE, weight="bold", fontsize=12, fontfamily=self.font)
                for line_index, line in enumerate(role_lines):
                    ax.text(text_left, top-.039-.048*line_index, line, va="top", color=TEXT,
                            fontsize=role_font_size, fontfamily=self.font)
                ax.text(.135, bottom+.026,
                        f"Quality {recommendation.conservative_accuracy:.3f}  ·  ${recommendation.conservative_cost_usd*1000:.3f} / 1k  ·  {recommendation.observed_samples}/{self.total_questions} tested",
                        color=MINT, fontsize=10, fontfamily=self.font)
                for artist in (card, *ax.texts[first_text:]):
                    artist.set_clip_on(True)
                    artist.set_clip_path(outline_viewport if selected and artist is card else viewport)
            top = bottom

        if self._legend_scroll_limit:
            thumb_height = max(.08, viewport_height / content_height)
            thumb_top = 1 - (1 - thumb_height) * self._legend_offset / self._legend_scroll_limit
            self.scroll_thumb.set_ydata([thumb_top-thumb_height, thumb_top])
        else:
            self.scroll_thumb.set_ydata([0, 1])

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
        self._request_draw()

    def _request_draw(self):
        if not self._closed:
            self.figure.canvas.draw_idle()

    def _on_close(self, _event):
        if self._closed:
            return
        self._closed = True
        # Tk has already begun destroying the window. Use an off-screen canvas
        # for final snapshots/exports so no new Tk after_idle callback is queued.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self._closed_canvas = _event.canvas
        FigureCanvasAgg(self.figure)
        self.finish_search(None)

    def _release_closed_window(self):
        """Release Tk-owned image/variable objects on the UI thread, not by GC
        during a later selection worker's allocation or explicit collection.
        """
        canvas = self._closed_canvas
        if canvas is None:
            return
        if current_thread() is not main_thread():
            raise RuntimeError("Window resources must be released on the main thread")
        if hasattr(canvas, "_tkphoto"):
            canvas._tkphoto = None
            manager = canvas.manager
            if manager is not None and hasattr(manager, "_window_dpi"):
                manager._window_dpi = None
        self._closed_canvas = None

    def finish_search(self, _event):
        if self.finished or self.finish_requested:
            return
        self.finish_requested = True
        self.paused = False
        self._resume.set()
        self.stop_event.set()
        self._render_status()
        self._request_draw()

    def update(self, snapshot):
        was_warming_up = self._warming_up
        self._warming_up = snapshot.stop_reason == "warming_up"
        self._stop_reason = snapshot.stop_reason
        self._pending_batch_evaluations = getattr(snapshot, "pending_batch_evaluations", 0)
        self._last_evaluations = snapshot.total_evaluations
        self._last_spent = "Unavailable" if snapshot.selection_cost is None else f"${snapshot.selection_cost:.4f}"
        recommendations = sorted(snapshot.get_recommendations(),
            key=lambda r: (r.conservative_cost_usd, -r.conservative_accuracy, r.model_name))
        redraw_legend = (recommendations != self._recommendations or self.finished != snapshot.is_final
                         or was_warming_up != self._warming_up)
        self._recommendations = recommendations
        if not any(r.model_name == self._selected_combination for r in recommendations):
            self._selected_combination = None
        self.finished = snapshot.is_final
        if self.finished:
            self.paused = False
            self._pause_active = False
        self._render_chart()
        if redraw_legend:
            self._render_legend()
        self._render_status()
        self._request_draw()
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
                self.figure.savefig(self.output, format=self._output_format, dpi=160, facecolor=BG)
            finally:
                self.figure.set_size_inches(size, forward=False)
                self.figure.set_canvas(window_canvas)

    def pump_events(self):
        if not self._closed and self.show_window and self.plt.fignum_exists(self.figure.number):
            self.figure.canvas.flush_events()
            # flush_events can deliver close_event and destroy the Tk widgets.
            if self._closed:
                return
            previous = self._last_status
            self._render_status()
            if previous != self._last_status:
                self._request_draw()

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
        if self.output is not None:
            self.output = self.output.resolve()
        checkpoint = getattr(selector, "checkpoint_path", None)
        if self.output is not None and checkpoint is not None:
            from ..model_selection.checkpoint import checkpoint_paths, require_distinct_paths
            require_distinct_paths({"plot output": self.output, **checkpoint_paths(checkpoint)})
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
            self._release_closed_window()
            raise
        worker.join()
        self._release_closed_window()
        if "error" in outcome:
            raise outcome["error"]
        return outcome["results"]

    def wait_for_close(self):
        if not self._closed and self.show_window and self.plt.fignum_exists(self.figure.number):
            self.plt.ioff()
            self.plt.show()
        self._release_closed_window()
