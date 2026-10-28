"""Optional live Pareto view for a real AgentOpt multi-objective search."""

from __future__ import annotations

from queue import Empty, Queue
from pathlib import Path
from threading import Event, Thread, current_thread, main_thread
from textwrap import wrap


class LiveParetoPlot:
    """Display selector snapshots and control a search without blocking the GUI.

    Construct this on the main thread, then call :meth:`run` with a Gittins
    multi-objective selector built from the user's own agent and dataset.
    Matplotlib is imported only when the view is constructed.
    """

    def __init__(self, *, total_questions: int, warm_total: int | None = None,
                 title: str = "Estimated Pareto frontier", output: Path | None = None,
                 show_window: bool = True, stop_event: Event | None = None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
            from matplotlib.widgets import Button
        except ImportError as exc:
            raise RuntimeError("Live plotting requires matplotlib: pip install agentopt-py[plot]") from exc

        if show_window and plt.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
            raise RuntimeError("--live-plot needs an interactive Matplotlib backend; use --plot-output for a file")

        self.plt = plt
        self.money_axis = FuncFormatter(lambda value, _: f"${value:.4f}")
        if total_questions < 1 or (warm_total is not None and warm_total < 1):
            raise ValueError("total_questions and warm_total must be positive")
        self.title = title
        self.warm_total = warm_total
        self.total_questions = total_questions
        self.output = Path(output) if output is not None else None
        self.show_window = show_window
        self.stop_event = stop_event if stop_event is not None else Event()
        self._resume = Event()
        self._resume.set()
        self.paused = False
        self._pause_active = False
        self.finished = False
        self.finish_requested = False
        self._recommendations = []
        self._legend_offset = 0
        self._last_evaluations = 0
        self._last_spent = "unavailable"
        self.figure = plt.figure(figsize=(10, 6))
        self.axis = self.figure.add_axes((0.10, 0.17, 0.45, 0.69))
        self.legend_text = self.figure.text(0.59, 0.84, "", va="top", fontsize=9)
        self.warm_axis = self.figure.add_axes((0.59, 0.72, 0.32, 0.035))
        self.warm_axis.set_xlim(0, 1)
        self.warm_axis.set_ylim(0, 1)
        self.warm_axis.set_xticks([])
        self.warm_axis.set_yticks([])
        self.warm_axis.set_facecolor("#e5e7eb")
        self.warm_fill = self.warm_axis.barh(0.5, 0, height=1, color="#2563eb")[0]
        self.status_text = self.figure.text(0.10, 0.07, "", fontsize=10)
        self.note_text = self.figure.text(
            0.10, 0.025, "Estimates for the supplied benchmark; final scores may differ.",
            fontsize=9, color="#555555",
        )
        self.title_text = self.figure.suptitle(
            self.title, fontsize=15,
        )
        self.figure.canvas.mpl_connect("resize_event", self._on_resize)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("close_event", self.finish_search)
        if show_window:
            self.finish_button = Button(self.figure.add_axes((0.63, 0.055, 0.14, 0.07)),
                                        "Finish")
            self.finish_button.on_clicked(self.finish_search)
            self.pause_button = Button(self.figure.add_axes((0.78, 0.055, 0.13, 0.07)), "Pause")
            self.pause_button.on_clicked(self.toggle_pause)
            plt.ion()
            plt.show(block=False)
        self._layout()
        self.figure.canvas.draw_idle()

    def _on_resize(self, _event):
        self._layout()
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event):
        width, _ = self.figure.canvas.get_width_height()
        if event.x < self.legend_text.get_position()[0] * width - 15:
            return
        if event.button == "down":
            self._legend_offset = min(self._legend_offset + 1, max(0, len(self._recommendations) - 1))
        elif event.button == "up":
            self._legend_offset = max(0, self._legend_offset - 1)
        self._render_legend()
        self.figure.canvas.draw_idle()

    def _layout(self):
        width, height = self.figure.get_size_inches()
        compact = width < 9
        tiny = width < 6.5 or height < 4.5
        self.axis.set_position((0.12 if compact else 0.10,
                                0.29 if tiny else 0.20 if compact else 0.17,
                                0.43 if compact else 0.45,
                                0.55 if tiny else 0.64 if compact else 0.69))
        self.axis.set_xlabel("USD / question" if tiny else "Conservative USD / question" if compact
                             else "Estimated USD / question (conservative)")
        self.axis.set_ylabel("Score" if tiny else "Conservative score" if compact
                             else "Estimated score (conservative)")
        self.axis.tick_params(labelsize=7 if tiny else 8 if compact else 10)
        self.legend_text.set_position((0.59, 0.84))
        self.legend_text.set_fontsize(max(7, min(9, width * 0.95, height * 1.5)))
        self.title_text.set_fontsize(12 if tiny else 15)
        self.note_text.set_visible(not tiny)
        self.status_text.set_fontsize(8 if tiny else 10)
        self._render_status()
        self._render_legend()

    def _render_status(self):
        width, height = self.figure.get_size_inches()
        tiny = width < 6.5 or height < 4.5
        phase = ("Finished" if self.finished else "Finishing" if self.finish_requested
                 else "Paused" if self._pause_active else "Pausing after batch" if self.paused
                 else "Warm start" if not self._recommendations
                 else "Searching")
        if tiny:
            status = f"{phase} | {self._last_evaluations} cells | {self._last_spent} spent"
        else:
            status = (f"{phase}  |  {self._last_evaluations} evaluated cells  |  "
                      f"search cost {self._last_spent}")
        self.status_text.set_text(status)

    def _render_legend(self):
        if not self._recommendations:
            progress = (str(self._last_evaluations) if self.warm_total is None else
                        f"{self._last_evaluations}/{self.warm_total}")
            self.legend_text.set_text(f"Warm start: {progress} evaluations")
            self.warm_axis.set_visible(self.warm_total is not None)
            if self.warm_total is not None:
                self.warm_fill.set_width(min(1, self._last_evaluations / self.warm_total))
            return

        self.warm_axis.set_visible(False)

        width_inches, height_inches = self.figure.get_size_inches()
        font_points = self.legend_text.get_fontsize()
        panel_points = width_inches * 72 * (0.98 - self.legend_text.get_position()[0])
        # Leave room for proportional glyph widths and figure margins.
        chars = max(16, int(panel_points / (font_points * 0.68)))
        metric_chars = max(16, int(panel_points / (font_points * 0.52)))
        available_lines = max(4, int(height_inches * 72 * 0.64 / (font_points * 1.35)))
        start = min(self._legend_offset, len(self._recommendations) - 1)
        entries = []
        used_lines = 2
        for number, recommendation in enumerate(self._recommendations[start:], start + 1):
            combo = " + ".join(f"{node}={model}" for node, model in recommendation.combo.items())
            metrics = (f"score={recommendation.conservative_accuracy:.4f} | "
                       f"cost=${recommendation.conservative_cost_usd:.6f} | "
                       f"evaluated={recommendation.observed_samples}/{self.total_questions}")
            lines = (wrap(f"{number}. {combo}", width=chars, subsequent_indent="   ",
                          break_on_hyphens=False)
                     + wrap(metrics, width=metric_chars, initial_indent="   ",
                            subsequent_indent="   "))
            if entries and used_lines + len(lines) + 1 > available_lines:
                break
            if used_lines + len(lines) > available_lines:
                lines = lines[:max(1, available_lines - used_lines - 1)] + ["   …"]
            entries.append("\n".join(lines))
            used_lines += len(lines) + 1

        end = start + len(entries)
        header = f"Pareto combinations {start + 1}-{end}/{len(self._recommendations)}"
        if start or end < len(self._recommendations):
            header += " (scroll here)"
        self.legend_text.set_text(header + "\n\n" + "\n\n".join(entries))

    def toggle_pause(self, _event):
        """Pause at the next selector update, or continue the same run."""
        if self.finished or self.finish_requested:
            return
        self.paused = not self.paused
        if self.paused:
            self._resume.clear()
        else:
            self._resume.set()
        if self.show_window:
            self.pause_button.label.set_text("Continue" if self.paused else "Pause")
        self._render_status()
        self.figure.canvas.draw_idle()

    def finish_search(self, _event):
        """Request a final snapshot and let the caller save its results."""
        if self.finished or self.finish_requested:
            return
        self.finish_requested = True
        self.paused = False
        self._resume.set()
        if self.show_window:
            self.pause_button.label.set_text("Pause")
        self._render_status()
        self.stop_event.set()
        self.figure.canvas.draw_idle()

    def _process_window(self, snapshot):
        if not self.show_window:
            return
        if not self.plt.fignum_exists(self.figure.number):
            self.paused = False
            self._resume.set()
            return
        self.figure.canvas.draw_idle()
        # pyplot.pause() calls show() on every update, repeatedly raising the window.
        # Pump this figure's events without asking the window manager to show it again.
        self.figure.canvas.flush_events()
        self._render_status()

    def update(self, snapshot):
        # There is no frontier before every combination has its warm samples.
        if (snapshot.stop_reason == "warming_up" and not snapshot.is_final
                and not snapshot.get_recommendations()):
            self._last_evaluations = snapshot.total_evaluations
            self._last_spent = ("unavailable" if snapshot.selection_cost is None
                                else f"${snapshot.selection_cost:.4f}")
            self._render_status()
            self._render_legend()
            self._process_window(snapshot)
            return

        figure = self.figure
        axis = self.axis
        axis.clear()
        axis.xaxis.set_major_formatter(self.money_axis)
        axis.grid(alpha=0.2)

        recommendations = sorted(
            snapshot.get_recommendations(),
            key=lambda r: (r.conservative_cost_usd, -r.conservative_accuracy, r.model_name),
        )
        self._recommendations = recommendations
        self._last_evaluations = snapshot.total_evaluations
        self._legend_offset = min(self._legend_offset, max(0, len(recommendations) - 1))
        if recommendations:
            costs = [r.conservative_cost_usd for r in recommendations]
            scores = [r.conservative_accuracy for r in recommendations]
            axis.plot(costs, scores, color="#2563eb", linewidth=1.5, alpha=0.7)
            axis.scatter(costs, scores, color="#2563eb", s=70, zorder=3)
            for number, (cost, score) in enumerate(zip(costs, scores), 1):
                axis.annotate(str(number), (cost, score), xytext=(5, 5),
                              textcoords="offset points", fontsize=9)
            axis.set_ylim(min(-0.05, min(scores) - 0.08), max(1.05, max(scores) + 0.08))
        else:
            axis.text(0.5, 0.5, "No recommendation yet", transform=axis.transAxes,
                      ha="center", va="center", fontsize=12)
        self._layout()

        spent = "unavailable" if snapshot.selection_cost is None else f"${snapshot.selection_cost:.4f}"
        self._last_spent = spent
        if snapshot.is_final:
            self.finished = True
            if self.show_window:
                self.finish_button.ax.set_visible(False)
                self.pause_button.ax.set_visible(False)
        self._render_status()
        self._process_window(snapshot)
        if snapshot.is_final and self.output is not None:
            figure.savefig(self.output, dpi=160)

    def pump_events(self):
        """Keep the window responsive while real API calls are in flight."""
        if self.show_window and self.plt.fignum_exists(self.figure.number):
            self.figure.canvas.flush_events()
            self._render_status()
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
        updates = Queue()
        outcome = {}

        def receive(snapshot):
            updates.put(snapshot)
            if not self._resume.is_set():
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
                    snapshot = updates.get(timeout=0.05)
                except Empty:
                    self.pump_events()
                    continue
                if on_update is not None:
                    on_update(snapshot)
                self.update(snapshot)
                self.pump_events()
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
