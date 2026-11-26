import sqlite3

import pytest

from agentopt.model_selection.checkpoint import CheckpointError
from tests.test_gittins import make_selector


@pytest.mark.parametrize("target_suffix", [".pdf", ".bin"])
def test_symlink_export_keeps_requested_format(tmp_path, target_suffix):
    from PIL import Image
    from agentopt.visualization import LiveParetoPlot

    target = tmp_path / ("actual" + target_suffix)
    target.touch()
    alias = tmp_path / "alias.png"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("file symlinks require Windows developer mode or privilege")
        raise
    selector, _ = make_selector(count=2)
    plot = LiveParetoPlot(total_questions=2, output=alias, show_window=False)
    try:
        plot.run(selector)
        with Image.open(target) as image:
            image.load()
            assert image.format == "PNG"
    finally:
        plot.plt.close(plot.figure)


def test_failed_late_batch_replay_preserves_original_budget(tmp_path):
    path = tmp_path / "run.db"
    options = dict(count=2, checkpoint_path=path, checkpoint_id="workflow")
    selector, _ = make_selector(**options, max_search_cost_usd=1)
    selector.select_best(on_update=lambda snapshot: snapshot.total_evaluations < 3)
    with sqlite3.connect(path) as db:
        # Keep the earlier prefix valid. This arm exists but was never evaluated.
        assert db.execute("SELECT COUNT(*) FROM batches WHERE arm='solver=dominated'").fetchone() == (0,)
        db.execute("UPDATE batches SET arm='solver=dominated' WHERE id=(SELECT MAX(id) FROM batches)")
    before = path.read_bytes()
    restored, tracker = make_selector(**options)
    with pytest.raises(CheckpointError, match="evaluation order differs"):
        restored.select_best()
    assert not tracker.records
    assert path.read_bytes() == before


@pytest.mark.parametrize("retry_stop", [False, True])
@pytest.mark.parametrize("has_record", [False, True])
def test_failed_session_archives_only_recorded_work(monkeypatch, retry_stop, has_record):
    from unittest.mock import sentinel
    from agentopt.proxy import _local_backend

    backend = _local_backend.LocalBackend(cache=False)
    manager = backend._session_manager

    class FailedMaster:
        def __init__(self, *, session, **kwargs):
            self.session = session
            self.retry = retry_stop

        def start(self):
            raise RuntimeError("startup failed")

        def stop(self):
            if self.retry:
                self.retry = False
                raise RuntimeError("stop needs retry")
            if has_record:
                manager.add_record(self.session.session_id, sentinel.record)

    monkeypatch.setattr(_local_backend, "SessionMaster", FailedMaster)
    backend.start()
    try:
        with pytest.raises(RuntimeError, match="startup failed"):
            backend.open_session()
        assert len(manager._active) == int(retry_stop)
        assert len(backend._masters) == int(retry_stop)
    finally:
        backend.stop()
    assert not manager._active and not backend._masters
    assert len(manager._ended) == int(has_record)
    assert manager.get_all_records() == ([sentinel.record] if has_record else [])
