"""Progress displays must close on failures without hiding the original error."""

from threading import enumerate as threads

import pytest
from osa_anurans_app.src import progress


@pytest.mark.parametrize("fail", [False, True])
def test_activity_closes_display_and_stops_worker(monkeypatch, fail):
    class Bar:
        n = 0
        closed = False

        def update(self, count):
            self.n += count

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    bar = Bar()
    monkeypatch.setattr(progress, "tqdm", lambda **kwargs: bar)
    failure = RuntimeError("upload failed")
    try:
        with progress.activity("Uploading clips", total=2, unit="clip") as display:
            display.update(1)
            if fail:
                raise failure
            display.update(1)
    except RuntimeError as error:
        assert fail and error is failure
    assert bar.closed
    assert bar.n == (1 if fail else 2)
    assert not any(thread.name == "cloud-progress" for thread in threads())
