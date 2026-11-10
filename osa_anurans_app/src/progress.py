"""Show counts when known and elapsed activity while cloud requests block."""

import logging
import sys
from contextlib import contextmanager
from threading import Event, Thread

from tqdm import tqdm

logger = logging.getLogger(__name__)


@contextmanager
def activity(label: str, *, total: int | None = None, unit: str = "item"):
    """Keep the terminal responsive without inventing cloud-job percentages.

    - Count completed items when callers update the returned progress bar.
    - Refresh elapsed time during blocking requests; log heartbeats when redirected.
    - Stop the display on success, failure, or interruption.
    """
    logger.info("%s...", label)
    terminal = sys.stderr.isatty()
    options = {} if total is not None else {"bar_format": "{desc} | elapsed {elapsed}"}
    bar = tqdm(
        total=total, desc=label, unit=unit, dynamic_ncols=True, disable=not terminal, **options
    )
    stopped = Event()

    def refresh():
        interval = 1 if terminal else 15
        while not stopped.wait(interval):
            if terminal:
                bar.refresh()
            else:
                logger.info("%s: still working...", label)

    worker = Thread(target=refresh, name="cloud-progress", daemon=True)
    worker.start()
    try:
        yield bar
    finally:
        stopped.set()
        worker.join()
        bar.close()
