from time import perf_counter
from typing import Callable
import winsound


class Timer:
    # context manager
    def __init__(self, beep: bool = True, printer: Callable = print):
        self.start = None
        self.beep = beep
        self.printer = printer

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, *args):
        msg = f'Executed in {(perf_counter() - self.start):.1f} seconds'
        self.printer(msg)
        if self.beep:
            winsound.PlaySound('SystemDefault', winsound.SND_ALIAS)
            # winsound.PlaySound(MY_DIRECTORY + '/smw_power_up.wav', winsound.SND_FILENAME)


if __name__ == '__main__':
    import time
    with Timer() as t:
        print("first: sleeping 1 second")
        time.sleep(1)

    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    with Timer(printer=log.warning):
        log.info("second: sleeping 1 second")
        time.sleep(1)
