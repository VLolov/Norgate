import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

# https://stackoverflow.com/questions/20333674/pycharm-logging-output-colours
# https://xsnippet.org/359377/
#
# activate ANSI colors in cmd
# FYI, in latest Windows 10, you can enable ANSI in conhost via the following reghack --
# in HKCU\Console create a DWORD named VirtualTerminalLevel and set it to 0x1; then restart cmd.exe.
# -- You can test it with the following powershell "?[1;31mele ?[32mct ?[33mroni ?[35mX ?[36mtar ?[m".Replace('?', [char]27);
#

_COLOR_HANDLER = True   # False: to get the 'standard' python handler


class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    # normal colors
    _colors = dict(black=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37)
    # bright colors
    # _colors = dict(black=30, red=91, green=92, yellow=93, blue=94, magenta=95, cyan=96, white=97)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except Exception:
                raise

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%s;1m%s\x1b[0m' % (color, text))


class ColorHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stderr):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.CRITICAL: "magenta",
            logging.FATAL: "magenta",
            logging.DEBUG: "green",
            logging.INFO: "blue",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
        }
        color = msg_colors.get(record.levelno, "blue")
        # self.stream.write(record.msg + "\n", color)
        self.stream.write(self.format(record) + "\n", color)


def logger(level=logging.INFO, logfile=None, console=True):
    """ Create and configure file and console logging.
    :param level: console debugging level only.
    :param logfile: log destination file name
    :param console: boolean, true= don't log on console
    :return: configured logging object
    """
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    format = "%(asctime)s - %(name)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(level)
    if console:
        # These line is needed if 'Emulate terminal in output console' is not checked (Run/Edit Configurations/Templates/Python)
        # is_running_in_pycharm = "PYCHARM_HOSTED" in os.environ
        # console_handler = ColorHandler() if _COLOR_HANDLER and is_running_in_pycharm else logging.StreamHandler()

        # With ANSI emulation (this gives colors also when running in CMD.com):
        console_handler = ColorHandler() if _COLOR_HANDLER else logging.StreamHandler()

        console_formatter = logging.Formatter(format)
        console_handler.setFormatter(console_formatter)
        logger.removeHandler(console_handler)
        logger.addHandler(console_handler)

    if logfile is not None:
        # new log file every day
        directory = os.path.dirname(logfile)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_handler = TimedRotatingFileHandler(logfile, when="midnight", interval=1)
        file_handler.suffix = "%Y_%m_%d"
        # file_handler = logging.FileHandler(logfile) # one file for all logs

        def my_namer(default_name):
            # see: https://stackoverflow.com/questions/338450/timedrotatingfilehandler-changing-file-name
            # This will be called when doing the log rotation
            # default_name is the default filename that would be assigned, e.g. Rotate_Test.txt.YYYY-MM-DD
            # Do any manipulations to that name here, for example this changes the name to Rotate_Test.YYYY-MM-DD.txt
            base_filename, ext, date = default_name.split(".")
            return f"{base_filename}.{date}.{ext}"

        file_handler.namer = my_namer

        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format)
        file_handler.setFormatter(file_formatter)
        logger.removeHandler(file_handler)
        logger.addHandler(file_handler)
    return logger


# class MyLog():
#     def __init__(self, log: logging.Logger):
#         self.log = log
#
#     def info(self, msg):
#         self.log.info(f">{msg}")
#
#     def warning(self, msg):
#         self.log.warning(f">{msg}")

if __name__ == "__main__":
    # log = MyLog(logging.getLogger(__name__))
    log = logging.getLogger(__name__)
    logger(level=logging.DEBUG, logfile='c:/tmp/example_logfile.log', console=True)

    log.fatal("Some fatal output")
    log.critical("Some critical output")
    log.debug("Some debugging output")
    log.info("Some info output")
    log.error("Some error output")
    log.warning("Some warning output")


