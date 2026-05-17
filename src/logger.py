import logging
from enum import Enum

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

class Color(Enum):
    RESET = "[0m"
    RED = "[31m"
    GREEN = "[32m"
    YELLOW = "[33m"
    BLUE = "[34m"
    MAGENTA = "[35m"
    CYAN = "[36m"
    BRED = "[91m"
    BGREEN = "[92m"
    BYELLOW = "[93m"
    BBLUE = "[94m"
    BMAGENTA = "[95m"
    BCYAN = "[96m"

class ColorLogger(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log messages."""

    @staticmethod
    def log(color: Color, message: str) -> str:
        return f"{color.value}{message}{Color.RESET.value}"
    
    LEVEL_NAMES = {
        logging.DEBUG: "DEBUG:",
        logging.INFO: "INFO :",
        logging.WARNING: "WARN :",
        logging.ERROR: "ERROR:",
        logging.CRITICAL: "CRIT :",
    }

    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = self.LEVEL_NAMES.get(record.levelno, record.levelname[:5].ljust(5))
        if record.levelno >= logging.ERROR:
            record.levelname = ColorLogger.log(Color.RED, record.levelname)
            record.msg = ColorLogger.log(Color.RED, record.msg)
        elif record.levelno >= logging.WARNING:
            record.levelname = ColorLogger.log(Color.YELLOW, record.levelname)
            record.msg = ColorLogger.log(Color.YELLOW, record.msg)
        return super().format(record)


def set_log_level(level: str) -> None:
    logging.getLogger().setLevel(_LEVELS.get(level.lower(), logging.INFO))

def setup_logging(level: str = "info") -> None:
    log_level = _LEVELS.get(level.lower(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorLogger(
        fmt='%(levelname)s %(message)s'
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)

def cprint(color: str, message: str, end: str = "\n", flush: bool = False) -> None:
    color_codes = {
        "rd": Color.RED.value,
        "gn": Color.GREEN.value,
        "yl": Color.YELLOW.value,
        "bl": Color.BLUE.value,
        "mg": Color.MAGENTA.value,
        "cn": Color.CYAN.value,
        "brd": Color.BRED.value,
        "bgn": Color.BGREEN.value,
        "byl": Color.BYELLOW.value,
        "bbl": Color.BBLUE.value,
        "bmg": Color.BMAGENTA.value,
        "bcn": Color.BCYAN.value,
    }
    print(f"{color_codes.get(color.lower(), '')}{message}{Color.RESET.value}",
          end=end,
          flush=flush)

def test_colors():
    cprint("rd", "Test red log")
    cprint("gn", "Test green log")
    cprint("yl", "Test yellow log")
    cprint("bl", "Test blue log")
    cprint("mg", "Test magenta log")
    cprint("cn", "Test cyan log")
    cprint("brd", "Test bright red log")
    cprint("bgn", "Test bright green log")
    cprint("byl", "Test bright yellow log")
    cprint("bbl", "Test bright blue log")
    cprint("bmg", "Test bright magenta log")
    cprint("bcn", "Test bright cyan log")
