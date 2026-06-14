import logging 
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
):
    
    """
    Configure logging for the whole package

    Args:
        log_file: path to log file, if none, file logging is disabled
        console_level: log level for console
        file_lebel: log level for log file
        log_format: string format of the log message
        date_format: date format for timestamp
    
    """

    root_logger = logging.getLogger()   # get root logger (no name)
    root_logger.handlers.clear()        # remove previous handlers so they don't accumulate

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    console_num = level_map.get(console_level.upper(), logging.INFO)
    file_num = level_map.get(file_level.upper(), logging.DEBUG)

    # set root logger to the lower level (more detailed)
    root_level = min(console_num, file_num) if log_file else console_num   
    root_logger.setLevel(root_level)

    formatter = logging.Formatter(log_format, date_format)

    # adding console handler
    console_handler = logging.StreamHandler(sys.stdout)     # sneds log messages to the console
    console_handler.setLevel(console_num)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # adding file_handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_num)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # silent other third-party loggers
    for lib in ['matplotlib', 'PIL', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str = None):
    """
    Get a logger instance for a module

    Usage:
        in any module. just do:
            from hne.utils import get_logger
            logger = get_logger(__name__)
        
        or simply:
            logger = get_logger() # auto-detects __name__
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'hne')
    return logging.getLogger(name)    

class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger        