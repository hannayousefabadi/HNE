import logging 
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: str = "WARNING",
    file_level: str = "DEBUG",
    log_format: str = '%(asctimes)s - %(name)s - %(levelname)s - %(message)s',
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

    root_level = min(level_map[console_level], level_map[file_level])   # set root logger to the lower level (more detailed)
    root_logger.setLevel(root_level)

    formatter = logging.Formatter(log_format, date_format)

    # adding console handler
    console_handler = logging.StreamHandler(sys.stdout)     # sneds log messages to the console
    console_handler.setLevel(level_map[console_level])
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # adding file_handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level_map[file_level])
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # silent other third-party loggers
    for lib in ['matplotlib', 'PIL', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.WARNING)




def get_logger(name: str = None):
    """
    Get a logger instance for a module
    
    Args:
        name: usually __name__ from the calling module
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back



 class LoggerMixin:
    """Mixin class to add logging capability to any class"""

    @property
    def logger(self):
        if not hasattr(Self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__module__ + '.' +
                                             self.__class__.__name__)
            return self._logger