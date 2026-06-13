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

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    root_level = min(level_map[console_level], level_map[file_level])
    root_logger.setLevel(root_level)


    formatter = logging.Formatter(log_format, date_format)

    # adding console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_map[console_level])
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # silent other third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)




def get_logger(
        name: str = None
        
):
    """
    Get a logger instance for a module

    Args:
        name: usually __name__ from the calling module

    Returns:
        configured logger instance    
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

        
                
