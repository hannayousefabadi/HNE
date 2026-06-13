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
    







def get_logger(
        
):
