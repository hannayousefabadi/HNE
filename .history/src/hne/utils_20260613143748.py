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
    con
    
    """