"""
src/core/s3_io.py 
"""
import boto3
import io
import os
import json
import pandas as pd
import scanpy as sc
from PIL import Image
import tifffile
from typing import Tuple
import tempfile
from io import BytesIO
import openslide
import tempfile
import pyvips
from contextlib import contextmanager

from hne.utils import get_logger

logger = get_logger()

class S3DataLoader:
    """Load data directly from S3 using boto3 and IAM role."""

    def __init__(self):
        self.s3_client = boto3.client('s3')    

    def _parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse s3://bucker/prefix into bucket, prefix""" 
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        if '/' in s3_path:
            bucket, prefix = s3_path.split("/", 1)
        return bucket, prefix

    def _read_bytes(self, s3_path: str) -> bytes:
        """Read files from s3 as bytes"""
        bucket, prefix = self._parse_s3_path(s3_path)  
        response = self.s3_client.get_object(Bucket=bucket, Key=prefix)
        return response["Body"].read()

    def read_h5ad(self, s3_path: str) -> sc.AnnData:
        """Read AnnData directly from S3 without loading the whole file into RAM."""

        bucket, prefix = self._parse_s3_path(s3_path)

        response = self.s3_client.get_object(Bucket=bucket, Key=prefix)

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=True) as tmp:
            while True:
                chunk = response["Body"].read(8 * 1024 * 1024)  # 8 MB
                if not chunk:
                    break
                tmp.write(chunk)

            tmp.flush()
            return sc.read_h5ad(tmp.name)
        
    def read_csv(self, s3_path: str, **kwargs) -> pd.DataFrame:
        "Read CSV directly from s3"  
        data_bytes = self._read_bytes(s3_path)
        return pd.read_csv(io.BytesIO(data_bytes), **kwargs)
    
    def read_json(self, s3_path: str) -> dict:
        """Read JSON directly from s3"""
        data_bytes = self._read_bytes(s3_path)
        return json.loads(data_bytes.decode('utf-8'))
    
    def read_tif(self, s3_path: str) -> Image.Image:
        """Read TIF image directly from S3"""
        data_bytes = self._read_bytes(s3_path)
        
        with BytesIO(data_bytes) as bio:
            img_array = tifffile.imread(bio)

        if img_array.ndim == 3:
            return Image.fromarray(img_array)
        else:
            return Image.fromarray(img_array, mode='L')
        
    @contextmanager
    def open_tif_as_openslide(self, s3_path: str):
        """
        Stream TIFF from S3 directly to disk, convert to pyramidal TIFF, 
        and strictly clean up all temporary files on completion.
        """
        bucket, prefix = self._parse_s3_path(s3_path)
        
        # Suppress pyvips internal memory caching to prevent RAM leaks across slides
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_mem(0)

        # Allocate file paths for streaming
        raw_tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        raw_tmp_path = raw_tmp.name
        raw_tmp.close()

        pyramid_tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        pyramid_tmp_path = pyramid_tmp.name
        pyramid_tmp.close()

        slide = None
        try:
            # Stream directly to disk: ZERO memory allocation in Python RAM
            self.s3_client.download_file(bucket, prefix, raw_tmp_path)

            # Pyramidal conversion via pyvips
            image = pyvips.Image.new_from_file(raw_tmp_path, access="sequential")
            image.tiffsave(
                pyramid_tmp_path,
                tile=True,
                pyramid=True,
                compression="jpeg",
                Q=90,
                bigtiff=True,
                tile_width=256,
                tile_height=256,
            )
            del image

            # Remove raw flat file immediately to free disk space
            if os.path.exists(raw_tmp_path):
                os.unlink(raw_tmp_path)

            slide = openslide.OpenSlide(pyramid_tmp_path)
            yield slide

        finally:
            if slide is not None:
                slide.close()
                del slide
            if os.path.exists(raw_tmp_path):
                os.unlink(raw_tmp_path)
            if os.path.exists(pyramid_tmp_path):
                os.unlink(pyramid_tmp_path)