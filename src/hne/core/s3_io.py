"""
src/core/s3_io.py 
"""
import boto3
import io
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
        
    def read_tif_as_openslide(self, 
                              s3_path: str) -> tuple[openslide.OpenSlide, str]:
        """Download tif to a temp file an open it as OpenSlide for patching module."""
        data_bytes = self._read_bytes(s3_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(data_bytes)
        tmp.flush()
        tmp.close()
        slide = openslide.OpenSlide(tmp.name)
        return slide, tmp.name  # return the path too, so caller can delete it when done
