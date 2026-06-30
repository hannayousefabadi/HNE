"""
s3 i/o utility - read only from S3, save results to repo
"""


import boto3
import io
import json
import pandas as pd
import scanpy as sc
from PIL import Image
import tifffile
from typing import Tuple, List, Optional
import tempfile
from io import BytesIO


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
        bucket, prefix = s3_path.split("/", 1)
        return bucket, prefix

    def _read_bytes(self, s3_path: str) -> bytes:
        """Read files from s3 as bytes"""
        bucket, prefix = self._parse_s3_path(s3_path)  
        response = self.s3_client.get_object(Bucket=bucket, Key=prefix)
        return response["Body"].read()
    
    def read_h5ad(self, s3_path: str) -> sc.AnnData:
        """Read AnnData directly from S3"""
        logger.debug(f"Reading AnnData from: {s3_path}")
        data_bytes = self._read_bytes(s3_path)

        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=True) as tmp:
            tmp.write(data_bytes)
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

    def list_patients_from_processed(self, bucket: str, prefix: str) -> List[str]:
        """
        Discover patient IDs from PROCESSED data folders.
        Looks for folders like CH_L_XXXa_vis/
        """
        logger.info(f"Discovering patients from processed data: s3://{bucket}/{prefix}")
        
        patients = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        # Use delimiter to only get top-level folders
        for page in paginator.paginate(
            Bucket=bucket, 
            Prefix=prefix,
            Delimiter='/'
        ):
            if 'CommonPrefixes' in page:
                for common in page['CommonPrefixes']:
                    folder = common['Prefix']
                    # Extract patient ID from folder name
                    # Example: "CH_L_403a_vis/" → "CH_L_403a_vis"
                    patient_id = folder.rstrip('/')
                    # Keep the full folder name (with _vis)
                    patients.append(patient_id)
        
        return sorted(patients)

    def list_patients_from_raw(self, bucket: str, prefix: str, pattern: str = "CH_L_") -> List[str]:
        """
        Discover patient IDs from RAW image files.
        Looks for TIF files and extracts patient IDs from filenames.
        """
        logger.info(f"Discovering patients from raw images: s3://{bucket}/{prefix}")
        
        patients = set()
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.tif'):
                        filename = key.split('/')[-1]
                        parts = filename.split('_')
                        
                        for part in parts:
                            if part.startswith(pattern):
                                # Clean up patient ID
                                patient_id = part
                                # Remove any trailing stuff
                                if patient_id.endswith('.tif'):
                                    patient_id = patient_id[:-4]
                                patients.add(patient_id)
                                break
        
        return sorted(list(patients))

    def find_tif_for_patient(self, bucket: str, prefix: str, patient_id: str) -> Optional[str]:
        """
        Find TIF file for a patient.
        The patient_id can be with or without _vis suffix.
        """
        # Clean patient ID (remove _vis if present)
        clean_id = patient_id.replace('_vis', '')
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        prefix = f"{prefix}/image_files/" if not prefix.endswith('image_files') else prefix
        
        # Search for TIF files
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    
                    # Check if this TIF belongs to this patient
                    # Look for the patient ID in the filename
                    if clean_id in filename and key.endswith('.tif'):
                        logger.info(f"Found TIF for {patient_id}: {filename}")
                        return key
        
        logger.warning(f"No TIF found for patient {patient_id}")
        return None            
    
    
