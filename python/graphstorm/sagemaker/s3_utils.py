"""
S3 utilities for SageMaker jobs.
"""
import logging
import os
from typing import List

import boto3
from sagemaker.s3 import S3Downloader, S3Uploader
from sagemaker.session import Session

def download_data_from_s3(input_s3, local_data_path, sagemaker_session=None):
    """ Download  data info from S3

    Parameters
    ----------
    input_s3: str
        S3 prefix of the input file(s)
    local_data_path: str
        Local file path under which to store the downloaded file(s).
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    logging.debug("download %s into %s", input_s3, local_data_path)
    if not sagemaker_session:
        boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
        sagemaker_session = Session(boto_session)
    try:
        S3Downloader.download(input_s3,
            local_data_path, sagemaker_session=sagemaker_session)
    except Exception as e: # pylint: disable=broad-except
        raise RuntimeError(f"Error while downloading {input_s3}: {e}")

def upload_file_to_s3(s3_data_path, local_data_path, sagemaker_session=None):
    """ Upload local data to S3 prefix

    Parameters
    ----------
    s3_data_path: str
        S3 path prefix for data upload
    local_data_path: str
        Path to local data that we will upload. Filenames will be appended
        to S3 prefix during upload
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run upload
    """
    s3_data_path = s3_data_path[:-1] if s3_data_path.endswith('/') else s3_data_path
    logging.debug("upload %s into %s", local_data_path, s3_data_path)
    if not sagemaker_session:
        boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
        sagemaker_session = Session(boto_session)
    try:
        ret = S3Uploader.upload(local_data_path, s3_data_path,
            sagemaker_session=sagemaker_session)
    except Exception as e: # pylint: disable=broad-except
        raise RuntimeError(f"Error uploading data to {s3_data_path}: {e}")
    return ret

def list_s3_files(s3_file_path, sagemaker_session=None) -> List[str]:
    """ List files in S3 under a given prefix

    Parameters
    ----------
    s3_file_path: str
        S3 URI to list files (e.g. s3://bucket/prefix/path)
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run file listing

    Returns
    -------
    List[str]
        List of S3 URIs for all files under the given prefix

    Raises
    ------
    RuntimeError
        If there's an error parsing the S3 URI or listing files
    ValueError
        If the S3 URI is malformed
    """
    if not s3_file_path.startswith(('s3://', 's3n://', 's3a://')):
        raise ValueError(f"Invalid S3 URI format: {s3_file_path}. "
                         "Must start with s3://, s3n:// or s3a://")

    # Remove trailing slash and protocol prefix
    s3_file_path = s3_file_path.rstrip('/')
    path_without_protocol = s3_file_path.split('://', 1)[1]

    # Split into bucket and prefix
    parts = path_without_protocol.split('/', 1)
    bucket = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ''

    if not sagemaker_session:
        boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
        sagemaker_session = Session(boto_session)
    try:
        file_keys = sagemaker_session.list_s3_files(bucket, key_prefix)
        return [
            f"s3://{bucket}/{key}" for key in file_keys
        ]
    except Exception as e: # pylint: disable=broad-except
        raise RuntimeError(f"Error listing files in {s3_file_path}: {e}")
