"""
S3 utilities for SageMaker jobs.
"""
import logging
import os

import boto3
import sagemaker
from sagemaker.s3 import S3Downloader, S3Uploader

def download_data_from_s3(input_s3, local_data_path, sagemaker_session=None):
    """ Download  data info from S3

    Parameters
    ----------
    input_s3: str
        S3 uri of the input file
    local_data_path: str
        Local file path to store the downloaded file.
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    logging.debug("download %s into %s", input_s3, local_data_path)
    if not sagemaker_session:
        boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
        sagemaker_session = sagemaker.Session(boto_session)
    try:
        S3Downloader.download(input_s3,
            local_data_path, sagemaker_session=sagemaker_session)
    except Exception as e: # pylint: disable=broad-except
        raise RuntimeError(f"Error while downloading {input_s3}: {e}")

def upload_file_to_s3(s3_data_path, local_data_path, sagemaker_session=None):
    """ Upload data info to S3

    Parameters
    ----------
    data_path_s3_path: str
        S3 uri to upload the data
    local_data_path: str
        Path to local data
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run upload
    """
    s3_data_path = s3_data_path[:-1] if s3_data_path.endswith('/') else s3_data_path
    logging.debug("upload %s into %s", local_data_path, s3_data_path)
    if not sagemaker_session:
        boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
        sagemaker_session = sagemaker.Session(boto_session)
    try:
        ret = S3Uploader.upload(local_data_path, s3_data_path,
            sagemaker_session=sagemaker_session)
    except Exception as e: # pylint: disable=broad-except
        raise RuntimeError(f"Error uploading data to {s3_data_path}: {e}")
    return ret
