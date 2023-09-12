"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
from typing import List

import boto3
from botocore.config import Config


def get_high_retry_s3_client() -> boto3.client:
    """
    Returns a high-retry S3 client.
    """
    config = Config(retries={"max_attempts": 10, "mode": "adaptive"})

    return boto3.client("s3", config=config)


def list_s3_objects(bucket: str, prefix: str, s3_boto_client: boto3.client = None) -> List[str]:
    """
    Lists all objects under provided S3 bucket and prefix. Returns ordered list of object key paths.

    Note that the function returns key paths, not the full S3 uri.
    E.g. if called with list_s3_objects('my-bucket', 'my-prefix/'), this function will return:
    [
        'my-prefix/file1.txt',
        'my-prefix/file2.txt',
        'my-prefix/subdir/file3.txt',
    ]
    """
    assert not prefix.startswith(
        "s3://"
    ), f"Prefix should not start with 's3://' but be relative to bucket, got {prefix}"
    s3_boto_client = get_high_retry_s3_client() if s3_boto_client is None else s3_boto_client
    paginator = s3_boto_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    object_list = []
    for page in pages:
        if "Contents" in page.keys():
            for obj in page["Contents"]:
                object_list.append(obj["Key"])
        else:
            logging.warning("No objects found under prefix: %s/%s", bucket, prefix)
            break

    return sorted(object_list)


def get_bucket_region(bucket: str, s3_boto_client: boto3.client = None) -> str:
    """
    Returns the region of the provided S3 bucket.
    """
    s3_boto_client = get_high_retry_s3_client() if s3_boto_client is None else s3_boto_client
    response = s3_boto_client.head_bucket(Bucket=bucket)
    return response["ResponseMetadata"]["HTTPHeaders"]["x-amz-bucket-region"]


def s3_path_remove_trailing(s3_path: str) -> str:
    """
    Removes trailing slash from S3 path.
    """
    if s3_path.endswith("/"):
        return s3_path[:-1]
    return s3_path
