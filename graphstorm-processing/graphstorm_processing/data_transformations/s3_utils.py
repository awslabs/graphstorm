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
from typing import List, Optional

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
    ), f"Prefix should not start with 's3://' but be relative to the bucket, {bucket}, got {prefix}"
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


def extract_bucket_and_key(
    path_with_bucket: str, relative_path: Optional[str] = None
) -> tuple[str, str]:
    """Given an S3 path that includes a bucket, and a relative path,
    extracts the bucket name and full key path. If only `path_with_bucket`
    is provided, will split that path into bucket name and prefix path.

    Parameters
    ----------
    path_with_bucket : str
        An S3 path that can include a bucket name and a key prefix, e.g.
        's3://my-bucket/my/prefix/'.
    relative_path : Optional[str]
        An S3 key path that's relative to `path_with_bucket`, e.g.
        'rest/of/path/to/key'. If not provided only `path_with_bucket`
        will be split.

    Returns
    -------
    str
        A tuple whose first element is the bucket name and the second
        the full path to the key.

    Example
    -------
    .. code::

        >>> extract_bucket_and_key("s3://my-bucket/prefix", "rest/of/path/to/key")
        ("my_bucket", "prefix/rest/of/path/to/key")
        >>> extract_bucket_and_key("s3://my-bucket/prefix/key")
        ("my_bucket", "prefix/key")
        >>> extract_bucket_and_key("s3://my-bucket/")
        ("my_bucket", "")
    """
    if not path_with_bucket.startswith("s3://"):
        path_with_bucket = f"s3://{path_with_bucket}"
    path_with_bucket = s3_path_remove_trailing(path_with_bucket)
    if relative_path:
        if relative_path.startswith("/"):
            relative_path = relative_path[1:]
        file_s3_uri = f"{path_with_bucket}/{relative_path}"
    else:
        file_s3_uri = path_with_bucket
    # We split on '/' to get the bucket, as it's always the third split element in an S3 URI
    file_bucket = file_s3_uri.split("/")[2]
    # Similarly, by having maxsplit=3 we get the S3 key value as the fourth element
    file_parts = file_s3_uri.split("/", maxsplit=3)
    if len(file_parts) == 4:
        file_key = file_parts[3]
    else:
        file_key = ""
    # We remove any trailing '/' from the key
    file_key = s3_path_remove_trailing(file_key)

    return file_bucket, file_key
