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

from enum import Enum

from pyspark.sql.types import FloatType, DoubleType

################### Categorical Limits #######################
MAX_CATEGORIES_PER_FEATURE = 100
RARE_CATEGORY = "GSP_CONSTANT_OTHER"
MISSING_CATEGORY = "GSP_CONSTANT_UNKNOWN"
SINGLE_CATEGORY_COL = "SINGLE_CATEGORY"

SUPPORTED_FILE_TYPES = ["csv", "parquet"]

################### Label Properties  ########################
MIN_VALUE = "MIN_VALUE"
MAX_VALUE = "MAX_VALUE"
COLUMN_NAME = "COLUMN_NAME"
VALUE_COUNTS = "VALUE_COUNTS"

############## Spark-specific constants #####################
SPECIAL_CHARACTERS = {".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "{", "}", "|", "\\"}

"""Configuration to define driver and executor memory for SageMaker PySpark"""
# Percentage of instance memory to allocate to the driver process
DRIVER_MEM_INSTANCE_MEM_RATIO = 0.9
# Fraction of driver memory to be allocated as additional non-heap memory per process
DRIVER_MEM_OVERHEAD_RATIO = 0.1
# Percentage of instance memory to allocate to executor processes
EXECUTOR_MEM_INSTANCE_MEM_RATIO = 0.95
# Fraction of executor memory to be allocated as additional non-heap memory per process
EXECUTOR_MEM_OVERHEAD_RATIO = 0.1

################# Numerical transformations  ################
VALID_IMPUTERS = ["none", "mean", "median", "most_frequent"]
VALID_NORMALIZERS = ["none", "min-max", "standard", "rank-gauss"]
TYPE_FLOAT32 = "float32"
TYPE_FLOAT64 = "float64"
VALID_OUTDTYPE = [TYPE_FLOAT32, TYPE_FLOAT64]
DTYPE_MAP = {TYPE_FLOAT32: FloatType(), TYPE_FLOAT64: DoubleType()}

################# Bert transformations  ################
HUGGINGFACE_TRANFORM = "huggingface"
HUGGINGFACE_TOKENIZE = "tokenize_hf"
HUGGINGFACE_EMB = "embedding_hf"


################# Supported execution envs  ##############
class ExecutionEnv(Enum):
    """Supported execution environments"""

    LOCAL = 1
    SAGEMAKER = 2
    EMR_SERVERLESS = 3
    EMR_ON_EC2 = 4


################# Supported filesystem types#############
class FilesystemType(Enum):
    """Supported filesystem types"""

    LOCAL = 1
    S3 = 2


# NOTE: These need to be updated with each Spark release
# See the value for <hadoop.version> at the respective Spark version
# https://github.com/apache/spark/blob/v3.5.1/pom.xml#L125
# replace both Hadoop versions below with the one there
SPARK_HADOOP_VERSIONS = {
    "3.5": "3.3.4",
    "3.4": "3.3.4",
    "3.3": "3.3.2",
}

########## Precomputed transformations ################
TRANSFORMATIONS_FILENAME = "precomputed_transformations.json"
