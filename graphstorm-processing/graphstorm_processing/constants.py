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

"""Configuration to define driver and executor memory for distributed"""
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
