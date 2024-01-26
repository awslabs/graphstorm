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

from pyspark.sql import DataFrame

from graphstorm_processing.config.feature_config_base import FeatureConfig
from .dist_transformations import (
    DistributedTransformation,
    NoopTransformation,
    DistNumericalTransformation,
    DistMultiNumericalTransformation,
    DistBucketNumericalTransformation,
    DistCategoryTransformation,
    DistMultiCategoryTransformation,
    DistHFTransformation,
)


class DistFeatureTransformer(object):
    """
    Given a feature configuration selects the correct transformation type,
    which can then be be applied through a call to apply_transformation.
    """

    def __init__(self, feature_config: FeatureConfig):
        feat_type = feature_config.feat_type
        feat_name = feature_config.feat_name
        args_dict = feature_config.transformation_kwargs
        self.transformation: DistributedTransformation

        default_kwargs = {"cols": feature_config.cols}
        logging.info("Feature name: %s", feat_name)
        logging.info("Transformation type: %s", feat_type)

        if feat_type == "no-op":
            self.transformation = NoopTransformation(**default_kwargs, **args_dict)
        elif feat_type == "numerical":
            self.transformation = DistNumericalTransformation(**default_kwargs, **args_dict)
        elif feat_type == "multi-numerical":
            self.transformation = DistMultiNumericalTransformation(**default_kwargs, **args_dict)
        elif feat_type == "bucket-numerical":
            self.transformation = DistBucketNumericalTransformation(**default_kwargs, **args_dict)
        elif feat_type == "categorical":
            self.transformation = DistCategoryTransformation(**default_kwargs, **args_dict)
        elif feat_type == "multi-categorical":
            self.transformation = DistMultiCategoryTransformation(**default_kwargs, **args_dict)
        elif feat_type == "huggingface":
            self.transformation = DistHFTransformation(**default_kwargs, **args_dict)
        else:
            raise NotImplementedError(
                f"Feature {feat_name} has type: {feat_type} that is not supported"
            )

    def apply_transformation(self, input_df: DataFrame) -> DataFrame:
        """
        Given an input dataframe, select only the relevant columns
        and apply the expected transformation to them.
        """
        input_df = input_df.select(self.transformation.cols)  # type: ignore

        return self.transformation.apply(input_df)

    def get_transformation_name(self) -> str:
        """
        Get the name of the underlying transformation.
        """
        return self.transformation.get_transformation_name()
