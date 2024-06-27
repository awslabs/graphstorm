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

from pyspark.sql import DataFrame, SparkSession

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

    def __init__(
        self, feature_config: FeatureConfig, spark: SparkSession, json_representation: dict
    ):
        feat_type = feature_config.feat_type
        feat_name = feature_config.feat_name
        args_dict = feature_config.transformation_kwargs
        self.transformation: DistributedTransformation
        # We use this to re-apply transformations
        self.json_representation = json_representation

        default_kwargs = {
            "cols": feature_config.cols,
        }
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
            self.transformation = DistCategoryTransformation(
                **default_kwargs, **args_dict, spark=spark, json_representation=json_representation
            )
        elif feat_type == "multi-categorical":
            self.transformation = DistMultiCategoryTransformation(**default_kwargs, **args_dict)
        elif feat_type == "huggingface":
            self.transformation = DistHFTransformation(**default_kwargs, **args_dict)
        else:
            raise NotImplementedError(
                f"Feature {feat_name} has type: {feat_type} that is not supported"
            )

    def apply_transformation(self, input_df: DataFrame) -> tuple[DataFrame, dict]:
        """
        Given an input DataFrame, select only the relevant columns
        and apply the expected transformation to them.

        Returns
        -------
        tuple[DataFrame, dict]
            A tuple with two items, the first is the transformed input DataFrame,
            the second is a JSON representation of the transformation. This will
            allow us to apply the same transformation to new data.
        """
        input_df = input_df.select(self.transformation.cols)  # type: ignore

        if self.json_representation:
            logging.info("Applying precomputed transformation...")
            return (
                self.transformation.apply_precomputed_transformation(input_df),
                self.json_representation,
            )
        else:
            return (
                self.transformation.apply(input_df),
                self.transformation.get_json_representation(),
            )

    def get_transformation_name(self) -> str:
        """
        Get the name of the underlying transformation.
        """
        return self.transformation.get_transformation_name()
