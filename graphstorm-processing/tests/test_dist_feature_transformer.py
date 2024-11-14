"""Tests for DistFeatureTransformer which wraps individual transformations with a common API"""

from unittest.mock import Mock

from pyspark.sql import DataFrame, SparkSession

from graphstorm_processing.data_transformations.dist_feature_transformer import (
    DistFeatureTransformer,
)
from graphstorm_processing.config.numerical_configs import NumericalFeatureConfig


def test_precomputed_transformer(spark: SparkSession, user_df: DataFrame):
    """Ensure the pre-computed transformation is used when we provide one as input"""

    num_feature_config = NumericalFeatureConfig(
        {
            "column": "age",
            "transformation": {
                "name": "numerical",
                "kwargs": {"imputer": "mean", "normalizer": "min-max"},
            },
        }
    )

    json_rep = {
        "cols": ["age"],
        "imputer_model": {
            "imputed_val_dict": {"age": 27.2},
            "imputer_name": "mean",
        },
        "normalizer_model": {
            "norm_name": "min-max",
            "norm_reconstruction": {
                "originalMinValues": [33],
                "originalMaxValues": [22],
            },
        },
        "out_dtype": "float32",
        "transformation_name": "DistNumericalTransformation",
    }

    numerical_transformer = DistFeatureTransformer(
        num_feature_config,
        spark,
        json_rep,
    )

    # Mock the inner transformation function to check if it's called
    numerical_transformer.transformation.apply_precomputed_transformation = Mock()

    # Call the outer transformation
    _, new_rep = numerical_transformer.apply_transformation(user_df)

    # Assert the precomputed method was called exactly once
    numerical_transformer.transformation.apply_precomputed_transformation.assert_called_once()

    # Assert the newly returned rep matches the previous one
    assert new_rep == json_rep
