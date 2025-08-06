"""Unit tests for sagemaker_train.py"""
import os

from unittest.mock import patch

from graphstorm.sagemaker.sagemaker_train import copy_best_model_to_sagemaker_output

def test_copy_best_model_to_sagemaker_output_basic(tmp_path):
    """Test copying latest model checkpoint to SageMaker output directory.

    This test verifies that:
    1. The latest epoch directory is correctly identified
    2. Model files are copied from the latest epoch directory
    3. Config files from root directory are copied
    """
    # Create test directory structure
    save_model_path = tmp_path / "model_save"
    save_model_path.mkdir()

    # Create epoch directories with dummy model files
    epoch1 = save_model_path / "epoch-1"
    epoch2 = save_model_path / "epoch-2"
    epoch1.mkdir()
    epoch2.mkdir()

    # Create dummy files including nested directory structure
    (epoch1 / "model.bin").write_text("model1 content")

    # Create nested structure in epoch2
    (epoch2 / "model.bin").write_text("model2 content")
    nested_dir = epoch2 / "nested"
    nested_dir.mkdir()
    (nested_dir / "extra.pt").write_text("extra content")

    (save_model_path / "config.yaml").write_text("config content")

    # Mock SM_MODEL_OUTPUT and ensure directory exists
    mock_output = str(tmp_path / "sm_output")
    os.makedirs(mock_output, exist_ok=True)

    with patch("graphstorm.sagemaker.sagemaker_train.SM_MODEL_OUTPUT", mock_output):
        copy_best_model_to_sagemaker_output(str(save_model_path))

        # Verify latest epoch files were copied
        assert os.path.exists(os.path.join(mock_output, "model.bin"))
        with open(os.path.join(mock_output, "model.bin"), 'r') as f:
            assert f.read() == "model2 content"

        # Verify nested directory was copied
        nested_path = os.path.join(mock_output, "nested", "extra.pt")
        assert os.path.exists(nested_path)
        with open(nested_path, 'r') as f:
            assert f.read() == "extra content"

        # Verify config file was copied
        assert os.path.exists(os.path.join(mock_output, "config.yaml"))
        with open(os.path.join(mock_output, "config.yaml"), 'r') as f:
            assert f.read() == "config content"

def test_copy_best_model_to_sagemaker_output_with_iterations(tmp_path):
    """Test copying latest model checkpoint when using iteration-specific epochs.

    This test verifies that:
    1. The latest epoch-iteration directory is correctly identified
    2. Files from the latest epoch-iteration are copied correctly
    """
    # Create test directory structure
    save_model_path = tmp_path / "model_save"
    save_model_path.mkdir()

    # Create epoch directories with iterations
    epoch5_iter1 = save_model_path / "epoch-5-iter-100"
    epoch5_iter2 = save_model_path / "epoch-5-iter-200"
    epoch5_iter1.mkdir()
    epoch5_iter2.mkdir()

    # Create dummy files
    (epoch5_iter1 / "model.bin").write_text("model5-100 content")
    (epoch5_iter2 / "model.bin").write_text("model5-200 content")
    (save_model_path / "config.yaml").write_text("config content")

    # Mock SM_MODEL_OUTPUT and ensure directory exists
    mock_output = str(tmp_path / "sm_output")
    os.makedirs(mock_output, exist_ok=True)

    with patch("graphstorm.sagemaker.sagemaker_train.SM_MODEL_OUTPUT", mock_output):
        copy_best_model_to_sagemaker_output(str(save_model_path))

        # Verify latest iteration files were copied
        assert os.path.exists(os.path.join(mock_output, "model.bin"))
        with open(os.path.join(mock_output, "model.bin"), 'r') as f:
            assert f.read() == "model5-200 content"

        # Verify config file was copied
        assert os.path.exists(os.path.join(mock_output, "config.yaml"))
        with open(os.path.join(mock_output, "config.yaml"), 'r') as f:
            assert f.read() == "config content"

def test_copy_best_model_to_sagemaker_output_specific_epoch(tmp_path):
    """Test copying a specific epoch when best_epoch is provided.

    This test verifies that:
    1. The specified best_epoch is used instead of the latest
    2. Files from the specified epoch are copied correctly
    """
    # Create test directory structure
    save_model_path = tmp_path / "model_save"
    save_model_path.mkdir()

    # Create epoch directories with dummy model files
    epoch1 = save_model_path / "epoch-0"
    epoch2 = save_model_path / "epoch-1"
    epoch1.mkdir()
    epoch2.mkdir()

    # Create dummy files
    (epoch1 / "model.bin").write_text("model0 content")
    (epoch2 / "model.bin").write_text("model1 content")
    (save_model_path / "config.yaml").write_text("config content")

    # Mock SM_MODEL_OUTPUT and ensure directory exists
    mock_output = str(tmp_path / "sm_output")
    os.makedirs(mock_output, exist_ok=True)

    with patch("graphstorm.sagemaker.sagemaker_train.SM_MODEL_OUTPUT", mock_output):
        # Specify epoch-0 as best epoch even though epoch-1 is latest
        copy_best_model_to_sagemaker_output(str(save_model_path), best_epoch="epoch-0")

        # Verify specified epoch files were copied
        assert os.path.exists(os.path.join(mock_output, "model.bin"))
        with open(os.path.join(mock_output, "model.bin"), 'r') as f:
            assert f.read() == "model0 content"

        # Verify config file was copied
        assert os.path.exists(os.path.join(mock_output, "config.yaml"))
        with open(os.path.join(mock_output, "config.yaml"), 'r') as f:
            assert f.read() == "config content"
