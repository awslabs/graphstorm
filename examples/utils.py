"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.

    Utility functions to help create require raw data tables and JSON for using
    GraphStorm
"""


def convert_tensor_to_list_arrays(tensor):
    """ Convert Pytorch Tensor to a list of arrays
    
    Since a pandas DataFrame cannot save a 2D numpy array in parquet format, it is necessary to
    convert the tensor (1D or 2D) into a list of lists or a list of array. This converted tensor
    can then be used to build a pandas DataFrame, which can be saved in parquet format. However,
    tensor with a dimension greater than or equal to 3D cannot be processed or saved into parquet
    files.

    Parameters:
    tensor: Pytorch Tensor
        The input Pytorch tensor (1D or 2D) to be converted
    
    Returns:
    list_array: list of numpy arrays
        A list of numpy arrays
    """
    
    np_array = tensor.numpy()
    list_array = [np_array[i] for i in range(len(np_array))]

    return list_array
