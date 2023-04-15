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
"""

class NoopMap:
    """ It doesn't map IDs.

    This is an identity map. It doesn't do any mapping on IDs.

    Parameters
    ----------
    size : int
        The map size.
    """
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def map_id(self, ids):
        """ Map the input IDs to the new IDs.

        This is identity map, so we don't need to do anything.

        Parameters
        ----------
        ids : tensor
            The input IDs

        Returns
        -------
        tuple of tensors : the tensor of new IDs, the location of the IDs in the input ID tensor.
        """
        return ids, np.arange(len(ids))

    def get_key_vals(self):
        """ Get the key value pairs.
        """
        return None

class IdMap:
    """ Map an ID to a new ID.

    This creates an ID map for the input IDs.

    Parameters
    ----------
    ids : Numpy array
        The input IDs
    """
    def __init__(self, ids):
        self._ids = {id1: i for i, id1 in enumerate(ids)}

    def __len__(self):
        return len(self._ids)

    def map_id(self, ids):
        """ Map the input IDs to the new IDs.

        Parameters
        ----------
        ids : tensor
            The input IDs

        Returns
        -------
        tuple of tensors : the tensor of new IDs, the location of the IDs in the input ID tensor.
        """
        for id_ in self._ids:
            # If the data type of the key is string, the input Ids should also be strings.
            if isinstance(id_, str):
                assert isinstance(ids[0], str), \
                        "The key of ID map is string, input IDs should also be strings."
            elif isinstance(id_, int) or np.issubdtype(id_.dtype, np.integer):
                # If the data type of the key is integer, the input Ids should
                # also be integers.
                assert np.issubdtype(ids.dtype, np.integer), \
                        "The key of ID map is integer, input IDs should also be integers. " \
                        + f"But get {type(ids[0])}."
            else:
                raise ValueError(f"Unsupported key data type: {type(id_)}")
            break

        # If the input ID exists in the ID map, map it to a new ID
        # and keep its location in the input ID array.
        # Otherwise, skip the ID.
        new_ids = []
        idx = []
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                new_ids.append(self._ids[id_])
                idx.append(i)
        return np.array(new_ids), np.array(idx)

    def get_key_vals(self):
        """ Get the key value pairs.

        Returns
        -------
        tuple of tensors : The first one has keys and the second has corresponding values.
        """
        return np.array(list(self._ids.keys())), np.array(list(self._ids.values()))
