import os
import tempfile
import numpy as np
import graphstorm as gs

def test_parquet():
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)

    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    gs.graph_construct.construct_graph.write_data_parquet(data, tmpfile)
    data1 = gs.graph_construct.construct_graph.read_data_parquet(tmpfile)
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    assert np.all(data1['data1'] == data['data1'])
    assert np.all(data1['data2'] == data['data2'])

if __name__ == '__main__':
    test_parquet()
