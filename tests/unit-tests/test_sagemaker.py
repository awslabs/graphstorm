"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import pytest
from graphstorm.sagemaker import GSRealTimeInferenceResponseMessage as res_msg


def test_realtime_infer_res_msg():
    """ A simple unit test of the GSRealTimeInferenceResponseMessage
    """
    # success case
    data = {"result": [42]}
    resp = res_msg.success(data)
    assert resp.status_code == 200
    assert resp.message is not None
    assert resp.error is None
    assert resp.data == data
    assert isinstance(resp.to_dict(), dict)

    # to_dict
    # Ensure that all output fields in to_dict match the instance attributes if present
    resp = res_msg.success({"foo": "bar"})
    res = resp.to_dict()
    assert res["status_code"] == resp.status_code
    assert res["message"] == resp.message
    assert res["data"] == resp.data
    # 'error' may not be present in a success response
    assert "error" not in res or res["error"] is None

    # missing required field case
    resp = res_msg.missing_required_field("node_features")
    assert resp.status_code == 401
    assert resp.message is None
    assert "node_features" in resp.error
    assert resp.data is None

    # internal server error
    detail = "Unexpected exception"
    resp = res_msg.internal_server_error(detail)
    assert resp.status_code == 500
    assert resp.error is not None
    assert detail in resp.error


if __name__ == '__main__':
    test_realtime_infer_res_msg()