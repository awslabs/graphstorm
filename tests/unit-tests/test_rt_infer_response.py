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

from graphstorm.sagemaker import GSRealTimeInferenceResponseMessage as RERespMsg


def test_realtime_infer_res_msg():
    """ A simple unit test of the GSRealTimeInferenceResponseMessage
    """
    dummy_req_uid = '12345678abcdefgh'

    # success case
    data = {"results": [42]}
    resp = RERespMsg.success(request_uid=dummy_req_uid, data=data)
    assert resp.status_code == 200
    assert resp.message is not None
    assert resp.error is None
    assert resp.data == data
    assert isinstance(resp.to_dict(), dict)

    # to_dict
    # Ensure that all output fields in to_dict match the instance attributes if present
    resp = RERespMsg.success(request_uid=dummy_req_uid, data={"foo": "bar"})
    res = resp.to_dict()
    assert res["status_code"] == resp.status_code
    assert res["message"] == resp.message
    assert res["data"] == resp.data
    assert res["error"] == ''

    # missing required field case
    resp = RERespMsg.missing_required_field(request_uid=dummy_req_uid, field="node_features")
    assert resp.status_code == 401
    assert resp.message is None
    assert "node_features" in resp.error
    assert resp.data == {}

    # internal server error
    detail = "Unexpected exception"
    resp = RERespMsg.internal_server_error(request_uid=dummy_req_uid, detail=detail)
    assert resp.status_code == 500
    assert resp.error is not None
    assert detail in resp.error
