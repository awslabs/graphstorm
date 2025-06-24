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

    GraphStorm SageMaker endpoint http response for realtime inference
"""

class GSRealTimeInferenceResponseMessage:
    """
    ResponseMessage provides standardized HTTP response structures for real-time inference endpoint.

    The class supports:
    - Successful responses (2XX) with result data.
    - Client-side error responses (4XX) for input validation, graph construction, and task mapping
      errors.
    - Server-side error responses (5XX) for internal failures.

    Parameters
    ----------
    status_code: int
        The HTTP status code for the response.
    message:str
        The success message to include. Default is None.
    error: str
        The error message to include. Default is None.
    data: dict
        The prediction results payload for the response. Default is None
    """
    def __init__(
            self,
            status_code,
            message = None,
            error = None,
            data = None
        ):
            self.status_code = status_code
            self.message = message
            self.error = error
            if data is not None:
                self.data = data
            else:
                self.data = {}

    def to_dict(self):
        """Return a dict representation for JSON serialization."""
        res = {"status_code": self.status_code}
        if self.message is not None:
            res["message"] = self.message
        else:
            res["message"] = ''
        if self.error is not None:
            res["error"] = self.error
        else:
            res["error"] = ''
        if self.data is not None:
            res["results"] = self.data
        else:
            res["results"] = {}
        return res

    def success(cls, data):
        """
        Create a successful response with data.
        """
        return cls(
            status_code = 200,
            message = "Request processed successfully.",
            data = data
        )

    def missing_required_field(cls, field):
        """
        Create a response for missing required field errors (401).
        """
        return cls(
            status_code=401,
            error=(
                f"Missing Required Field: The input payload missed the '{field}' field. " \
                "Please refer to the GraphStorm realtime inference documentation for required " \
                "fields."
            )
        )

    def missing_data_in_field(cls, field):
        """
        Create a response for missing data in a field (402).
        """
        return cls(
            status_code=402,
            error=(
                f"Missing Data in Field: The input payload contains the '{field}' field but no " \
                "associated data."
            )
        )

    def mismatch_target_nid(cls, target_nid):
        """
        Create a response for target indexes are missing in the subgraph (403)
        """
        return cls(
            status_code=403,
            error=(
                f"Mismatch target node IDs: the target node ID: {target_nid} does not existing in " \
                 "the payload graph."
            )
        )

    def graph_construction_failure(cls, track):
        """
        Create a response for graph construction failures (411).
        """
        return cls(
            status_code = 411,
            error = (
                f"Graph Construction Failure: Failed to construct a DGL graph from the http "
                 "payload. Details: {track}"
            )
        )

    def model_mismatch_error(cls, track):
        """
        Create a response for model/data mismatch errors (421).
        """
        return cls(
            status_code = 421,
            error = (
                "Task Mismatch: Input payload\'s task mismatched with the endpoint model. " \
                f"Details: {track}"
            )
        )

    def internal_server_error(cls, detail):
        """
        Create a generic internal server error response (500).
        """
        msg = ("Internal Server Error: Please Please contact with your endpoint administrators " \
               "for this error. " )
        if detail:
            msg += f"Details: {detail}"
        return cls(
            status_code=500,
            error=msg
        )