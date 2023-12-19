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

    The primitives required for distributed computations.
"""

from dgl.distributed import rpc

from ..utils import barrier

FLUSH_DATA = 1000001

class FlushRequest(rpc.Request):
    """This request flushes data in DGL's distributed computation components.

    When DGL performs writing to distributed tensors, it returns without data
    being fully written to the distributed tensors. This operation is to ensure
    that all data has been written to the distributed tensors on the server
    when the operation returns. In practice, we don't need to perform any operations
    in the request, except just sending responses to the client. The reason is
    that when servers receive requests from clients, they processes them in
    the FIFO order. When a server gets the opportunities to process the Flush request,
    it means that the server has processed all requests before it and has written
    data to the distributed tensors.
    """

    def __init__(self):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass

    def process_request(self, server_state):
        """ Process the request.

        Here we don't need to do anything except returning the flush response.
        """
        return FlushResponse()

class FlushResponse(rpc.Response):
    """Ack the flush request"""

    def __init__(self):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass

rpc.register_service(FLUSH_DATA, FlushRequest, FlushResponse)

def flush_data():
    """ Flush data in distributed writes of DGL.

    All processes need to talk to all server processes and make sure
    all server processes complete processing the write requests issued by
    the trainer processes. The reason that we need to have all processes
    to communicate with all servers is that there are N*M communication channels,
    where N is the number of trainer processes and M is the number of servers.
    We need to make sure we flush data in all communication channels.
    This function is called after trainer processes have finished issuing write
    requests to servers and have written data to shared memory.
    We can guarantee that all data are written to distributed tensors when
    this function returns.
    """
    request = FlushRequest()
    # send request to all the server nodes
    server_count = rpc.get_num_server()
    for server_id in range(server_count):
        rpc.send_request(server_id, request)
    # recv response from all the server nodes
    for _ in range(server_count):
        _ = rpc.recv_response()
    barrier()
