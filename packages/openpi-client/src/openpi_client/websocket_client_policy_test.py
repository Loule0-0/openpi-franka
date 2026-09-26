from __future__ import annotations

from collections import deque

import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeConnection:
    def __init__(self, responses: list[bytes | BaseException]) -> None:
        self.responses = deque(responses)
        self.closed = False
        self.sent: list[bytes] = []
        self.recv_timeouts: list[float | None] = []

    def recv(self, timeout: float | None = None) -> bytes:
        self.recv_timeouts.append(timeout)
        result = self.responses.popleft()
        if isinstance(result, BaseException):
            raise result
        return result

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


def test_client_uses_bounded_metadata_and_inference_receives(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(
        [
            msgpack_numpy.packb({"schema": "test"}),
            msgpack_numpy.packb({"actions": [1, 2]}),
        ]
    )
    connect_args = {}

    def connect(uri: str, **kwargs):
        connect_args.update(uri=uri, **kwargs)
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)
    policy = websocket_client_policy.WebsocketClientPolicy(
        "127.0.0.1",
        8000,
        connect_timeout_s=2.0,
        receive_timeout_s=0.5,
    )

    assert policy.get_server_metadata() == {"schema": "test"}
    assert policy.infer({"state": [0]}) == {"actions": [1, 2]}
    assert connect_args["uri"] == "ws://127.0.0.1:8000"
    assert connect_args["open_timeout"] <= 2.0
    assert connection.recv_timeouts == [pytest.approx(0.5), pytest.approx(0.5)]


def test_inference_timeout_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection([msgpack_numpy.packb({}), TimeoutError("late response")])
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *_args, **_kwargs: connection,
    )
    policy = websocket_client_policy.WebsocketClientPolicy(receive_timeout_s=0.1)

    with pytest.raises(TimeoutError, match="late response"):
        policy.infer({})
    assert connection.closed
