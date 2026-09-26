import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        connect_timeout_s: Optional[float] = None,
        receive_timeout_s: Optional[float] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._receive_timeout_s = receive_timeout_s
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = None if self._connect_timeout_s is None else time.monotonic() + self._connect_timeout_s
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for policy server at {self._uri}")
            conn = None
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                remaining_s = None if deadline is None else max(0.001, deadline - time.monotonic())
                open_timeout_s = 5.0 if remaining_s is None else min(5.0, remaining_s)
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=open_timeout_s,
                )
                # Recompute the remaining budget after the TCP/WebSocket
                # handshake.  A slow connect must not grant a second, full
                # timeout window to the metadata receive.
                remaining_s = None if deadline is None else max(0.001, deadline - time.monotonic())
                metadata_timeout_s = self._receive_timeout_s
                if remaining_s is not None:
                    metadata_timeout_s = (
                        remaining_s if metadata_timeout_s is None else min(metadata_timeout_s, remaining_s)
                    )
                metadata = msgpack_numpy.unpackb(conn.recv(timeout=metadata_timeout_s))
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError):
                if conn is not None:
                    conn.close()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for policy server at {self._uri}") from None
                logging.info("Still waiting for server...")
                retry_sleep_s = 1.0 if deadline is None else min(1.0, max(0.0, deadline - time.monotonic()))
                time.sleep(retry_sleep_s)
            except BaseException:
                if conn is not None:
                    conn.close()
                raise

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv(timeout=self._receive_timeout_s)
        except (OSError, TimeoutError):
            self.close()
            raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        """Close the underlying websocket; safe to call more than once."""

        self._ws.close()

    @override
    def reset(self) -> None:
        pass
