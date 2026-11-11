# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

from dataclasses import dataclass
import io
from typing import Any, Callable, Protocol

import msgpack
import numpy as np
import torch
import zmq

from rldx.data.types import ModalityConfig
from rldx.data.utils import to_json_serializable


DEFAULT_MODEL_SERVER_PORT = 5555


class Policy(Protocol):
    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def get_modality_config(self) -> dict[str, ModalityConfig]: ...

    def get_rtc(self) -> dict[str, int] | None: ...

    def get_memory(self) -> dict[str, int] | None: ...


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, torch.Tensor):
            # bf16 / fp16 have no native numpy dtype; up-cast to float32 so
            # decoding stays lossless for the values the wire carries (RTC
            # action_prefix, in particular).
            t = obj.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            obj = t.numpy()
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, policy: Policy, host: str = "*", port: int = 5555, api_token: str = None):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            self.policy.get_modality_config,
            requires_input=False,
        )
        self.register_endpoint("get_rtc", self.policy.get_rtc, requires_input=False)
        self.register_endpoint("get_memory", self.policy.get_memory, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))


class PolicyClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def get_rtc(self) -> dict[str, int] | None:
        return self.call_endpoint("get_rtc", requires_input=False)

    def get_memory(self) -> dict[str, int] | None:
        return self.call_endpoint("get_memory", requires_input=False)


def serve(policy: Any, host: str, port: int) -> None:
    server = PolicyServer(policy=policy, host=host, port=port)
    print(f"\n✓ Server ready — listening on {host}:{port}\n")
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


def require_cadence(
    rtc: dict[str, int] | None, memory: dict[str, int] | None, n_action_steps: int
) -> None:
    """A chunk's executed steps are what RTC's prefix or memory's snapshots assume."""
    if rtc is not None:
        expected = rtc["exec_horizon"]
    elif memory is not None:
        expected = memory["stride"]
    else:
        return
    if expected != n_action_steps:
        raise ValueError(
            f"the server expects {expected} executed steps per chunk, "
            f"the env executes n_action_steps={n_action_steps}"
        )


def memory_options(
    memory: dict[str, int] | None, session_ids: list[str], first_step: list[bool]
) -> dict[str, Any]:
    if memory is None:
        return {}
    return {"session_ids": list(session_ids), "reset_memory": list(first_step)}


def rtc_options(
    rtc: dict[str, int] | None, first_step: list[bool], previous: np.ndarray | None
) -> dict[str, Any]:
    """The rest of each env's previous chunk; an env that just reset has none."""
    if rtc is None or previous is None:
        return {}
    return {
        "action_prefix": previous,
        "rtc_prefix_len": [0 if first else rtc["delay"] for first in first_step],
    }


def remaining_chunk(actions: dict[str, np.ndarray], rtc: dict[str, int]) -> np.ndarray:
    return np.concatenate(list(actions.values()), axis=-1)[:, rtc["exec_horizon"] :]
