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

import builtins
import os

import torch


def _env_rank(default=0):
    return int(os.environ.get("RANK", str(default)))


def is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_global_rank() -> int:
    if is_dist_initialized():
        return torch.distributed.get_rank()
    return _env_rank(0)


def is_global_zero() -> bool:
    return get_global_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def rank_zero_print(*args, force: bool = False, **kwargs):
    """Print only on global rank 0 (or if force=True)."""
    if force or is_global_zero():
        return builtins.print(*args, **kwargs)
