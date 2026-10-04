# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from collections import defaultdict

import httpx

from .base import Metric

_SAMPLE = re.compile(r"^(\w+:\w+)(?:\{([^}]*)\})?\s+(\S+)")
_POSITION = re.compile(r'position="(\d+)"')
_VLLM = {
    "running": "vllm:num_requests_running",
    "generation": "vllm:generation_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "accepted_per_pos": "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}
# tensorfold counts finished requests only, a draft per verify round, and no positions.
_TENSORFOLD = {
    "running": "tensorfold:requests_running",
    "generation": "tensorfold:generation_tokens_total",
    "drafts": "tensorfold_health:rounds_total",
    "draft_tokens": "tensorfold:mtp_drafted_total",
    "accepted": "tensorfold:mtp_accepted_total",
    "accepted_per_pos": None,
}


def parse_metrics(text):
    """Sum Prometheus samples across engines and other labels, keyed by (name, position)."""
    counters = defaultdict(float)
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name, labels, value = match.groups()
        position = _POSITION.search(labels or "")
        counters[(name, int(position.group(1)) if position else None)] += float(value)
    return counters


class ServerSpecDecode(Metric):
    """Server-side acceptance for ``--engine CLIENT``, from the server's ``/metrics`` counters.

    Snapshots vLLM's ``vllm:spec_decode_*`` counters, or tensorfold's ``tensorfold:mtp_*``
    and ``tensorfold_health:rounds_total``, before and after the run and reports the
    difference. It cross-checks the client-side acceptance length, which is inferred from
    streamed chunk sizes. The counters are server-wide, so other traffic sent to the server
    during the run is counted too: compare ``Server_Generation_Tokens`` with the client's
    output token count, and heed the warning printed when requests were already running.
    """

    def __init__(self, base_url, api_key=None):
        super().__init__()
        self.name = "server_spec_decode"
        self.url = f"{base_url.rstrip('/').removesuffix('/v1')}/metrics"
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.start = self._snapshot()
        tensorfold = self.start is not None and (_TENSORFOLD["drafts"], None) in self.start
        self.names = _TENSORFOLD if tensorfold else _VLLM
        self.running = (self.names["running"], None)
        if self.start is not None and self.start.get(self.running, 0) > 0:
            print(
                f"Warning: {self.url} already has {self.start[self.running]:.0f} request(s) running; "
                "they share the batch with this benchmark and are counted in the server metrics"
            )

    def _snapshot(self):
        try:
            response = httpx.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Server spec-decode metrics unavailable at {self.url}: {e}")
            return None
        return parse_metrics(response.text)

    def process_step(self, step_outputs, request_id, turn_id):
        pass

    def process_final(self, text_outputs):
        end = self._snapshot()
        if self.start is None or end is None:
            return
        delta = {key: end[key] - self.start.get(key, 0.0) for key in end}
        names = self.names
        drafts = delta.get((names["drafts"], None), 0.0)
        if drafts <= 0:
            print("Server reported no speculative drafts during the run")
            return
        draft_tokens = delta.get((names["draft_tokens"], None), 0.0)
        accepted = delta.get((names["accepted"], None), 0.0)
        per_position = {
            position: value / drafts
            for (name, position), value in delta.items()
            if name == names["accepted_per_pos"]
        }
        self.out["Requests_Running_At_Start"] = self.start.get(self.running, 0.0)
        self.out["Server_Generation_Tokens"] = delta.get((names["generation"], None), 0.0)
        self.out["Num_Drafts"] = drafts
        self.out["Num_Draft_Tokens"] = draft_tokens
        self.out["Num_Accepted_Tokens"] = accepted
        self.out["Average_AL"] = 1 + accepted / drafts
        self.out["Draft_Acceptance_Rate"] = accepted / draft_tokens if draft_tokens else None
        self.out["Per_Position_Acceptance_Rate"] = dict(sorted(per_position.items()))
        print("Server-side Average AL:", self.out["Average_AL"])
        print("Server-side per-position acceptance rate:", self.out["Per_Position_Acceptance_Rate"])
        self.write()

    def clear(self):
        pass
