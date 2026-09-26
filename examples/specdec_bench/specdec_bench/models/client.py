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

import json
import os
import time

import httpx

from .base import Model


class ClientModel(Model):
    """Benchmark an already-running OpenAI-compatible server instead of an in-process engine.

    Prompts are sent pre-tokenized to ``/v1/completions`` with streaming on, so the
    chat template is applied client-side exactly as it is for the in-process engines.
    Each streamed chunk is one engine step: its ``token_ids`` are the tokens that step
    emitted (accepted draft tokens plus the bonus token), which is the same per-step
    signal the in-process vLLM wrapper reads from ``AsyncLLM.generate``.

    Requires the server to return ``token_ids`` per chunk (vLLM's ``return_token_ids``
    extension). The server owns the speculative-decoding config; ``--draft_model_dir``,
    ``--draft_length`` and ``--speculative_algorithm`` have no effect in this mode.

    ``--runtime_params`` ``engine_args`` accepted here:

    - ``timeout``: per-request timeout in seconds (default: none).
    - ``extra_body``: dict merged into every request body, for server-specific fields.
    """

    def __init__(self, model_dir, max_concurrent_requests, sampling_kwargs, **kwargs):
        base_url = kwargs.get("base_url")
        if not base_url:
            raise ValueError("--engine CLIENT requires --base_url, e.g. http://host:8000/v1")
        self.base_url = base_url.rstrip("/")
        self.served_model_name = model_dir
        self.sampling_kwargs = sampling_kwargs
        self.extra_body = kwargs.get("extra_body") or {}
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            headers=self.headers,
            timeout=httpx.Timeout(kwargs.get("timeout")),
            limits=httpx.Limits(max_connections=max_concurrent_requests),
        )

    def _request_body(self, prompt_ids, max_length, end_id):
        body = {
            "model": self.served_model_name,
            "prompt": prompt_ids,
            "max_tokens": max_length,
            "temperature": self.sampling_kwargs.get("temperature", 1.0),
            "top_p": self.sampling_kwargs.get("top_p", 1.0),
            "stream": True,
            "return_token_ids": True,
        }
        if "top_k" in self.sampling_kwargs:
            body["top_k"] = self.sampling_kwargs["top_k"]
        if end_id == -1:
            body["ignore_eos"] = True
        else:
            body["stop_token_ids"] = [end_id]
        body.update(self.extra_body)
        return body

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        steps, timing = await self.generate(self._request_body(prompt_ids, max_length, end_id))
        # Drop the terminating EOS so output lengths match the in-process engines.
        if steps and steps[-1][-1] == end_id:
            steps[-1] = steps[-1][:-1]
            if not steps[-1]:
                steps.pop()
                timing.pop()
        return {"output_ids": [steps], "output_logits": None, "token_times": timing}

    async def generate(self, body):
        steps = []
        timing = [time.perf_counter()]
        async with self.client.stream(
            "POST", f"{self.base_url}/completions", json=body
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")[:500]
                raise RuntimeError(
                    f"{self.base_url}/completions -> {response.status_code}: {detail}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                for choice in json.loads(payload).get("choices", []):
                    if "token_ids" not in choice:
                        raise RuntimeError(
                            "Server did not return token_ids per streamed chunk; --engine CLIENT "
                            "needs a server that honors return_token_ids (vLLM >= 0.10.2)."
                        )
                    if choice["token_ids"]:
                        steps.append(list(choice["token_ids"]))
                        timing.append(time.perf_counter())
        return steps, timing

    def get_serving_config(self):
        config = {"base_url": self.base_url, "served_model_name": self.served_model_name}
        config["server_version"] = self._get_json(f"{self.base_url.removesuffix('/v1')}/version")
        config["models"] = self._get_json(f"{self.base_url}/models")
        return config

    def _get_json(self, url):
        try:
            return httpx.get(url, headers=self.headers, timeout=10).json()
        except (httpx.HTTPError, ValueError):
            return None
