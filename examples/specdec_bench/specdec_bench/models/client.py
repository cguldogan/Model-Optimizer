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
from itertools import accumulate

import httpx

from .base import Model


def split_rounds(ids, chunks, decode, end_time):
    """Split a reply's token ``ids`` into one step per streamed text chunk.

    For servers that stream text instead of token ids (tensorfold). Such a server streams each
    decoding round as one text chunk once the round's tokens decode to whole characters, the way
    tensorfold's ``StreamDecoder`` does. This replays that incremental decoding token by token:
    the token that makes the decoded text as long as the text streamed so far ends that chunk's
    step. A round that adds no visible text streams no chunk and is merged into the next step.
    Tokens left after the last chunk, such as the end token, join the last step.

    ``chunks`` holds ``(text, arrival_time)`` pairs. Returns the steps and each step's time.
    """
    steps, times, step = [], [], []
    ends = list(accumulate(len(text) for text, _ in chunks))
    chunk, length, prefix, read = 0, 0, 0, 0
    for i, token in enumerate(ids):
        step.append(token)
        before = decode(ids[prefix:read])
        after = decode(ids[prefix : i + 1])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            length += len(after) - len(before)
            prefix, read = read, i + 1
        if chunk < len(chunks) and length >= ends[chunk]:
            while chunk + 1 < len(chunks) and length >= ends[chunk + 1]:
                chunk += 1
            steps.append(step)
            times.append(chunks[chunk][1])
            step = []
            chunk += 1
    if step:
        if steps:
            steps[-1].extend(step)
        else:
            steps.append(step)
            times.append(end_time)
    return steps, times


class ClientModel(Model):
    """Benchmark an already-running OpenAI-compatible server instead of an in-process engine.

    Prompts are sent pre-tokenized to ``/v1/completions`` with streaming on, so the
    chat template is applied client-side exactly as it is for the in-process engines.
    Each streamed chunk is one engine step: its ``token_ids`` are the tokens that step
    emitted (accepted draft tokens plus the bonus token), which is the same per-step
    signal the in-process vLLM wrapper reads from ``AsyncLLM.generate``.

    Requires the server to return ``token_ids`` per chunk (vLLM's ``return_token_ids``
    extension), or, as tensorfold does, to stream one text chunk per round and return the
    reply's ids in the last chunk's ``tensorfold.token_ids``; the steps are then recovered
    with ``split_rounds``, which needs ``--tokenizer`` to be the server's tokenizer. The
    server owns the speculative-decoding config; ``--draft_model_dir``, ``--draft_length``
    and ``--speculative_algorithm`` have no effect in this mode.

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
        self.tokenizer_path = kwargs.get("tokenizer_path") or model_dir
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self._tokenizer = None
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
        # Chunks without token_ids (tensorfold): their text and arrival, and the reply's ids.
        texts, reply_ids, missing = [], None, False
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
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError(f"{self.base_url}/completions stream error: {data['error']}")
                reply_ids = (data.get("tensorfold") or {}).get("token_ids", reply_ids)
                for choice in data.get("choices", []):
                    if "token_ids" not in choice:
                        missing = True
                        if choice.get("text"):
                            texts.append((choice["text"], time.perf_counter()))
                    elif choice["token_ids"]:
                        steps.append(list(choice["token_ids"]))
                        timing.append(time.perf_counter())
        if missing:
            if reply_ids is None:
                raise RuntimeError(
                    "Server did not return token_ids per streamed chunk; --engine CLIENT "
                    "needs a server that honors return_token_ids (vLLM >= 0.10.2, or tensorfold)."
                )
            steps, times = split_rounds(reply_ids, texts, self._decode, time.perf_counter())
            timing += times
        return steps, timing

    def _decode(self, ids):
        # The server's own decoding: the fast tokenizer's backend, special tokens kept.
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=self.trust_remote_code
            )
            self._tokenizer = getattr(tokenizer, "backend_tokenizer", tokenizer)
        return self._tokenizer.decode(ids, skip_special_tokens=False)

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
