# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for --engine CLIENT: the streaming client model and the server-side acceptance metric."""

import asyncio
import json

import httpx
import pytest
from specdec_bench.metrics.server_spec_decode import ServerSpecDecode, parse_metrics
from specdec_bench.models.client import ClientModel, split_rounds

EOS = 2


def _sse(chunks):
    lines = [f"data: {json.dumps({'choices': [c] if c is not None else []})}\n\n" for c in chunks]
    return "".join(lines) + "data: [DONE]\n\n"


def _model(handler, **kwargs):
    model = ClientModel(
        "served-model",
        max_concurrent_requests=1,
        sampling_kwargs={"temperature": 0},
        base_url="http://server:8000/v1/",
        **kwargs,
    )
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return model


def _run(model, end_id=EOS):
    return asyncio.run(model.run([1, 2, 3], 16, end_id, request_id=0, turn_id=0))


def _stream(chunks, captured=None):
    def handler(request):
        if captured is not None:
            captured.append(request)
        return httpx.Response(200, text=_sse(chunks))

    return handler


def test_each_chunk_is_one_step_and_eos_is_dropped():
    chunks = [{"token_ids": [10]}, {"token_ids": []}, {"token_ids": [11, 12, 13]}]
    chunks += [{"token_ids": [14, EOS]}, None]
    out = _run(_model(_stream(chunks)))
    assert out["output_ids"] == [[[10], [11, 12, 13], [14]]]
    # One start time plus one per non-empty step.
    assert len(out["token_times"]) == 4


def test_step_holding_only_eos_is_removed_with_its_time():
    out = _run(_model(_stream([{"token_ids": [10, 11]}, {"token_ids": [EOS]}])))
    assert out["output_ids"] == [[[10, 11]]]
    assert len(out["token_times"]) == 2


def test_request_body_is_pretokenized_streaming_completion():
    captured = []
    model = _model(_stream([{"token_ids": [10]}], captured), extra_body={"top_k": 5})
    _run(model)
    request = captured[0]
    assert str(request.url) == "http://server:8000/v1/completions"
    body = json.loads(request.content)
    assert body["model"] == "served-model"
    assert body["prompt"] == [1, 2, 3]
    assert body["stream"] is True
    assert body["return_token_ids"] is True
    assert body["stop_token_ids"] == [EOS]
    assert body["top_k"] == 5
    assert "ignore_eos" not in body


def test_ignore_eos_when_end_id_is_negative():
    captured = []
    _run(_model(_stream([{"token_ids": [10]}], captured)), end_id=-1)
    body = json.loads(captured[0].content)
    assert body["ignore_eos"] is True
    assert "stop_token_ids" not in body


def test_missing_token_ids_fails_loudly():
    with pytest.raises(RuntimeError, match="return_token_ids"):
        _run(_model(_stream([{"text": "hi"}])))


# A byte-level tokenizer: 13 and 14 are the two UTF-8 bytes of "é", EOS decodes as text.
_PIECES = {10: b"He", 11: b"llo", 12: b" w", 13: b"\xc3", 14: b"\xa9", 15: b"!", EOS: b"<eos>"}


class _ByteTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return b"".join(_PIECES[i] for i in ids).decode("utf-8", errors="replace")


def _texts(*texts):
    return [(text, float(i)) for i, text in enumerate(texts, 1)]


def test_split_rounds_recovers_rounds_across_a_split_character():
    # Rounds [10, 11], [12, 13, 14], [15, EOS]: "é" is whole by the end of the second round.
    ids = [10, 11, 12, 13, 14, 15, EOS]
    steps, times = split_rounds(ids, _texts("Hello", " wé", "!"), _ByteTokenizer().decode, 9.0)
    assert steps == [[10, 11], [12, 13, 14], [15, EOS]]
    assert times == [1.0, 2.0, 3.0]


def test_split_rounds_merges_a_round_that_streams_no_text():
    # Rounds [10, 11], [12, 13], [14, 15], [EOS]: the second ends inside "é", so streams nothing.
    ids = [10, 11, 12, 13, 14, 15, EOS]
    steps, times = split_rounds(ids, _texts("Hello", " wé!"), _ByteTokenizer().decode, 9.0)
    assert steps == [[10, 11], [12, 13, 14, 15, EOS]]
    assert times == [1.0, 2.0]


def test_split_rounds_without_text_is_one_step():
    assert split_rounds([EOS], [], _ByteTokenizer().decode, 9.0) == ([[EOS]], [9.0])


def _tensorfold_sse(texts, token_ids):
    events = [{"choices": [{"index": 0, "text": text, "finish_reason": None}]} for text in texts]
    end = {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}
    end["tensorfold"] = {"rounds": len(texts), "token_ids": token_ids}
    lines = [f"data: {json.dumps(event)}\n\n" for event in [*events, end]]
    return "".join(lines) + "data: [DONE]\n\n"


def test_text_stream_with_reply_ids_is_split_into_rounds():
    sse = _tensorfold_sse(["Hello", " wé", "!"], [10, 11, 12, 13, 14, 15, EOS])
    model = _model(lambda request: httpx.Response(200, text=sse))
    model._tokenizer = _ByteTokenizer()
    out = _run(model)
    assert out["output_ids"] == [[[10, 11], [12, 13, 14], [15]]]
    assert len(out["token_times"]) == 4


def test_stream_error_is_raised():
    sse = 'data: {"error": {"message": "boom"}}\n\ndata: [DONE]\n\n'
    with pytest.raises(RuntimeError, match="boom"):
        _run(_model(lambda request: httpx.Response(200, text=sse)))


def test_http_error_is_raised_with_status():
    model = _model(lambda request: httpx.Response(404, text="model not found"))
    with pytest.raises(RuntimeError, match="404: model not found"):
        _run(model)


def test_base_url_is_required():
    with pytest.raises(ValueError, match="--base_url"):
        ClientModel("m", max_concurrent_requests=1, sampling_kwargs={})


def _metrics_text(drafts, draft_tokens, accepted, per_pos, generated=0, running=0):
    lines = [
        f'vllm:num_requests_running{{engine="0",model_name="m"}} {running}',
        f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {generated}',
        "# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.",
        f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="m"}} {drafts}',
        'vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7e+09',
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {draft_tokens}',
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {accepted}',
    ]
    lines += [
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="m",position="{i}"}} {v}'
        for i, v in enumerate(per_pos)
    ]
    return "\n".join(lines)


def test_parse_sums_engines_and_keys_positions():
    text = _metrics_text(10, 30, 12, [7, 5]) + "\n" + _metrics_text(10, 30, 8, [5, 3])
    counters = parse_metrics(text.replace('engine="0"', 'engine="1"', 5))
    assert counters[("vllm:spec_decode_num_drafts_total", None)] == 20
    assert counters[("vllm:spec_decode_num_accepted_tokens_total", None)] == 20
    assert counters[("vllm:spec_decode_num_accepted_tokens_per_pos_total", 0)] == 12


def test_server_metric_reports_delta(monkeypatch, tmp_path):
    snapshots = iter(
        [
            _metrics_text(100, 300, 50, [30, 20], generated=1000, running=1),
            _metrics_text(140, 420, 110, [60, 40], generated=1100, running=0),
        ]
    )
    urls = []

    def fake_get(url, **kwargs):
        urls.append(url)
        return httpx.Response(200, text=next(snapshots), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    metric = ServerSpecDecode("http://server:8000/v1")
    metric.directory = str(tmp_path)
    metric.process_final([])
    assert urls[0] == "http://server:8000/metrics"
    assert metric.out["Requests_Running_At_Start"] == 1
    assert metric.out["Server_Generation_Tokens"] == 100
    assert metric.out["Num_Drafts"] == 40
    assert metric.out["Average_AL"] == pytest.approx(1 + 60 / 40)
    assert metric.out["Draft_Acceptance_Rate"] == pytest.approx(60 / 120)
    assert metric.out["Per_Position_Acceptance_Rate"] == {0: 0.75, 1: 0.5}
    assert (tmp_path / "server_spec_decode.json").exists()


def test_server_metric_tolerates_unreachable_metrics(monkeypatch, tmp_path):
    def fake_get(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    metric = ServerSpecDecode("http://server:8000/v1")
    metric.directory = str(tmp_path)
    metric.process_final([])
    assert metric.out == {}


def _tensorfold_metrics_text(rounds, drafted, accepted, generated=0, running=0):
    return "\n".join(
        [
            "# HELP tensorfold:requests_running Requests in prefill or decode.",
            f"tensorfold:requests_running {running}",
            f"tensorfold:generation_tokens_total {generated}",
            f"tensorfold:mtp_drafted_total {drafted}",
            f"tensorfold:mtp_accepted_total {accepted}",
            'tensorfold:request_latency_seconds_bucket{le="0.5"} 2',
            f"tensorfold_health:rounds_total {rounds}",
        ]
    )


def test_server_metric_reads_tensorfold_counters(monkeypatch, tmp_path):
    snapshots = iter(
        [
            _tensorfold_metrics_text(100, 600, 200, generated=1000, running=1),
            _tensorfold_metrics_text(140, 900, 290, generated=1131),
        ]
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, text=next(snapshots), request=httpx.Request("GET", url)
        ),
    )
    metric = ServerSpecDecode("http://server:8000/v1")
    metric.directory = str(tmp_path)
    metric.process_final([])
    assert metric.out["Requests_Running_At_Start"] == 1
    assert metric.out["Server_Generation_Tokens"] == 131
    assert metric.out["Num_Drafts"] == 40
    assert metric.out["Average_AL"] == pytest.approx(1 + 90 / 40)
    assert metric.out["Draft_Acceptance_Rate"] == pytest.approx(90 / 300)
    assert metric.out["Per_Position_Acceptance_Rate"] == {}
