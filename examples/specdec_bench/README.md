# Speculative Decoding (SpecDec) Bench

## Installation

This benchmark is meant to be a lightweight layer ontop of an existing vLLM/SGLang/TRTLLM installation. For example, no install
is required if one is running in the following dockers: `vllm/vllm-openai:v0.11.0` (vLLM), `lmsysorg/sglang:v0.5.4.post2` (SGLang), or
`nvcr.io/nvidia/tensorrt-llm/release:1.2.0` (TRT-LLM).

Next

```bash
cd examples/specdec_bench
```

## Purpose

Collect relevant metrics on acceptance rate, timing, and outputs for Speculative Decoding methods.
Acceptance rate refers to the number of tokens generated on every iteration.  For a standard Autoregressive LLM, this number
is just 1.

## Getting Started

A basic example run script is provided which benchmarks MTBench (a standard 160 prompts spanning 8 categories).
MTBench is available [here](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)

### Running MTBench on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py \
    --model_dir openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --draft_model_dir /path/to/eagle \
    --mtbench question.jsonl \
    --tp_size 1 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --num_requests 80 \
    --engine TRTLLM \
    --concurrency 1 \
    --postprocess gptoss
```

### Running Random ids on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py \
    --model_dir openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --draft_model_dir /path/to/eagle \
    --random_isl 1024 \
    --tp_size 1 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --num_requests 40 \
    --engine TRTLLM \
    --concurrency 1
```

### Running [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) on Llama 3.3 70B + Eagle 3

1. Install the requirements file using `pip install -r requirements.txt`

2. Prepare the data using the provided script:

```bash
python3 prepare_data.py --dataset speed --config all
```

The data will be saved to `data/` directory, each config type (qualitative, throughput_1k, ...) to each own directory.

#### License

GOVERNING TERMS: This dataset is governed by the NVIDIA Evaluation Dataset License Agreement.

ADDITIONAL INFORMATION: MIT for bigcode/humanevalpack, RUCAIBox/MMATH, RUCAIBox/BAMBOO and EQ-Bench. Apache 2.0 for Writing Bench and Spec-Bench. CC BY 4.0 for FBK-MT/MCIF. MIT and Apache 2.0 for tianyang/repobench_python_v1.1, JetBrains-Research/lca-project-level-code-completion and tianyang/repobench_java_v1.1.

NOTICE: For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The `prepare_data.py` script automatically fetches data from all the source datasets.

Additional details are in [HuggingFace dataset repository](https://huggingface.co/datasets/nvidia/SPEED-Bench).

#### Qualitative split

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/qualitative \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress
```

#### Throughput split

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/throughput_1k \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress
```

For longer context (>8192 tokens), please use the following configuration when using TRTLLM:

```yaml
engine_args:
  max_seq_len: 131072   # Model max context length (for Llama 3.3 70B)
  enable_chunked_prefill: true
```

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/throughput_16k \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress \
    --runtime_params runtime_args_long_context.yaml
```

### Benchmarking a running server (client mode)

`--engine CLIENT` benchmarks a server that is already running instead of loading the model
in-process. This helps when the deployment is multi-node or already running with its own
patches, or when the GPUs are shared. Use it with a vLLM server that supports `return_token_ids`
(vLLM >= 0.10.2). The client needs no GPU, and neither vLLM nor SGLang nor TRT-LLM has to be
installed. Pass the served model name as `--model_dir` and the server's tokenizer as `--tokenizer`:

```bash
python3 run.py \
    --engine CLIENT \
    --base_url http://spark-head:8000/v1 \
    --model_dir <served-model-name> \
    --tokenizer /path/to/tokenizer \
    --mtbench question.jsonl \
    --output_length 1024 \
    --concurrency 8 \
    --tp_size 2 \
    --save_dir results/client
```

Prompts are tokenized and chat-templated on the client, then sent as token ids to
`/v1/completions` with streaming on. Each streamed chunk counts as one engine step, so the
acceptance length comes from chunk sizes, the same way the in-process vLLM wrapper measures it.
The server owns the speculative-decoding setup, so `--draft_model_dir`, `--draft_length` and
`--speculative_algorithm` do nothing in this mode. `--api_key` (or `OPENAI_API_KEY`) sets a bearer
token. `--runtime_params` `engine_args.timeout` sets a per-request timeout, and
`engine_args.extra_body` is merged into every request.

The run also writes `server_spec_decode.json`. It holds the acceptance the server itself reports,
taken from the difference in its `vllm:spec_decode_*` Prometheus counters over the run. These
counters are server-wide, so any other traffic during the run is counted too. The run warns when
requests are already running at start. Compare `Server_Generation_Tokens` with the client's output
token count. A server started with `--stream-interval` greater than 1 merges steps into one chunk,
which inflates the client-side acceptance length.

## Uploading results to S3

Each `run.py` invocation writes a result directory containing `configuration.json`,
`timing.json`, `acceptance_rate.json`, and (when applicable) `mtbench.json` / `specbench.json`.
`upload_to_s3.py` is a single-file, drop-in tool that uploads one run — or an entire sweep —
to any S3-compatible bucket:

```bash
python upload_to_s3.py /path/to/run_or_sweep_dir s3://your-bucket/some/prefix \
    --endpoint https://your-s3-endpoint \
    --key-id YOUR_KEY_ID \
    --secret YOUR_SECRET
```

`--endpoint`, `--key-id`, and `--secret` default to the `S3_ENDPOINT`, `S3_KEY_ID`, and
`S3_SECRET` environment variables. Omit `--endpoint` (or set `S3_ENDPOINT=""`) to use AWS S3's
default endpoint. Use `--dry-run` to preview the upload plan, and `--skip-existing` to skip
runs already present at the destination instead of failing.

The tool handles two directory layouts and mirrors them into S3:

- **Flat** — `LOCAL_DIR/run_name/{configuration,timing,...}.json`
- **Sweep** — `LOCAL_DIR/sweep_name/run_name/{configuration,timing,...}.json`

`LOCAL_DIR`'s basename is preserved in the destination prefix, so re-uploads from the same
source land in the same place.

## Notes

The goal of this benchmark is to provide an easy way to configure, run, and compare speculative implementations across frameworks in an apples-to-apples method.
This benchmark sends request in a single-threaded fashion, so running large concurrency (>256) may result in python async scheduling delays and skew metrics.
If larger concurrency is needed, it is recommended to fully deploy the model using `vllm serve`, `python -m sglang.launch_server`, or `trtllm-serve` (for vLLM, SGlang, or TRTLLM respectively) and
use a more robust benchmarking client like NVIDIA AI Perf.
