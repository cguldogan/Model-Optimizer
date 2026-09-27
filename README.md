<div align="center">

![Banner image](docs/source/assets/model-optimizer-banner.png)

# NVIDIA Model Optimizer

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/Model-Optimizer)
[![version](https://img.shields.io/pypi/v/nvidia-modelopt?label=Release)](https://pypi.org/project/nvidia-modelopt/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)

[Documentation](https://nvidia.github.io/Model-Optimizer) |
[Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699) |
[Announcement Blogs](https://nvidia.github.io/Model-Optimizer/#announcements)

</div>

______________________________________________________________________

> [!NOTE]
> **This is a personal fork of [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer).**
> It adds one feature, **client mode for `specdec_bench`**, on the
> `feat/specdec-bench-client-mode` branch. Everything else follows upstream. The fork is kept for our
> own use and is not meant to be merged back.

## About this fork: benchmarking a running server with `specdec_bench`

Upstream [`specdec_bench`](./examples/specdec_bench) loads the model itself, inside vLLM, SGLang or
TRT-LLM on the machine that runs the benchmark. It cannot measure a deployment that is already
serving. That rules out our DGX Sparks: they serve GLM-5.3-Flash with tensor parallelism across two
nodes (head on `martin`, worker on `olivia`), with a custom set of patches and the DFlash2 drafter.
That setup uses nearly all of both GPUs, and `specdec_bench` cannot span two nodes anyway.

This fork adds **`--engine CLIENT`**, which benchmarks a running vLLM server over HTTP instead.

**What it gives us**

- **We can benchmark the deployment we actually run**, including its patches, multi-node setup and
  drafter, without taking the Sparks down or reserving GPU memory for a second copy of the model.
- **The same metrics and outputs as upstream.** MT-Bench, SpecBench, SPEED-Bench and random-token
  datasets all work, and the run writes the usual `acceptance_rate.json`, `timing.json` and
  per-category acceptance length (AL). Results can be compared directly with in-process runs.
- **The server's own view of acceptance.** Each run also writes `server_spec_decode.json`, built from
  the change in the server's `vllm:spec_decode_*` Prometheus counters. It gives server-side average
  AL, draft acceptance rate, and acceptance for each draft position. This checks the client-side
  numbers.
- **A warning when the numbers are contaminated.** The counters are server-wide. The run warns when
  other requests are already running at start, and reports the server's generated-token count so
  you can compare it with the client's.
- **No GPU and no engine install on the client.** The benchmark runs from a laptop and needs only
  Python with `transformers` and `httpx`.

**How it works.** Prompts are chat-templated and tokenized on the client, then sent as token ids to
`/v1/completions` with streaming on. vLLM streams one chunk per decoding step, and each chunk
carries `token_ids` (vLLM's `return_token_ids`, available from vLLM 0.10.2). The number of tokens in
a chunk is that step's accepted draft tokens plus one. This is the same per-step signal that
upstream's in-process vLLM wrapper reads.

**Run it against the Sparks**

```bash
cd examples/specdec_bench
python3 run.py \
    --engine CLIENT \
    --base_url http://100.90.44.53:8888/v1 \
    --model_dir GLM-5.3-Flash-EXL3 \
    --tokenizer /path/to/glm-tokenizer \
    --mtbench question.jsonl \
    --output_length 1024 \
    --concurrency 2 \
    --tp_size 2 \
    --save_dir results/glm-sparks
```

- `--model_dir` is the served model name.
- `--tokenizer` must be the server's tokenizer and chat template. For GLM, copy `tokenizer*.json`
  from the model snapshot on `martin` and use the deployment's `files/chat_template.jinja` as the
  chat template.
- The server runs with `--max-num-seqs 2`. Any higher concurrency just waits in the server's queue,
  and that wait shows up as time to first token.

**First results** (2026-09-26): GLM-5.3-Flash with the DFlash2 drafter (7 draft tokens), TP=2 across
two Sparks, full MT-Bench (80 questions, 2 turns each), 1024 output tokens.

| Metric | Value |
|---|---|
| Average acceptance length, averaged per request | 3.88 |
| Acceptance length over all tokens (client / server counters) | 3.39 / 3.44 |
| Acceptance at each draft position, 1 → 7 | 0.75, 0.54, 0.38, 0.28, 0.20, 0.16, 0.12 |
| Acceptance length by category | math 5.68 · extraction 5.23 · coding 4.39 · reasoning 3.65 · writing 3.26 · stem 3.24 · roleplay 2.89 · humanities 2.71 |
| Tokens per second for one request (median) | 28 |

One other request was running on the server during this run. About 4% of the server's generated
tokens were not ours, so these acceptance numbers are reliable but the latency figures are not.

The full usage notes are in the
[client mode section](./examples/specdec_bench/README.md#benchmarking-a-running-server-client-mode)
of the `specdec_bench` README. The code is in
[`models/client.py`](./examples/specdec_bench/specdec_bench/models/client.py) and
[`metrics/server_spec_decode.py`](./examples/specdec_bench/specdec_bench/metrics/server_spec_decode.py).

______________________________________________________________________

**NVIDIA Model Optimizer** (referred to as **Model Optimizer**, or **ModelOpt**) is a library comprising state-of-the-art model optimization [techniques](#techniques) including quantization, pruning, Neural Architecture Search (NAS), distillation, speculative decoding and sparsity to accelerate models.

**[Input]** Model Optimizer currently supports inputs of a [Hugging Face](https://huggingface.co/), [PyTorch](https://github.com/pytorch/pytorch) or [ONNX](https://github.com/onnx/onnx) model.

**[Optimize]** Model Optimizer provides Python APIs for users to easily compose the above model optimization techniques and export an optimized quantized checkpoint.
Model Optimizer is also integrated with [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for training required inference optimization techniques.

**[Export for deployment]** Seamlessly integrated within the NVIDIA AI software ecosystem, the quantized checkpoint generated from Model Optimizer is ready for deployment in downstream inference frameworks like [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization), [TensorRT](https://github.com/NVIDIA/TensorRT), or [vLLM](https://github.com/vllm-project/vllm). The unified Hugging Face export API now supports both transformers and diffusers models.

## Latest News

- [2026/09/16] [**End-to-end W4A4 NVFP4 + QAD tutorial for Qwen3.6-35B-A3B**](./examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B): NVFP4 W4A4 PTQ plus quantization-aware distillation, reaching up to 1.30x vLLM throughput over BF16 and 3.1x smaller checkpoints while recovering the accuracy W4A4 costs.
- [2026/09/09] [BLOG: Improving NVFP4 Accuracy with Local-Hessian Weight Scales](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html)
- [2026/08/24] [BLOG: AutoQuantize: A Fast Automatic Mixed-Precision Assignment](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html)
- [2026/08/17] [BLOG: Developing Nemotron 3.5 Lightning NVFP4 with QAD Using NVIDIA Model Optimizer](https://developer.nvidia.com/blog/developing-nemotron-3-5-lightning-nvfp4-with-qad-using-nvidia-model-optimizer/): Learn how quantization-aware distillation recovers accuracy from aggressive NVFP4 quantization while reducing model size and increasing throughput.
- [2026/06/26] [BLOG: Creating the NVIDIA Nemotron 3 Ultra NVFP4 Checkpoint with NVIDIA Model Optimizer](https://developer.nvidia.com/blog/creating-the-nvidia-nemotron-3-ultra-nvfp4-checkpoint-with-nvidia-model-optimizer/): How we quantized Nemotron 3 Ultra (550B) to NVFP4 with Model Optimizer — up to 5.9× higher decode-heavy inference throughput than GLM-5.1 754B FP4 while matching BF16 accuracy. [NVFP4 Checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) on Hugging Face.
- [2026/05/27] [**End-to-end Optimization tutorial for Nemotron-3-Nano-30B-A3B**](./examples/megatron_bridge/tutorials/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16): Pruning + two-phase distillation + FP8 quantization achieving 2.6× vLLM throughput and 2.6× memory reduction.
- [2026/05/13] [**Puzzletron**](./examples/puzzletron): A new algorithm for heterogeneous pruning & NAS of LLM and VLM models.
- [2026/04/15] Customer story: [Domyn compresses Colosseum-355B → 260B using ModelOpt's Minitron pruning + distillation](https://www.domyn.com/blog/domyn-large-the-journey-of-a-european-sovereign-ai-model-for-regulated-industries)
- [2026/03/17] Customer story: [Bielik.AI builds Bielik Minitron 7B (33% smaller, 50% faster, 90% quality retained) using ModelOpt's Minitron pruning + distillation](https://bielik.ai/en/nvidia-gtc-bielik-minitron-premiere/)
- [2026/03/11] Model Optimizer quantized Nemotron-3-Super checkpoints are available on Hugging Face for download: [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4). Learn more in the [Nemotron 3 Super release blog](https://blogs.nvidia.com/blog/nemotron-3-super-agentic-ai/). Check out how to quantize Nemotron 3 models for deployment acceleration [here](./examples/hf_ptq/README.md)
- [2026/03/11] [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) now supports Nemotron-3-Super quantization (PTQ and QAT) and export workflows using the Model Optimizer library. See the [Quantization (PTQ and QAT) guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/super-v3/docs/models/llm/nemotron3-super.md#quantization-ptq-and-qat) for FP8/NVFP4 quantization and HF export instructions.
- [2025/12/11] [BLOG: Top 5 AI Model Optimization Techniques for Faster, Smarter Inference](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/)
- [2025/12/08] NVIDIA TensorRT Model Optimizer is now officially rebranded as NVIDIA Model Optimizer.
- [2025/10/07] [BLOG: Pruning and Distilling LLMs Using NVIDIA Model Optimizer](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/)
- [2025/09/17] [BLOG: An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [2025/09/11] [BLOG: How Quantization Aware Training Enables Low-Precision Accuracy Recovery](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)
- [2025/08/29] [BLOG: Fine-Tuning gpt-oss for Accuracy and Performance with Quantization Aware Training](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/)
- [2025/08/01] [BLOG: Optimizing LLMs for Performance and Accuracy with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
- [2025/06/24] [BLOG: Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [2025/05/14] [NVIDIA TensorRT Unlocks FP4 Image Generation for NVIDIA Blackwell GeForce RTX 50 Series GPUs](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
- [2025/04/21] [Adobe optimized deployment using Model-Optimizer + TensorRT leading to a 60% reduction in diffusion latency, a 40% reduction in total cost of ownership](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
- [2025/04/05] [NVIDIA Accelerates Inference on Meta Llama 4 Scout and Maverick](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/). Check out how to quantize Llama4 for deployment acceleration [here](./examples/hf_ptq/README.md#support-matrix)
- [2025/03/18] [World's Fastest DeepSeek-R1 Inference with Blackwell FP4 & Increasing Image Generation Efficiency on Blackwell](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- [2025/02/25] Model Optimizer quantized NVFP4 models available on Hugging Face for download: [DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4), [Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4), [Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4)
- [2025/01/28] Model Optimizer has added support for NVFP4. Check out an example of NVFP4 PTQ [here](./examples/hf_ptq/README.md#getting-started).
- [2025/01/28] Model Optimizer is now open source!

<details close>
<summary>Previous News</summary>

- [2024/10/23] Model Optimizer quantized FP8 Llama-3.1 Instruct models available on Hugging Face for download: [8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8), [70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8), [405B](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8).
- [2024/09/10] [Post-Training Quantization of LLMs with NVIDIA NeMo and Model Optimizer](https://developer.nvidia.com/blog/post-training-quantization-of-llms-with-nvidia-nemo-and-nvidia-tensorrt-model-optimizer/).
- [2024/08/28] [Boosting Llama 3.1 405B Performance up to 44% with Model Optimizer on NVIDIA H200 GPUs](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-performance-by-up-to-44-with-nvidia-tensorrt-model-optimizer-on-nvidia-h200-gpus/)
- [2024/08/28] [Up to 1.9X Higher Llama 3.1 Performance with Medusa](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)
- [2024/08/15] New features in recent releases: [Cache Diffusion](./examples/diffusers/cache_diffusion), [QLoRA workflow with NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/24.09/sft_peft/qlora.html), and more. Check out [our blog](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/) for details.
- [2024/06/03] Model Optimizer now has an experimental feature to deploy to vLLM as part of our effort to support popular deployment frameworks. Check out the workflow [here](./examples/hf_ptq/README.md#vllm)
- [2024/05/08] [Announcement: Model Optimizer Now Formally Available to Further Accelerate GenAI Inference Performance](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- [2024/03/27] [Model Optimizer supercharges TensorRT-LLM to set MLPerf LLM inference records](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)
- [2024/03/18] [GTC Session: Optimize Generative AI Inference with Quantization in TensorRT-LLM and TensorRT](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/)
- [2024/03/07] [Model Optimizer's 8-bit Post-Training Quantization enables TensorRT to accelerate Stable Diffusion to nearly 2x faster](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- [2024/02/01] [Speed up inference with Model Optimizer quantization techniques in TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md)

</details>

## Install

To install stable release packages for Model Optimizer with `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/):

```bash
pip install -U nvidia-modelopt[all]
```

Model Optimizer will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

To install from source in editable mode with all development dependencies or to use the latest features, run:

```bash
# Clone the Model Optimizer repository
git clone git@github.com:NVIDIA/Model-Optimizer.git
cd Model-Optimizer

pip install -e .[dev]
```

You can also directly use NVIDIA container images, which have Model Optimizer pre-installed:

- `nvcr.io/nvidia/pytorch:<version>-py3`
- `nvcr.io/nvidia/nemo:<version>`
- `nvcr.io/nvidia/tensorrt-llm/release:<version>`

Before pulling and using the container images, please review their respective license terms.
Make sure to upgrade Model Optimizer to the latest version as described above.
Visit our [installation guide](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for
more fine-grained control on installed dependencies or for alternative docker images and environment variables to setup.

## Techniques

<div align="center">

| **Technique** | **Description** | **Examples** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Post Training Quantization | Compress model size by 2x-4x, speeding up inference while preserving model quality! | \[[HF LLMs / VLMs](./examples/hf_ptq/)\] \[[Megatron-Bridge LLMs / VLMs](./examples/megatron_bridge/)\] \[[Diffusers](./examples/diffusers/)\] \[[ONNX](./examples/onnx_ptq/)\] \[[Windows](./examples/windows/)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| Quantization Aware Training / Distillation | Refine accuracy of quantized models even further with a few training steps! | \[[Hugging Face](./examples/llm_qat/)\] \[[Megatron-Bridge](./examples/megatron_bridge)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| Pruning | Reduce your model parameters or memory footprint and accelerate inference by removing unnecessary weights! | \[[General](./examples/pruning/)\] \[[Megatron-Bridge](./examples/megatron_bridge/)\] | |
| Distillation | Reduce deployment model size by teaching small models to behave like larger models! | \[[Hugging Face](./examples/llm_distill/)\] \[[Megatron-Bridge](./examples/megatron_bridge/)\] \[[Megatron-LM](./examples/llm_distill/README.md#knowledge-distillation-kd-in-nvidia-megatron-lm-framework)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html)\] |
| Speculative Decoding | Train draft modules to predict extra tokens during inference! | \[[Hugging Face](./examples/speculative_decoding/)\] \[[Megatron-LM](./examples/speculative_decoding#mlm-example)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html)\] |
| Sparsity | Efficiently compress your model by storing only its non-zero parameter values and their locations | \[[Hugging Face](./examples/llm_sparsity/)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/6_sparsity.html)\] |

</div>

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints \[[🤗 Hugging Face - Nvidia Model Optimizer Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)\]
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- More models coming soon!

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](./examples/benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

## Model Support Matrix

| Model Type | Support Matrix |
|------------|----------------|
| LLM / VLM Quantization | [View Support Matrix](./examples/hf_ptq/README.md#support-matrix) |
| Diffusers Quantization | [View Support Matrix](./examples/diffusers/README.md#support-matrix) |
| ONNX Quantization | [View Support Matrix](./examples/torch_onnx/README.md#onnx-export-supported-llm-models) |
| Windows Quantization | [View Support Matrix](./examples/windows/README.md#support-matrix) |
| Quantization Aware Training | [View Support Matrix](./examples/llm_qat/README.md#support-matrix) |
| Pruning | [View Support Matrix](./examples/pruning/README.md#support-matrix) |
| Distillation | [View Support Matrix](./examples/llm_distill/README.md#support-matrix) |
| Speculative Decoding | [View Support Matrix](./examples/speculative_decoding/README.md#support-matrix) |

## Deprecation Policy

Model Optimizer follows a structured approach to managing deprecated features:

- **Communication:** Deprecation notices are documented in the [Changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html). Deprecated items include source code statements indicating deprecation timing, with runtime warnings issued upon use.
- **Migration Period:** Since Model Optimizer is still pre-1.0, we provide a 1-release (~1-month) migration period after deprecation. During this window, deprecated features continue functioning while issuing warnings.
- **Scope:** The policy addresses both complete deprecations (entire APIs removed) and partial ones (specific parameters removed while methods remain).
- **Removal:** Following the migration period, deprecated elements are removed in alignment with semantic versioning standards, potentially including breaking changes in minor version updates while Model Optimizer remains in 0.x.

## Citation

If you use NVIDIA Model Optimizer in your research, please cite it as follows:

```bibtex
@misc{nvidia-modelopt,
  author       = {{NVIDIA Corporation}},
  title        = {{NVIDIA Model Optimizer}},
  howpublished = {\url{https://github.com/NVIDIA/Model-Optimizer}},
  year         = {2024--2026},
  note         = {GitHub repository}
}
```

## Contributing

Model Optimizer is now open source! We welcome any feedback, feature requests and PRs.
Please read our [Contributing](./CONTRIBUTING.md) guidelines for details on how to contribute to this project.

## AI Agents

ModelOpt's agent skills can be installed from this repository and used in any
workspace.

### Claude Code

```bash
claude plugin marketplace add https://github.com/NVIDIA/Model-Optimizer.git
claude plugin install modelopt@modelopt
```

### Codex

```bash
codex plugin marketplace add https://github.com/NVIDIA/Model-Optimizer.git
```

Then open `/plugins`, select the `modelopt` marketplace, and install `modelopt`.
Contributors can also use the skills directly from a checkout. See the
[agent tooling notes](./.agents/TOOLING.md).

### Top Contributors

[![Contributors](https://contrib.rocks/image?repo=NVIDIA/Model-Optimizer)](https://github.com/NVIDIA/Model-Optimizer/graphs/contributors)

Happy optimizing!
