# Agent Instructions for ModelOpt

These instructions apply to AI-assisted work in this repository.

## Repository orientation

- Start with `README.md` for project overview and install.
- Use `modelopt/` for source, `tests/` for focused test coverage, and
  `examples/` or `docs/` for usage patterns.
- **Agent skills live under `plugins/modelopt/skills/`**, the installable
  plugin's canonical skill tree. `.agents/skills` and `.claude/skills` expose
  those skills through relative symlinks. Shared agent config and scripts
  remain under `.agents/`. See `.agents/README.md` for the convention.

## This fork: specdec_bench client mode

This is a personal fork. Its only addition is `--engine CLIENT` in `examples/specdec_bench`
(`specdec_bench/models/client.py`, `specdec_bench/metrics/server_spec_decode.py`, tests in
`tests/examples/specdec_bench/test_client.py`). It is not upstreamed; never open a PR to NVIDIA.
Work on the `feat/specdec-bench-client-mode` branch, which is also the fork's default branch.

### Unit tests (no GPU, no server)

Use a uv venv on Python 3.12. The macOS system `python3` (3.14) has broken arm64/x86_64 wheels.
`tests/conftest.py` imports torch and modelopt, and `pyproject.toml` adds cov, instafail and
timeout flags, so all of these are required even for this small suite:

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e . torch httpx numpy pyyaml transformers tqdm datasets rich \
    seaborn tiktoken jinja2 boto3 pytest pytest-cov pytest-instafail pytest-timeout pre-commit
.venv/bin/python -m pytest tests/examples/specdec_bench -q --no-cov   # expect 62 passed
.venv/bin/pre-commit run --files <changed files>                    # ruff, mypy, markdownlint, license
```

### Live benchmark against the DGX Sparks

The Sparks are `Martin` (head) and `Olivia` (worker). Reach them with `ssh Martin` / `ssh Olivia`
through the NVIDIA Sync ssh_config; plain `ssh martin` fails the host-key check. They serve
GLM-5.3-Flash (DFlash2 drafter) with TP=2 across both nodes, at `http://100.90.44.53:8888/v1`
over Tailscale. Since 2026-10 the server is tensorfold (container `glm53-flash-tf`, `--parallel 4`,
adaptive draft depth); the README results up to 2026-09-28 came from vLLM (`glm53-exl3-*`,
`--max-num-seqs 2`, 7 draft tokens). Check which one runs with `docker ps` on Martin. GPU memory
is full, so only `--engine CLIENT` works. Never stop or restart the serving containers without
the user's approval. Run the client from the Mac.

1. Check that the server is up and idle. If requests are running, the numbers will be
   contaminated: wait, or tell the user.

   ```bash
   curl -s http://100.90.44.53:8888/metrics | grep -E '^(vllm:num_requests|tensorfold:requests)_(running|waiting)'
   ```

2. Fetch the server's tokenizer and chat template, and the MT-Bench prompts, once:

   ```bash
   W=~/specdec-runs; mkdir -p $W/glm-tok
   ssh Martin 'cd ~/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/*/ && tar chzf - tokenizer_config.json tokenizer.json config.json generation_config.json -C ~/src/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/files chat_template.jinja' | tar xzf - -C $W/glm-tok
   curl -sL -o $W/question.jsonl https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/resolve/main/raw/question.jsonl   # 80 lines
   ```

3. Do a smoke run (about 30 s), then the full run (about 50 min). Run the full one in the
   background and watch its log for tqdm progress, `Traceback`, or `Error processing`.

   ```bash
   cd examples/specdec_bench
   bench() { ../../.venv/bin/python run.py --engine CLIENT --base_url http://100.90.44.53:8888/v1 \
       --model_dir GLM-5.3-Flash-EXL3 --tokenizer $W/glm-tok --mtbench $W/question.jsonl --tp_size 2 "$@"; }
   bench --num_requests 2 --output_length 256 --concurrency 1 --save_dir $W/smoke
   bench --output_length 1024 --concurrency 2 --show_progress --save_dir $W/mtbench-$(date +%F) \
       > $W/mtbench-$(date +%F).log 2>&1
   ```

   Keep `--concurrency` at or below the server's limit (2 on vLLM, 4 on tensorfold). Anything
   higher only waits in the server queue and inflates time to first token.

4. Validate before you report numbers. The log must not contain `already has N request(s)
   running`. In `server_spec_decode.json`, `Server_Generation_Tokens` must equal the client's
   token count, which is the sum of `length × count` over `Acceptance_Length_Histogram` in
   `acceptance_rate.json`. The client's step count should exceed the server's `Num_Drafts` by
   exactly the number of turns (160 for full MT-Bench). On tensorfold a few rounds that end inside
   a multi-byte character merge into one client step, so the step excess is a little lower (96 on
   2026-10-04); `Num_Drafts + Num_Accepted_Tokens + 160` must still equal the token count.

5. Report `Average_AL` (the per-request mean), the acceptance length over all tokens from the
   client and from the server, the per-position acceptance rates (vLLM only), `Category_AL`, and
   the median time to first token and request tokens/s from `timing.json`. Compare them against
   the results table in `README.md`, and update that table only if the user asks.

## Coding guidelines

- **Coding guide:** Code development and review require reading and following
  the [coding standards in CONTRIBUTING.md](CONTRIBUTING.md#-coding-standards);
  do not skip this step.
- **Use relative paths** from the repo root in commands and file references.

## Iterative development

- **Running tests:** Follow the
  [writing and running tests](CONTRIBUTING.md#-writing-and-running-tests)
  instructions. For fast initial iteration, choose focused tests for the
  changed area from `tests/`.
- **Running pre-commit:** Follow the
  [pre-commit hook instructions](CONTRIBUTING.md#pre-commit-hooks). Hooks may
  modify files; review and re-stage those changes before committing.
- **Signed commit:** Use `git commit -s -S -m "<message>"` for commits so they
  follow the [signing your work](CONTRIBUTING.md#-signing-your-work)
  requirements.
- **Never `git push` without explicit approval in the current turn.** Commit
  locally is fine; publishing to a remote is not.
- After `git commit`, stop and wait for the user to say "push", "publish",
  "ship", or equivalent before running `git push`, `gh pr create`, or any
  push-option flags like `-o merge_request.create`.

## Updating skills

- **Keep skill edits concise.** Skills are loaded into agent context, so every
  line costs tokens on each use. Add only what changes agent behavior, and
  prefer tightening existing text over appending new text.
- **Compress before opening the PR.** Make a final pass over the skill diff:
  drop unnecessary explanations and examples, cut redundancy, and merge
  overlapping guidance.

## Sizing and splitting PRs

- **Keep each PR that goes up for review under ~500 added lines of source.**
  Large PRs stall in review; exceed the budget only when the change genuinely
  cannot be split — a mechanical rename, generated files, or a self-contained
  drop such as a new example or a new model/backend that has no working
  intermediate state. Deletions, tests, and docs don't count; check the
  insertions from `git diff --shortstat <base>...HEAD -- . ':!tests' ':!docs'`
  before opening.
- **Propose the split before opening an oversized PR.** When the work in flight
  is already over budget, offer a series of smaller PRs and, once the user
  agrees, do the split — don't open the big one and ask afterwards.
- **Split on structure first, features second.** Look for a file, directory, or
  module boundary that carves the change into independent pieces. Only when no
  such boundary exists, split by feature: land the enabling refactor or
  plumbing first, then one PR per behavior it unlocks.
- **Keep the series acyclic and linearly ordered.** A sub-PR may depend only on
  ones earlier in the series; if two pieces need each other, they belong in the
  same PR, or their shared part belongs in an earlier one. State the merge
  order in each description.
- **Prefix titles with `[x/N]`** — e.g. `[2/4] Add NVFP4 export path` — so
  reviewers know it is one slice of a planned split, and link the sibling PRs.
- **Optional but strongly recommended: build and ship the series with `gh stack`**
  (the `github/gh-stack` extension). `gh stack init <branch1 ... branchN>` (bottom
  to top) turns the split branches into a linear stack, `gh stack submit` pushes
  them and opens the PRs in order with the stack cross-links added automatically —
  you still write each `[x/N]` title and description in its editor — and
  `gh stack sync` / `gh stack rebase` keep the upper slices consistent as the lower
  ones merge. Pushing and wiring each PR by hand works too.
- **Every sub-PR stands on its own:** it builds, it carries unit tests for the
  code it introduces, and CI passes on it without the later PRs.
- **When a split makes the whole hard to follow, the full change may also go
  up as a draft PR** — for reference only, never as a second review request.
  Keep it in draft, say in its description that it is not for merge, link it
  from each sub-PR, and list the sub-PRs in it. Rebase or close it as the
  series lands.

## Contributing and PR readiness

- Before opening or marking a PR ready for review, read the
  [submitting your code](CONTRIBUTING.md#submitting-your-code) guidance.
- Read `.github/PULL_REQUEST_TEMPLATE.md` and satisfy the checklist.
- **PR description:** fill the template sections — what changed and why, a usage
  snippet if it adds an API or flag, and what you actually ran under Testing.
  Root cause, benchmark numbers, and design rationale belong here. Don't restate
  the diff file by file.
- **Only changelog-worthy changes get a `CHANGELOG.rst` entry:** new features,
  backward breaking changes, deprecations, and fixes for critical or known bugs
  from a previous release. Skip bugs introduced and fixed within the same
  unreleased cycle.
- **Keep each entry to one or two sentences** written for external users: what
  changed and what they need to do. No internal bug numbers (e.g. NVBug IDs),
  root-cause analysis, or implementation detail — that belongs in the PR
  description. File features under the matching `**New Features**` sub-section
  used by recent releases (e.g. `*Quantization*`, `*Speculative Decoding*`,
  `*Megatron Framework (M-LM / M-Bridge)*`, `*Misc*`) rather than relabeling
  existing ones.

## Responding to PR review feedback

- **Judge each comment on its merits before acting.** Check it against the
  current code — reviewers comment on stale diffs, and bot findings (CodeRabbit,
  Claude) are claims to verify, not instructions. Weight CODEOWNERS reviewers
  above bots; if a reviewer reaffirms after your pushback, that settles it.
- **Pick one outcome per thread:** address it in a commit, push back citing the
  code that shows the comment is wrong, or postpone it as out of scope. Report
  which threads got which when you ask for push approval.
- **Reply in every thread the pushed commits addressed** — a sentence on what
  changed and where. Those replies need no extra approval; pushback and postpone
  replies do, since no commit backs them. Never resolve threads: that is the
  reviewer's call.
