# LTX-2.5 in Docker (CUDA 13.2, cloud-GPU ready)

A self-contained Docker setup for [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), Lightricks'
open-weights audio+video model. It wraps the upstream `Lightricks/LTX-2` repo (`uv sync`, `ltx_pipelines`
CLI) in a CUDA 13.2 image, keeps the ~71 GiB of gated weights on a mounted volume, and drives the
`DistilledPipeline` with settings sized for a **single 24 GB GPU** (RTX 3090 / 4090).

> **Heads-up for this PC, first.** This machine (HP Z440) cannot run LTX-2.5 either natively or in a
> container. The GPU is a **Quadro K5200 — Kepler, 2014, 8 GB VRAM, compute capability 3.0**. LTX-2.5
> needs Ampere or newer and 32 GB+ VRAM, and the K5200's last driver (462.59) tops out at CUDA 11.x
> while this image needs CUDA 13.2. WSL2 also isn't installed, which Docker Desktop requires. Details
> and numbers are in [Why this PC can't run it](#why-this-pc-cant-run-it). The stack below targets a
> **cloud GPU instance** (or a machine with a modern NVIDIA card).

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA **Ampere or newer**, **24 GB VRAM**, compute capability ≥ 8.0 | 32–48 GB (RTX 5090, A6000, A100, L40S) |
| NVIDIA driver | **≥ 580** (CUDA 13.2 runs on any 13.x-capable driver; R595 is the matching branch) | R595 / R610 |
| System RAM | 32 GB | 64 GB+ (needed by `--offload cpu`) |
| Disk | 100 GB free | 200 GB SSD |
| Docker | Docker Engine 24+, NVIDIA Container Toolkit, `docker compose` v2 | — |

Rough disk budget: image ~15 GB + weights ~71 GiB (76 GB) + HF cache/working space. Keep **120 GB+**
free to be comfortable. The a2vid path adds the full dev transformer and a distilled LoRA
(**+~51 GiB**, so ~300 GB free in total); see [Audio-driven video](#audio-driven-video-a2vid).

### Why CUDA 13.2 and not something older?

Upstream `ltx-core` pins **torch 2.13.0+cu132** from `download.pytorch.org/whl/cu132`, and the `natten`
extra (the fast diffusion-VAE decoder) ships only as `natten==0.21.7+torch2130cu132`. A `cu128`/`cu124`
base image resolves to a torch the natten wheel cannot link against. This is not a tunable — it's the
upstream lock.

---

## Quick start (cloud GPU)

Pick an instance image with a **driver ≥ 580** and ≥ 24 GB VRAM, then:

```bash
git clone <this-dir> ltx25-docker && cd ltx25-docker
cp .env.example .env          # optional: set MODELS_HOST_DIR / OUTPUT_HOST_DIR

# 1. Build the image (~10-20 min, needs network; ~15 GB)
./scripts/run.sh build

# 2. Accept the model terms, then download the weights (~71 GiB)
#    https://huggingface.co/Lightricks/LTX-2.5   (gated)
export HF_TOKEN=hf_xxxxxxxx
./scripts/run.sh download

# 3. Preflight: checks GPU, driver, VRAM, RAM, disk, weights
./scripts/run.sh

# 4. Generate (~5 s of video at 24 fps, 768x1280)
./scripts/run.sh run --prompt "a rusty robot walking through a rain-soaked market at night"
```

Output lands in `./output/ltx25-<timestamp>-<W>x<H>-<N>f.mp4` (video **and** synchronized audio).

### Plain docker compose (no wrapper)

```bash
docker compose build
docker compose run --rm ltx2                      # preflight
docker compose run --rm ltx2 run --width 1024 --height 1536
docker compose --profile download run --rm downloader
```

On Windows/Git Bash prefer `scripts/run.sh`: it sets `MSYS_NO_PATHCONV` so container paths aren't
rewritten into `C:/...` by the shell.

### Building the image without a local Docker

This PC's shell cannot reach the internet over HTTPS, so the image is best built where Docker and
network access exist. [`.github/workflows/build-image.yml`](.github/workflows/build-image.yml)
builds it on a GitHub runner and pushes to GHCR — free, no local Docker, and it includes a GPU-less
smoke test that imports the stack.

That image is also what **Vast.ai** needs, because Vast runs your image as the instance container and
does not allow `docker build` inside instances. See [docs/VAST.md](docs/VAST.md) for the
build → push → rent workflow, host driver requirements, and weight fetching on the instance.

---

## The 24 GB profile

The official minimum is 32 GB VRAM, so at 24 GB you run in a compressed regime. The defaults encode
what fits:

```text
--quantization fp8-cast   bf16 42 GB transformer -> ~21 GB on load, upcast on the fly
--offload cpu             the 26 GB Gemma-4 12B text encoder lives in host RAM
768x1280, 121 frames      stage 1 at 768x1280 (~9.2k latent tokens), stage 2 upsamples 2x
```

Why the offload is not optional: the transformer alone is ~21 GB in FP8, and the text encoder is
another 26 GB — they will never co-reside on 24 GB. `--offload cpu` means **32 GB of system RAM is a
hard floor**, and 64 GB is comfortable.

### Scaling it

| Goal | Command | Notes |
|---|---|---|
| Fast first test | `run --width 512 --height 896 --num-frames 97` | lowest VRAM, ~4 s clip |
| Default | `run` | 768x1280, ~5 s |
| Higher quality | `run --width 1024 --height 1536 --num-frames 97` | the pipeline's default size; tight on 24 GB |
| Long clips | `run --num-frames 193 --chunked` | 193 frames ≈ 8 s, windowed in 97-frame chunks |
| Better prompt adherence | `run --enhance-prompt` | rewrites the prompt before encoding |
| Prompt from a file | `run --prompt-file ./prompt.txt` | avoids shell quoting |
| Production quality | see [DFR](#production-quality-dfr) | more VRAM and time |

`--height`/`--width` must be **multiples of 64**; `--num-frames` must be on the **8k+1** grid
(97, 121, 193, …). The script validates both before loading 70 GB of weights.

### If you get a CUDA OOM

In order of least pain:

1. Drop resolution one step (1024x1536 → 768x1280 → 512x896).
2. Drop frames (121 → 97).
3. `OFFLOAD=disk` instead of `cpu` (streams transformer weights from disk; slower, needs a fast disk).
4. Use a 32 GB+ card if the resolution you want is non-negotiable.

---

## Production quality (DFR)

`DFRPipeline` uses the same distilled transformer plus a detailing IC-LoRA, and is the quality path.
It needs the IC-LoRA (~0.3 GB):

```bash
# download it alongside the main set
INCLUDE_DFR=1 ./scripts/run.sh download     # or: INCLUDE_DFR=1 ./scripts/download-models.sh
```

Then call the pipeline directly (the `run` wrapper is distilled-only):

```bash
docker compose run --rm ltx2 python -m ltx_pipelines.dfr_pipeline \
  --transformer-path       /models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path      /models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path         /models/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path         /models/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --detailing-lora         /models/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
  --spatial-upsampler-path /models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --height 768 --width 1280 --num-frames 121 \
  --output-path /output/dfr.mp4 \
  --prompt "..."
```

DFR also wants `--quantization fp8-cast --offload cpu` on 24 GB. Expect it to be several times slower
than the distilled path.

---

## Other pipelines

The image contains every upstream pipeline; pass the module name directly:

```bash
docker compose run --rm ltx2 python -m ltx_pipelines.distilled --help
docker compose run --rm ltx2 python -m ltx_pipelines.ic_lora --help
docker compose run --rm ltx2 shell            # interactive, venv already active
```

Available modules include `distilled`, `dfr_pipeline`, `ti2vid_two_stages`, `ti2vid_two_stages_hq`,
`ti2vid_one_stage`, `ic_lora`, `a2vid_two_stage`, `retake`, `dubit`, `t2a_one_stage`,
`hdr_ic_lora`, plus `*_mgpu` variants. Several of them need weights this repo does not fetch by
default — check the module's `--help` and its required `--*-path` arguments before running it.

---

## Audio-driven video (a2vid)

`A2VidPipelineTwoStage` generates video conditioned on an audio clip. It is wrapped as
`ltx2-a2vid` and needs **two extra weight files** the distilled profile does not use:

| File | Size | Why |
|---|---|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | ~42 GB | stage 1 denoises with the **full** model (CFG + STG) |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | ~8.9 GB | `--distilled-lora` is required; it drives the stage-2 refine |

```bash
INCLUDE_A2VID=1 ./scripts/run.sh download            # 5 distilled files + these 2 (~122 GiB)
mkdir -p audio && cp /path/to/clip.wav audio/        # mounted read-only at /audio
./scripts/run.sh a2vid --audio /audio/clip.wav
```

Pass `--image clip.jpg [FRAME [STRENGTH]]` to pin a start still at frame 0 — the analogue of a
Wan `start.jpg`; omit it for pure audio-to-video. The 48 GB class (L40S / A6000 / A100) is the
comfortable tier; on 24-32 GB drop to
`--height 768 --width 1280` or lower first. Because the frame count follows the audio
(`--audio-max-duration`) or `--num-frames`, a long clip at full resolution is the easiest way
to run out of memory — start with a ~5 s clip. `--height`/`--width` here are the **stage-2**
size; stage 1 runs at half. See [docs/A2VID-TEST.md](docs/A2VID-TEST.md) for a rented-GPU
runbook and the driver/disk requirements.

---

## Why this PC can't run it

Measured on this machine:

| Item | This PC | LTX-2.5 needs |
|---|---|---|
| GPU | **NVIDIA Quadro K5200** (Kepler, 2014) | Ampere or newer |
| Compute capability | **3.0** (sm_30) | ≥ 8.0 |
| VRAM | **8 GB** | 32 GB+ (24 GB with FP8 + offload) |
| Driver / CUDA | **462.59 / CUDA 11.8** (final Kepler driver) | ≥ 580 / CUDA 13.2 |
| CPU | Xeon E5-2680 v3, 12C/24T | — |
| RAM | ~32 GB | 32 GB minimum |
| WSL2 | **not installed** (Docker Desktop needs it) | — |

Each of the first three is independently disqualifying:

- **No CUDA 13.2.** Kepler was dropped from CUDA 12.0 onward. There is no driver that gives a K5200 a
  CUDA 12.7+ runtime, let alone 13.2, so no container can use this GPU for LTX-2.5.
- **No FP8.** `--quantization fp8-cast` needs FP8 support in the tensor cores; sm_30 has none.
- **8 GB VRAM** is below even the most aggressive quantized floor for a usable result. Community
  Q2/Q3 GGUF quants of the 22B model claim 10–16 GB, but upstream's own guidance is that Q2/Q3 output
  is not usable for final renders, and they still need Ampere+ and a modern CUDA stack.
- **No Docker/WSL2** on the box, and WSL2 is not installed — so Docker Desktop can't run either.

The image here is therefore built for **cloud GPUs**, which is also the cheapest route to try LTX-2.5
without buying a card.

---

## Project layout

```text
ltx25-docker/
├── Dockerfile                  # 2-stage build: uv-synced venv on CUDA 13.2
├── compose.yaml                # ltx2 service (GPU) + downloader profile
├── .env.example                # paths, token, generation defaults
├── .dockerignore
├── .gitattributes              # forces LF on scripts
├── scripts/
│   ├── run.sh                  # host wrapper: build/download/doctor/run/a2vid/shell
│   ├── download-models.sh      # gated HF weight download (~71 GiB, +51 GiB with a2vid)
│   ├── run-distilled.sh        # in-container DistilledPipeline wrapper (24 GB defaults)
│   ├── run-a2vid.sh            # in-container A2VidPipelineTwoStage wrapper (audio-driven)
│   ├── doctor.sh               # in-container preflight
│   ├── check_libs.py           # torch/natten/CUDA capability checks
│   └── entrypoint.sh           # doctor | shell | run | a2vid | <cmd>
├── docs/
│   ├── VAST.md                 # build -> push -> rent on Vast.ai
│   └── A2VID-TEST.md           # a2vid on a rented GPU: weights, driver, disk
├── audio/                      # a2vid conditioning clips (gitignored, host volume)
├── models/ltx-2.5/             # weights land here (gitignored, host volume)
└── output/                     # generated mp4s (gitignored, host volume)
```

Weights are mounted at `/models` and are laid out exactly as the HF repo:
`diffusion_models/…`, `text_encoders/…`, `vae/…`, `latent_upscale_models/…`, `loras/…`.

---

## Troubleshooting

**`could not select device driver "" with capabilities: [[gpu]]`** — the NVIDIA Container Toolkit isn't
wired into Docker. On a cloud instance: `nvidia-ctk runtime configure --runtime=docker && systemctl
restart docker`. Verify with `docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi`.

**`CUDA driver version is insufficient for CUDA runtime version`** — host driver is older than 580.
Update it; you cannot work around this from inside the container.

**`401`/`403` from Hugging Face** — accept the terms on the
[model page](https://huggingface.co/Lightricks/LTX-2.5) and use a **Read** token. Fine-grained tokens
need the *"read gated repos"* scope enabled.

**Build fails resolving torch/natten** — confirm the base image tag exists:
`docker pull nvidia/cuda:13.2.0-devel-ubuntu24.04`. If NVIDIA has moved the tag, override it:
`docker compose build --build-arg CUDA_VERSION=13.3.0`.

**`Cannot load symbol cublasLtGetVersion` / cuDNN `SUBLIBRARY_VERSION_MISMATCH`** — a torch/cuDNN
mismatch. Upstream pins this in `pyproject.toml` (`nvidia-cudnn-cu13==9.24.0.43`); the
`uv sync` in the Dockerfile honours that override. If it still appears, rebuild without the
Docker build cache: `docker compose build --no-cache`.

**Generation is slower than expected on 24 GB** — expected. FP8 upcast-on-the-fly plus CPU offload
trades speed for fit. Removing `--offload cpu` on a 24 GB card causes an OOM, not a speedup.

---

## Notes and caveats

- **Not yet built on real hardware.** This PC has no Docker and no compatible GPU, so the image has
  not been executed. Everything version-critical was verified against live sources: the
  `nvidia/cuda:13.2.0-*` tags exist, `torch-2.13.0` and `nvidia-cudnn-cu13==9.24.0.43` are in the cu132
  index, and `natten-0.21.7+torch2130cu132` is published for linux_x86_64. Note that upstream commits
  **no `uv.lock`**, so the Dockerfile resolves with plain `uv sync` (their docs' `--frozen` fails on a
  fresh clone). Expect the first CI build to be where any remaining drift surfaces.
- **Pinned to `main` by default.** Reproducible builds should pin `LTX_REF` in `.env` to a commit/tag
  once you have a build you trust.
- **Weights are gated.** Usage is governed by the [LTX-2.x Community License](https://huggingface.co/Lightricks/LTX-2.5):
  free for commercial use under $10M annual revenue, otherwise a commercial agreement is required. The
  `LTX-2` *code* is Apache-2.0; the models are not.
- **First run is slow.** Loading 70 GB of weights, FP8 conversion, and first-time kernel compilation
  dominate the first generation; later runs are much faster.

---

## References

- Model: <https://huggingface.co/Lightricks/LTX-2.5>
- Code: <https://github.com/Lightricks/LTX-2>
- Docs: <https://docs.ltx.io/open-source-model/getting-started/overview>
- System requirements: <https://docs.ltx.io/open-source-model/getting-started/system-requirements>
- Docker walkthrough: <https://ltx.io/blog/ltx-2-5-docker>
- CUDA driver compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/>
