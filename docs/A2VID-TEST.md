# Testing `a2vid_two_stage` on a rented GPU

`A2VidPipelineTwoStage` generates video conditioned on an audio clip. It is wired into this
repo as `ltx2-a2vid` (entrypoint: `a2vid`), and it needs **different weights** from the
distilled path.

## Why the extra weights

| File | Size | Why |
|---|---|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | ~42 GB | stage 1 denoises with the **full** model (CFG + STG, 40 steps by default) |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | ~8.9 GB | `--distilled-lora` is a **required** argument; it drives stage 2 |

The distilled transformer the repo already downloads is a different checkpoint and will not
substitute. Total weight set: **~122 GiB**, so budget **300 GB** of disk, not 120.

```bash
export HF_TOKEN=hf_xxx
INCLUDE_A2VID=1 ./scripts/run.sh download
```

## Machine spec

| | Minimum | Comfortable |
|---|---|---|
| GPU | 32 GB, Ampere+ | **48 GB** (L40S / A6000 / A100) |
| Driver | `cuda_max_good >= 13.2` (R595+) | 595.91+ |
| System RAM | 64 GB | 128 GB |
| Disk | 250 GB | 300 GB |
| Network | 500 Mb/s | 1 Gb/s+ (122 GB of weights) |

The driver is the filter that actually bites. On Vast:

```bash
vastai search offers 'num_gpus=1 rentable=true verified=true \
  gpu_name in [L40S,RTX_A6000,RTX_5090] cuda_max_good>=13.2 \
  gpu_ram>=30000 disk_space>=250 cpu_ram>=60000 inet_down>=400' -o dph_total
```

A host on a 580.x driver may report `cuda_max_good 13.0` and fail at the first CUDA call —
that is a host problem, not an image problem.

## Run

```bash
docker compose run --rm ltx2 a2vid --audio /audio/clip.wav

# or the host wrapper (sets MSYS_NO_PATHCONV on Windows/Git Bash)
mkdir -p audio && cp /path/to/clip.wav audio/
./scripts/run.sh a2vid --audio /audio/clip.wav
```

Defaults: 768x1280 final (stage 1 denoises at 384x640), 121 frames ≈ 5 s at 24 fps,
`fp8-cast`, `--offload cpu`. Useful flags:

| Flag | Effect |
|---|---|
| `--audio-max-duration S` | derive length from the audio instead of `--num-frames` (mutually exclusive) |
| `--audio-start-time S` | start reading the clip at an offset |
| `--height/--width` | **stage-2** size, multiples of 64; stage 1 runs at half |
| `--num-frames N` | 8k+1 grid (97, 121, 193…) |
| `-- --num-inference-steps 20` | cheaper first smoke test (anything after `--` goes to the pipeline) |

## Using the Wan2.2 S2V inputs

The Wan2.2-S2V pipeline took a **start image + audio.wav + prompt.txt** triple. All three map
onto a2vid; only the start image needs a flag:

| Wan2.2 S2V | a2vid |
|---|---|
| `start=@.../start.jpg` | `--image /audio/start.jpg` (frame 0, strength 1.0 → the still **replaces** frame 0) |
| `audio=@.../audio.wav` | `--audio /audio/audio.wav` |
| `prompt="$(cat prompt.txt)"` | `--prompt "$(cat prompt.txt)"` |

```bash
./scripts/run.sh a2vid \
  --audio  /audio/audio.wav \
  --image  /audio/start.png \
  --prompt "$(cat prompt.txt)" \
  --width 768 --height 512 \
  --audio-max-duration 5          # see the length note below
```

Measured on the S2V run this replaces (`scene01-sc01_001`): start.png **1280x768**,
audio.wav **10.0 s**, prompt **2481 chars**, output **768x512 @ 24 fps**, 10.0 s, 4 clips of
48 frames at 16 fps, **1167 s** wall on an **L40S 48 GB / driver 595.91.07**.

| | Wan2.2 S2V | a2vid |
|---|---|---|
| Length | derived from audio (10.0 s) | 121 frames @ 24 fps ≈ 5.04 s by default, or any `--audio-max-duration` |
| Frame rate | 16 fps internal, conformed to 24 fps | 24 fps default; `--frame-rate 16` matches Wan's internal cadence |
| Out size | 768x512 | `--width 768 --height 512`; default is portrait 768x1280 |
| Start frame | pinned from the still | pinned by `--image ... 0 <strength>` (centre-crops 1280x768 to the output aspect) |
| Long audio | split into 4 clips of 48 frames | **one chunk covers the whole clip by default** — 10 s is 241 frames in a single pass |
| Interface | HTTP API, multipart upload | CLI (`ltx2-a2vid`), files on disk |

**Two things to change from the Wan job:**

1. **Do not feed the full 10 s on the first run.** a2vid plans a single chunk spanning the whole
   clip, so 10 s at 24 fps means 241 frames denoised in one pass — that is what will OOM. Start
   with `--audio-max-duration 5`; for the full 10 s use chunking
   (`-- --chunk-pixel-frames 97`) once the short one works.
2. **Trim the prompt.** It narrates a 12-second beat-by-beat performance (crow at 5 s, hold to
   12 s). For a 5 s test, keep the first two paragraphs; the rest describes frames that will
   not exist.

**The one real behavioural difference:** S2V is a speech-to-video model tuned for lip sync on a
talking subject. a2vid is general audio-conditioned video (the upstream docs pitch it at
"audio-reactive video generation, music visualization"). Same inputs, but expect the beak and
beat-driven motion to be looser than Wan's — compare one clip before committing to a batch.

## First things to check

1. **Does it fit?** Watch `nvidia-smi` during stage 1. 48 GB should be fine at 768x1280;
   on 32 GB expect to drop resolution. A long clip at full resolution is the easiest way to
   OOM, because the frame count follows the audio.
2. **Does the audio drive the video?** The output keeps the *original* waveform, so a sync
   error is visual, not audible — compare mouth/beat movement against the audio.
3. **Timing.** 40 guided steps at half resolution plus a distilled stage 2; expect tens of
   minutes per ~5 s clip on one 48 GB card, plus 5-15 min of first-run weight load and
   Triton/JIT warmup. Budget 3-4 hours for install plus one or two generations.

## Cost sketch

48 GB at $0.43-0.55/hr → **$2-3** for a 4-hour session, plus instance storage. The image pull
and the 122 GB weight download are the long poles, not the compute — pick a host with a fast
`inet_down`.
