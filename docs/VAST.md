# Running the image on Vast.ai

Vast.ai runs **your Docker image as the instance's container** — it does not run
`docker build` inside instances. Their technical FAQ is explicit: *"Docker-in-Docker
is disabled for security."* So the workflow is **build elsewhere → push to a
registry → point a Vast template at it**.

```text
GitHub Actions (or any Docker host)     ghcr.io/<you>/ltx25:cu132     Vast.ai instance
        build + push            ──────────────────────────────▶      pull + run
```

## 1. Build and push

Use the included workflow: `.github/workflows/build-image.yml`.

```bash
git init && git add . && git commit -m "LTX-2.5 docker"
git remote add origin git@github.com:<you>/ltx25-docker.git
git push -u origin main
```

Then **Actions → Build LTX-2.5 image → Run workflow**. It builds on a runner with
Docker and real network access (this PC's shell has neither) and pushes
`ghcr.io/<you>/ltx25:cu132`.

Make the package **public** afterwards (GitHub → your profile → Packages →
`ltx25` → Package settings → Change visibility), or keep it private and give Vast
registry credentials through the template's docker-login fields.

## 2. Check the host driver first

This image is CUDA 13.2. A host must run **driver ≥ 580** (R595 is the matching
branch). Vast's own GPU guides still document **CUDA 12.8** as their standard, so
before renting, verify the specific machine with the search filters and prefer an
image/driver combination known to be current. If you land on a host stuck on
CUDA 12.x, the build fails at `torch.cuda.is_available()` with a driver-too-old
error — that is a host problem, not an image problem.

## 3. Rent and configure

Search Vast for a **≥ 32 GB VRAM** card (or 24 GB with the FP8+offload profile),
with **disk ≥ 100 GB** — the weights alone are ~71 GiB and are downloaded inside
the instance, not baked into the image.

Use the API/CLI to create an instance from the image:

```bash
vastai search offers 'gpu_name in [RTX 5090,RTX 4090] num_gpus=1 gpu_ram>=31000 disk_space>=120 rentable=true verified=true' -o dph_total

vastai create instance <OFFER_ID> ghcr.io/<you>/ltx25:cu132 \
  --disk 120 --ssh --direct \
  --label ltx25 \
  --env "-e HF_TOKEN=hf_xxx"
```

Because Vast replaces the image's entrypoint (for SSH/Jupyter runtypes) or passes
`args_str` to it (`runtype: args`), interact with the tooling explicitly:

```bash
# on the instance (SSH)
ltx2-doctor                        # preflight: driver, VRAM, RAM, disk
ltx2-run --help
```

## 4. Fetch the weights on the instance

Weights are gated on Hugging Face; the instance needs a token:

```bash
export HF_TOKEN=hf_xxx
bash /opt/ltx2-docker/scripts/download-models.sh    # if you copied the scripts in
# or, simplest, inside the image:
pip install -U "huggingface_hub[hf]"
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir /models
```

Then:

```bash
ltx2-run --height 768 --width 1280 --num-frames 121 --prompt "your prompt" \
         --out /output/clip.mp4
```

## Why not "build on Vast and download the image"?

Two independent reasons:

1. **Vast containers can't run Docker**, so there is no builder to use. (Vast *does*
   offer VM instances with nested containerization, but that costs VM overhead and
   still leaves reason 2.)
2. **Downloading an image is the wrong direction.** A ~15 GB image compressed
   tarball is a slow, one-off transfer; a registry is how Vast is designed to
   consume images, and it stays cached for the next instance. If you truly need a
   tarball (e.g. airgapped): `docker save ltx2.5:cu132 | zstd -19 > ltx25.tar.zst`
   then `scp` it, and `zstd -d < ltx25.tar.zst | docker load` on the target.
