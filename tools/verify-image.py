#!/usr/bin/env python3
"""Verify a GHCR image the way a rented GPU host will consume it.

    python tools/verify-image.py [--image ghcr.io/josephatwoosh/ltx25:cu132]
                                 [--expect-history 'run-a2vid.sh']
                                 [--expect-entrypoint /usr/local/bin/ltx2-entrypoint]

Checks, anonymously (no docker login):

  1. the tag resolves to a *single-arch manifest* -- an OCI index makes the pull a
     two-step by digest, which is what broke :cu132 once before
  2. the config blob is fetchable and parses
  3. every layer blob is fetchable (HEAD, 200) -- this is what `docker pull`
     actually does; a MANIFEST_UNKNOWN on one layer fails the pull
  4. optional: the build history mentions a path, i.e. the new files are inside
  5. optional: the entrypoint matches

Exit code 0 = pullable, 1 = a check failed. Prints the digest so it can be pinned
in a Vast template.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)
INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}


def parse_image(ref: str) -> tuple[str, str, str]:
    """ghcr.io/owner/name:tag -> (registry, repository, tag)."""
    registry, _, rest = ref.partition("/")
    repository, _, tag = rest.rpartition(":")
    if not registry or not repository or not tag:
        raise SystemExit(f"cannot parse image reference: {ref}")
    return registry, repository, tag


def request(url: str, *, token: str | None = None, method: str = "GET", accept: str | None = None):
    headers = {"User-Agent": "ltx25-verify"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ghcr.io/josephatwoosh/ltx25:cu132")
    ap.add_argument("--expect-history", default=None, help="substring that must appear in build history")
    ap.add_argument("--expect-entrypoint", default=None)
    args = ap.parse_args()

    registry, repository, tag = parse_image(args.image)
    base = f"https://{registry}/v2/{repository}"
    failures: list[str] = []

    def ok(msg):
        print(f"  ok    {msg}")

    def bad(msg):
        print(f"  FAIL  {msg}")
        failures.append(msg)

    # --- anonymous pull token (exactly what a Vast host gets)
    try:
        tok = json.load(request(f"https://{registry}/token?scope=repository:{repository}:pull&service={registry}"))["token"]
        ok("anonymous pull token issued")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  cannot get pull token: {exc}")
        return 1

    # --- manifest
    try:
        resp = request(f"{base}/manifests/{tag}", token=tok, accept=ACCEPT)
        digest = resp.headers.get("Docker-Content-Digest", "")
        manifest = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"  FAIL  manifest {tag}: HTTP {exc.code}")
        return 1

    media_type = manifest.get("mediaType", "")
    print(f"  tag     {args.image}")
    print(f"  digest  {digest}")
    print(f"  type    {media_type or 'unknown'}")

    if media_type in INDEX_TYPES or ("manifests" in manifest and "layers" not in manifest):
        bad("tag is an OCI index, not a single-arch manifest (multi-step pull; Vast can fail on it)")
        # still try to follow the amd64 child so the rest of the check is meaningful
        child = next(
            (m for m in manifest.get("manifests", []) if m.get("platform", {}).get("architecture") == "amd64"),
            None,
        )
        if child:
            resp = request(f"{base}/manifests/{child['digest']}", token=tok, accept=ACCEPT)
            manifest = json.load(resp)
            ok(f"followed amd64 child {child['digest'][:19]}")
    else:
        ok("single-arch manifest (one-step pull)")

    # --- config
    cfg_digest = manifest.get("config", {}).get("digest")
    try:
        cfg = json.load(request(f"{base}/blobs/{cfg_digest}", token=tok))
        ok(f"config blob {cfg_digest[:19]} parsed ({len(json.dumps(cfg))} bytes)")
    except Exception as exc:  # noqa: BLE001
        bad(f"config blob {cfg_digest}: {exc}")
        cfg = {}

    # --- layers (what docker pull actually fetches)
    layers = manifest.get("layers", [])
    total = sum(layer.get("size", 0) for layer in layers)
    missing = 0
    for layer in layers:
        try:
            r = request(f"{base}/blobs/{layer['digest']}", token=tok, method="HEAD")
            if r.status != 200:
                missing += 1
        except urllib.error.HTTPError as exc:
            missing += 1
            bad(f"layer {layer['digest'][:19]}: HTTP {exc.code}")
    if missing == 0 and layers:
        ok(f"all {len(layers)} layer blobs fetchable ({total / 1e9:.1f} GB compressed)")
    elif not layers:
        bad("manifest has no layers")

    # --- contents / metadata expectations
    config = cfg.get("config", {})
    entry = config.get("Entrypoint") or []
    if args.expect_entrypoint:
        if entry == [args.expect_entrypoint]:
            ok(f"entrypoint {entry}")
        else:
            bad(f"entrypoint is {entry}, expected [{args.expect_entrypoint}]")

    if args.expect_history:
        history = " ".join(h.get("created_by", "") for h in cfg.get("history", []))
        if args.expect_history in history:
            ok(f"build history mentions {args.expect_history!r}")
        else:
            bad(f"build history does not mention {args.expect_history!r} -- stale image?")

    print(f"  created {cfg.get('created')}")
    print()
    if failures:
        print(f"NOT PULLABLE / NOT READY: {len(failures)} problem(s)")
        return 1
    print("PULLABLE: tag resolves to a single manifest with every layer fetchable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
