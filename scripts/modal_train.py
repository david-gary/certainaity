"""Modal training wrappers for all Certainaity forensic models.

Setup
-----
    pip install modal
    modal setup          # authenticate once

Full pipeline
-------------
    # 1. Create volumes
    modal volume create certainaity-raw
    modal volume create certainaity-data
    modal volume create certainaity-weights

    # 2. Upload raw datasets (download CASIA v2 + NIST16 locally first)
    modal volume put certainaity-raw ./data/raw/casia_v2 /casia_v2
    modal volume put certainaity-raw ./data/raw/nist16   /nist16

    # 3. Process raw → patches on Modal (~20 min, CPU, ~$0.10)
    modal run scripts/modal_train.py::build_dataset

    # 4. Train
    modal run scripts/modal_train.py --model patchforensic

Train individual models:
    modal run scripts/modal_train.py::train_patchforensic
    modal run scripts/modal_train.py::train_mantranet
    modal run scripts/modal_train.py::train_inpainting
    modal run scripts/modal_train.py::train_spsl

Override hyperparameters:
    modal run scripts/modal_train.py::train_patchforensic --epochs 50

Download weights after training:
    modal volume get certainaity-weights /weights ./weights

HuggingFace token (required for InpaintingDetector / GANDetector CLIP downloads):
    modal secret create huggingface-token HF_TOKEN=hf_...
"""

from __future__ import annotations

from pathlib import Path

import modal

# ─── App ──────────────────────────────────────────────────────────────────────

app = modal.App("certainaity-training")

# ─── Images ───────────────────────────────────────────────────────────────────

_repo_root = Path(__file__).parent.parent

# Lightweight image for CPU-only data preprocessing (no torch).
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "Pillow>=10.2",
        "pydantic>=2.6",
        "pydantic-settings>=2.2",
        "PyWavelets>=1.5",
        "scikit-image>=0.22",
        "piexif>=1.1.3",
        "structlog>=24.1",
        "scipy>=1.12",
        "certifi>=2024.2",
    )
    .add_local_dir(str(_repo_root / "src"), remote_path="/root/src")
)

# Base packages for GPU work (no local files yet — add_local_* must come last,
# so derived images add diffusers etc. BEFORE mounting local code).
_train_base = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2",
    "torchvision>=0.17",
    "transformers>=4.38",
    "albumentations>=1.4",
    "mlflow>=2.10",
    "faiss-cpu>=1.8",
    "scipy>=1.12",
    "numpy>=1.26",
    "Pillow>=10.2",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "PyWavelets>=1.5",
    "scikit-image>=0.22",
    "piexif>=1.1.3",
    "structlog>=24.1",
)


def _with_local_code(image: modal.Image) -> modal.Image:
    return (
        image
        .add_local_dir(str(_repo_root / "src"), remote_path="/root/src")
        .add_local_dir(str(_repo_root / "scripts"), remote_path="/root/scripts")
    )


# Full image for GPU training.
training_image = _with_local_code(_train_base)

# ─── Volumes ──────────────────────────────────────────────────────────────────

raw_volume = modal.Volume.from_name("certainaity-raw", create_if_missing=True)
data_volume = modal.Volume.from_name("certainaity-data", create_if_missing=True)
weights_volume = modal.Volume.from_name("certainaity-weights", create_if_missing=True)

# ─── Shared training kwargs ───────────────────────────────────────────────────

_train_kwargs: dict = dict(
    image=training_image,
    volumes={"/data": data_volume, "/weights": weights_volume},
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(script: str, *args: str) -> None:
    """Run a training script from /root/scripts/ with src on PYTHONPATH."""
    import os
    import subprocess
    import sys
    env = {
        **os.environ,
        "PYTHONPATH": "/root/src:" + os.environ.get("PYTHONPATH", ""),
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    result = subprocess.run(
        [sys.executable, f"/root/scripts/{script}.py", *args],
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script} exited with code {result.returncode}")


def _run_module(module: str, *args: str) -> None:
    """Run a Python module with src on PYTHONPATH."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONPATH": "/root/src:" + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")

# ─── Data preparation ─────────────────────────────────────────────────────────

@app.function(
    image=prep_image,
    volumes={"/raw": raw_volume, "/data": data_volume},
    timeout=3_600,
    cpu=4,
)
def build_dataset(
    datasets: str = "casia_v2,nist16",
    split: str = "0.80/0.10/0.10",
    seed: int = 42,
    output: str = "/data/processed",
) -> None:
    """Process raw datasets into training patches (~20 min, CPU only, ~$0.10).

    Reads from the certainaity-raw volume (/raw) and writes patches +
    JSONL manifests to ``output`` on certainaity-data.

    Training set:
        modal run scripts/modal_train.py::build_dataset
    Held-out cross-dataset test set (separate dir so it never clobbers train):
        modal run scripts/modal_train.py::build_dataset \\
            --datasets imd2020 --split 0.0/0.0/1.0 --output /data/processed_imd2020
    """
    _run_module(
        "certainaity.data.build_dataset",
        "--datasets", *datasets.split(","),
        "--raw_dir", "/raw",
        "--output", output,
        "--split", split,
        "--seed", str(seed),
    )
    data_volume.commit()

# ─── Cross-dataset test data (IMD2020) ────────────────────────────────────────

@app.function(
    image=prep_image,
    volumes={"/raw": raw_volume},
    timeout=7_200,
    cpu=4,
)
def prepare_imd2020() -> None:
    """Download the IMD2020 'real-life' set and flatten it for build_dataset.

    Downloads the 4 real-life zip parts (~7.3 GB) directly onto the raw volume,
    extracts them, and flattens the nested per-image folders into
    ``/raw/imd2020/{tampered,masks}`` with matching stems.

    TLS: the host (staff.utia.cas.cz) is misconfigured — it does not send the
    intermediate certificate, so strict verifiers can't build the chain. We do
    NOT disable verification. Instead we fetch the published intermediate from
    the cert's AIA URL and add it to a full-verification (CERT_REQUIRED) context;
    the intermediate is cryptographically validated against the already-trusted
    HARICA/Hellenic Academic root during the handshake, so an HTTP-served forgery
    would simply fail to chain. This is exactly what AIA-aware browsers do.

    Mask pairing is structure-agnostic: any image whose name contains 'mask' is
    treated as a mask and paired with the sibling manipulated image of the same
    base name. Logs a sample of the extracted tree + the pairing count so the
    result can be sanity-checked before running build_dataset.
    """
    import shutil
    import ssl
    import urllib.request
    import zipfile
    from pathlib import Path

    import certifi

    base = "https://staff.utia.cas.cz/novozada/db/"
    # IMD2020.zip is the 'real-life' MANIPULATION set: per-image folders, each
    # with <id>_orig.jpg (authentic), <hash>_0.{jpg,png} (manipulated), and
    # <hash>_0_mask.png (mask). (IMD2020_real_*.zip is the 35K authentic
    # camera-organized images — no masks — and is NOT what we want here.)
    parts = ["IMD2020.zip"]

    # Complete the cert chain the server fails to send, WITHOUT weakening
    # verification: fetch the intermediate from the server cert's AIA CA-Issuers
    # URL and add it to an otherwise-default (CERT_REQUIRED, hostname-checked)
    # context based on the trusted certifi root bundle. Trust still derives from
    # the signature chain to the trusted root, so an HTTP-served forgery fails.
    intermediate_url = "http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer"
    print(f"[tls] fetching intermediate from {intermediate_url}")
    with urllib.request.urlopen(intermediate_url, timeout=120) as r:
        inter_der = r.read()
    inter_pem = ssl.DER_cert_to_PEM_cert(inter_der)
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(cadata=inter_pem)
    print(f"[tls] intermediate added; verify_mode={ctx.verify_mode} "
          f"check_hostname={ctx.check_hostname} (full verification)")

    dl = Path("/raw/imd2020_dl")
    ex = Path("/raw/imd2020_extract")
    dl.mkdir(parents=True, exist_ok=True)
    ex.mkdir(parents=True, exist_ok=True)

    # Reclaim space from an earlier wrong-set download (the 7.3 GB authentic
    # camera images) and clear any stale extract before re-extracting.
    for stale in dl.glob("IMD2020_real_*.zip"):
        print(f"[cleanup] removing stale {stale.name}")
        stale.unlink()
    if ex.exists():
        shutil.rmtree(ex)
    ex.mkdir(parents=True, exist_ok=True)

    for fn in parts:
        dest = dl / fn
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {fn} already downloaded ({dest.stat().st_size} B)")
        else:
            print(f"[download] {fn} …")
            req = urllib.request.Request(base + fn, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, context=ctx, timeout=600) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            print(f"[download] {fn} done ({dest.stat().st_size} B)")
        print(f"[extract] {fn} …")
        with zipfile.ZipFile(dest) as z:
            z.extractall(ex)

    # Sample of the extracted structure for verification.
    print("=== sample of extracted tree ===")
    for i, p in enumerate(sorted(ex.rglob("*"))):
        if p.is_file():
            print("  ", p.relative_to(ex))
        if i >= 30:
            break

    # Flatten with EXACT suffix pairing: a mask "<base>_mask.png" pairs with the
    # manipulated image "<base>.{jpg,jpeg,png,tif}" in the same folder. This is
    # unambiguous for the IMD2020 layout (no prefix-collision risk).
    tdir = Path("/raw/imd2020/tampered")
    mdir = Path("/raw/imd2020/masks")
    for d in (tdir, mdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    img_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    paired = 0
    unpaired = 0
    for mask in ex.rglob("*_mask.png"):
        if not mask.is_file():
            continue
        base = mask.name[: -len("_mask.png")]  # e.g. "c8yjv4u_0"
        cand = None
        for ext in img_exts:
            c = mask.parent / f"{base}{ext}"
            if c.exists():
                cand = c
                break
        if cand is None:
            unpaired += 1
            continue
        uid = f"{paired:06d}_{cand.stem}"
        shutil.copy(cand, tdir / f"{uid}{cand.suffix.lower()}")
        shutil.copy(mask, mdir / f"{uid}_gt.png")
        paired += 1
    print(f"=== flattened: {paired} manipulated/mask pairs "
          f"({unpaired} masks had no sibling image) → /raw/imd2020 ===")
    raw_volume.commit()

# ─── Training functions ───────────────────────────────────────────────────────

@app.function(gpu="A10G", timeout=10_800, **_train_kwargs)
def train_patchforensic(
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-3,
    curriculum_warmup: int = 30,
    patience: int = 20,
) -> None:
    """Train the PatchForensic FCN with AMP + augmentation (~2 hrs, ~$2.20 on A10G)."""
    _run(
        "train_patchforensic",
        "--data-dir", "/data/processed",
        "--weights-dir", "/weights",
        "--mlflow-uri", "/weights/mlruns",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--curriculum-warmup", str(curriculum_warmup),
        "--patience", str(patience),
        "--preload",
        "--augment",
        "--no-amp",
    )
    weights_volume.commit()


@app.function(gpu="A10G", timeout=10_800, **_train_kwargs)
def train_mantranet(
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-4,
    unfreeze_epoch: int = 10,
    patience: int = 15,
) -> None:
    """Fine-tune MantraNet VGG-16/BN with AMP + augmentation (~1.5 hrs, ~$1.65 on A10G)."""
    _run(
        "train_mantranet",
        "--data-dir", "/data/processed",
        "--weights-dir", "/weights",
        "--mlflow-uri", "/weights/mlruns",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--unfreeze-epoch", str(unfreeze_epoch),
        "--patience", str(patience),
        "--preload",
        "--augment",
    )
    weights_volume.commit()


@app.function(
    gpu="A10G",
    timeout=14_400,
    secrets=[modal.Secret.from_name("huggingface-token")],
    **_train_kwargs,
)
def train_inpainting(
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 5e-5,
    clip_lr: float = 5e-6,
    freeze_epochs: int = 5,
    unfreeze_layers: int = 4,
    patience: int = 12,
) -> None:
    """Fine-tune InpaintingDetector CLIP ViT-B/32 with AMP + augmentation (~2 hrs, ~$2.20 on A10G).

    Requires the ``huggingface-token`` Modal secret to download CLIP weights.
    """
    _run(
        "train_inpainting",
        "--data-dir", "/data/processed",
        "--weights-dir", "/weights",
        "--mlflow-uri", "/weights/mlruns",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--clip-lr", str(clip_lr),
        "--freeze-epochs", str(freeze_epochs),
        "--unfreeze-layers", str(unfreeze_layers),
        "--patience", str(patience),
        "--preload",
        "--augment",
    )
    weights_volume.commit()


@app.function(gpu="A10G", timeout=10_800, **_train_kwargs)
def train_spsl(
    epochs: int = 90,
    batch_size: int = 64,
    lr: float = 1e-4,
    margin: float = 1.0,
    embedding_dim: int = 256,
    patience: int = 12,
) -> None:
    """Train SPSL Siamese ResNet-50 + FAISS with AMP + augmentation (~1.5 hrs, ~$1.65 on A10G)."""
    _run(
        "train_spsl",
        "--data-dir", "/data/processed",
        "--weights-dir", "/weights",
        "--mlflow-uri", "/weights/mlruns",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--margin", str(margin),
        "--embedding-dim", str(embedding_dim),
        "--patience", str(patience),
        "--preload",
        "--augment",
    )
    weights_volume.commit()

# ─── Phase-1: AI-inpainting generation + union training ───────────────────────

# Diffusers stack for Stable Diffusion 2 inpainting generation — add the extra
# packages to the base BEFORE mounting local code (add_local_* must come last).
gen_image = _with_local_code(
    _train_base.pip_install("diffusers>=0.27", "accelerate>=0.27", "safetensors>=0.4")
)


@app.function(
    image=gen_image,
    gpu="A10G",
    timeout=21_600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/raw": raw_volume},
)
def generate_inpainting_data(n_images: int = 4000, steps: int = 30, seed: int = 0) -> None:
    """Generate a synthetic AI-inpainting set (SD2) into /raw/ai_inpainting.

    Downloads COCO val2017 (public, ~778 MB) as pristine sources on first run,
    then runs generate_inpainting.py. Smoke-test with a tiny n first:
        modal run scripts/modal_train.py::generate_inpainting_data --n-images 20
    Full run (~1.5–2 hr, ~$2 on A10G at 30 steps):
        modal run scripts/modal_train.py::generate_inpainting_data
    """
    import shutil
    import urllib.request
    import zipfile
    from pathlib import Path

    coco = Path("/raw/val2017")
    if not coco.exists() or not any(coco.iterdir()):
        zp = Path("/raw/val2017.zip")
        if not (zp.exists() and zp.stat().st_size > 0):
            url = "http://images.cocodataset.org/zips/val2017.zip"
            print(f"[download] COCO val2017 from {url} …")
            with urllib.request.urlopen(url, timeout=900) as r, open(zp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            print(f"[download] done ({zp.stat().st_size} B)")
        print("[extract] COCO val2017 …")
        with zipfile.ZipFile(zp) as z:
            z.extractall("/raw")  # creates /raw/val2017/
    n_src = sum(1 for _ in coco.glob("*.jpg"))
    print(f"[sources] {n_src} COCO images at {coco}")

    _run_module(
        "certainaity.data.generate_inpainting",
        "--source_dirs", "/raw/val2017",
        "--output_dir", "/raw/ai_inpainting",
        "--n_images", str(n_images),
        "--device", "cuda:0",
        "--steps", str(steps),
        "--seed", str(seed),
    )
    raw_volume.commit()


@app.function(gpu="A10G", timeout=28_800, memory=40_960, **_train_kwargs)
def train_patchforensic_union(
    epochs: int = 90,
    batch_size: int = 32,
    lr: float = 1e-3,
    curriculum_warmup: int = 30,
    patience: int = 20,
    extra_dirs: str = "/data/processed_ai_inpainting",
    ckpt_name: str = "patchforensic_union_v1.pth",
) -> None:
    """E1: PatchForensic on the CASIA + AI-inpainting UNION (fp32, geometric aug).

    Higher RAM (40 GB) so the larger union still fits the preload cache, and an
    8-hour timeout because the ~42k-sample union runs ~4-5 min/epoch fp32 (the
    14400s default timed out at epoch 57). Epoch budget trimmed to 90 (early
    stopping, patience 20, usually cuts it sooner). Saves to a distinct
    checkpoint so the CASIA-only baseline (patchforensic_v2.pth) is preserved.
    """
    _run(
        "train_patchforensic",
        "--data-dir", "/data/processed",
        "--extra-data-dirs", *extra_dirs.split(","),
        "--weights-dir", "/weights",
        "--mlflow-uri", "/weights/mlruns",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--curriculum-warmup", str(curriculum_warmup),
        "--patience", str(patience),
        "--ckpt-name", ckpt_name,
        "--preload",
        "--augment",
        "--no-amp",
    )
    weights_volume.commit()

# ─── Remote orchestrator ──────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=21_600,  # 6 hours — all four trainings run sequentially in one container
    secrets=[modal.Secret.from_name("huggingface-token")],
    **_train_kwargs,
)
def train_all(models: str = "patchforensic,mantranet,inpainting,spsl") -> None:
    """Run the selected trainings sequentially inside one Modal container.

    Because the entire sequence executes on Modal (not via a local entrypoint
    driving repeated ``.remote()`` calls), it is immune to local client
    disconnects.  Launch detached so it survives the terminal closing::

        modal run --detach scripts/modal_train.py::train_all
        modal run --detach scripts/modal_train.py::train_all --models patchforensic,spsl

    Each model's weights are committed to the volume as it finishes, so a
    failure in a later stage never discards an earlier model's checkpoint.
    """
    import logging
    log = logging.getLogger("train_all")
    fns = {
        "patchforensic": train_patchforensic,
        "mantranet": train_mantranet,
        "inpainting": train_inpainting,
        "spsl": train_spsl,
    }
    for name in [m.strip() for m in models.split(",") if m.strip()]:
        log.info("=== Starting %s ===", name)
        # .local() runs the function body in THIS container (using this
        # orchestrator's GPU, volumes, and HF secret); the per-function Modal
        # config is ignored, which is exactly what we want here.
        fns[name].local()
        log.info("=== Finished %s ===", name)

# ─── Evaluation ───────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=3_600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    **_train_kwargs,
)
def evaluate(
    manifest: str = "/data/processed/val.jsonl",
    models: str = "patchforensic,mantranet,inpainting,spsl",
    limit: int = 0,
    output: str = "/weights/reports/eval.json",
) -> None:
    """Run the rigorous eval harness on a manifest and persist the report.

    In-distribution (CASIA val):
        modal run scripts/modal_train.py::evaluate

    Cross-dataset generalization (once NIST16 is processed onto the volume):
        modal run scripts/modal_train.py::evaluate \\
            --manifest /data/processed_nist16/test.jsonl
    """
    _run(
        "evaluate",
        "--weights-dir", "/weights",
        "--data-dir", str(Path(manifest).parent),
        "--manifest", manifest,
        "--models", models,
        "--device", "cuda",
        "--limit", str(limit),
        "--output", output,
    )
    weights_volume.commit()

@app.function(image=training_image, volumes={"/weights": weights_volume}, timeout=600)
def swap_ckpt(src: str, dst: str) -> None:
    """Copy /weights/<src> → /weights/<dst> and commit.

    Used to point the eval harness (which loads the hardcoded WEIGHT_FILE
    ``patchforensic_v2.pth``) at an experiment checkpoint without losing the
    baseline:
        swap_ckpt patchforensic_v2.pth      patchforensic_v2.baseline.pth  # back up
        swap_ckpt patchforensic_union_v1.pth patchforensic_v2.pth          # use union
        ...evaluate...
        swap_ckpt patchforensic_v2.baseline.pth patchforensic_v2.pth       # restore
    """
    import shutil
    from pathlib import Path
    s, d = Path("/weights") / src, Path("/weights") / dst
    if not s.exists():
        raise FileNotFoundError(f"{s} not found")
    shutil.copy(s, d)
    print(f"copied {src} -> {dst}")
    weights_volume.commit()

# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "all") -> None:
    """
    Launch training for one or all models.

        modal run scripts/modal_train.py                  # train all (not detached)
        modal run scripts/modal_train.py --model spsl     # train one

    For unattended overnight runs, prefer the detached remote orchestrator:

        modal run --detach scripts/modal_train.py::train_all
    """
    fns = {
        "patchforensic": train_patchforensic,
        "mantranet": train_mantranet,
        "inpainting": train_inpainting,
        "spsl": train_spsl,
    }
    to_run = list(fns.items()) if model == "all" else [(model, fns[model])]
    for name, fn in to_run:
        print(f"Launching {name}...")
        fn.remote()
