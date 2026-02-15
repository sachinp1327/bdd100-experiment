# BDD100K Object Detection: Analysis, Training, Evaluation

This repository is organized into **three divisions**:
1. **Division 1: Data Analysis** (`src/bdd100k_analysis`)
2. **Division 2: Training + Evaluation** (`src/bdd100k_detection`)
3. **Division 3: Post‑Training Diagnostics** (`src/bdd100k_evaluation`)

Each division has its own CLI and (where applicable) Streamlit dashboard.

---

**Docker Build & Run (Do This First)**

This project is designed to run inside a CUDA‑enabled Docker container.

**My setup (for reference)**
- GPU: **NVIDIA RTX 2000 Ada**, **8 GB VRAM**
- Docker base image: **CUDA 11.8 + cuDNN 8** (`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`)
- PyTorch / Torchvision (from `requirements.txt`):
  - `torch==2.0.0+cu118`
  - `torchvision==0.15.0+cu118`
- NVIDIA driver is provided by the **host** (not by the Dockerfile). It must be
  compatible with CUDA 11.8 inside the container.

**1) Build the Docker image**
```bash
./docker-build.sh bdd100k-od:latest
```
or
```bash
docker build -t bdd100k-od:latest .
```

**2) Run the container (with GPU)**
```bash
./docker-setup.sh bdd100k-od:latest bdd100k-od-dev
```

The script:
- mounts the repo into `/workspace`
- exposes Streamlit on port `8501`
- enables GPU via `--gpus all`

Once inside the container, run the commands in the next sections.

---

**Division 1: Data Analysis**

This division analyzes the BDD100K detection labels, computes dataset statistics, and produces an analysis dashboard with visuals.

**Expected dataset paths**
- Labels: `data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json`
- Images: `data/bdd100k_images_100k/bdd100k/images/100k/{train,val}`

**1) Run analysis (train + val + comparison)**
```bash
python3 -m bdd100k_analysis.cli analyze-all \
  --output outputs \
  --render-visuals \
  --visuals-limit 5
```

**2) Launch the analysis dashboard**
```bash
streamlit run src/bdd100k_analysis/dashboard_app.py -- --outputs outputs
```

**Outputs produced**
- `outputs/train/summary.json`
- `outputs/val/summary.json`
- `outputs/compare.json`
- `outputs/{train,val}/visuals/` (Top‑K samples by size/aspect)

**Notes**
- If you change analysis logic, re‑run `analyze-all` to refresh the output JSON.
- Visual groups include `small`, `large`, `widest`, `tallest`.

---

**Division 2: Training + Evaluation**

This division trains Faster R‑CNN on BDD100K and evaluates checkpoints. Training and evaluation are separate commands.

**Expected dataset paths**
- Labels: `data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json`
- Images: `data/bdd100k_images_100k/bdd100k/images/100k/{train,val}`
- COCO outputs (auto‑generated): `data/bdd100k_coco/instances_{train,val}.json`

**Training (one command)**
```bash
python3 -m bdd100k_detection.cli train \
  --backbone resnet50 \
  --epochs 12 \
  --batch-size 1 \
  --amp \
  --device cuda \
  --output-dir outputs/training
```

**Common training flags**
- `--backbone {resnet50,resnet101}`: choose backbone.
- `--epochs N`: total epochs.
- `--batch-size N`: adjust based on VRAM.
- `--amp`: enable mixed precision.
- `--trainable-backbone-layers N`: freeze more layers for faster training.
- `--step-size` and `--gamma`: LR schedule (StepLR).
- `--pin-memory` / `--no-pin-memory`: DataLoader optimization.
- `--persistent-workers` / `--no-persistent-workers`: DataLoader optimization.
- `--resume /path/to/checkpoint.pth`: resume training with optimizer + scheduler state.

**Evaluation (separate command)**
```bash
python3 -m bdd100k_detection.cli eval \
  --checkpoint outputs/training/checkpoint_epoch_12.pth \
  --backbone auto \
  --device cuda \
  --output-dir outputs/eval
```

**Notes about evaluation**
- `--backbone auto` will infer ResNet‑50 vs ResNet‑101 from the checkpoint.
- COCO‑style checkpoints trained on different class counts (e.g., COCO‑81) will
  load with `strict=False` and skip incompatible head weights.
- Predictions are saved as COCO‑style JSON in:
  `outputs/eval/predictions_0.json` (or epoch‑specific if evaluating during training).

**Training‑time evaluation logs**
- Each epoch’s metrics are appended to:
  `outputs/training/eval_metrics.jsonl`

---

**Division 3: Post‑Training Diagnostics**

This division analyzes predictions against ground truth to find:
- Which classes the model performs best/worst on
- Where it fails by metadata (timeofday, weather, scene)
- Best/worst qualitative samples with overlays
- Links between data analysis stats and model performance

**Input required**
- COCO‑style predictions JSON (from Division 2 evaluation).

**1) Run diagnostics**
```bash
python3 -m bdd100k_evaluation.cli \
  --predictions outputs/eval/predictions_0.json \
  --output-dir outputs/eval_diagnostics
```

**2) Launch diagnostics dashboard**
```bash
streamlit run src/bdd100k_evaluation/dashboard_app.py -- \
  --output-dir outputs/eval_diagnostics
```

**Outputs produced**
- `outputs/eval_diagnostics/overall_metrics.json`
- `outputs/eval_diagnostics/per_class_metrics.json`
- `outputs/eval_diagnostics/attribute_metrics.json`
- `outputs/eval_diagnostics/samples.json`
- `outputs/eval_diagnostics/renders/` (GT + Pred overlays)
- `outputs/eval_diagnostics/analysis_links.json`

**Notes**
- If you want more best/worst samples, re‑run diagnostics with `--top-k N`.
- The dashboard lets you choose how many of the stored samples to display.
