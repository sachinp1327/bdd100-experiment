# Division 2 Findings: Model Choice, Training, and Evaluation

This document captures the key decisions and methodology for **Division 2** (training and evaluation of the detection model).

---

**Why Faster R‑CNN for BDD100K**

We selected **Faster R‑CNN** as the primary detector for the following reasons:
- **Accuracy‑first tradeoff:** Faster R‑CNN (two‑stage) is typically stronger on accuracy than single‑stage detectors, especially for small objects and crowded scenes.
- **Dataset characteristics:** The analysis showed that **small objects dominate**, **occlusion is common**, and **clutter is high**. Two‑stage detectors are generally more robust under these conditions.
- **Standard benchmark choice:** Faster R‑CNN is a widely accepted baseline on COCO and BDD100K, making results easier to compare with known references.

---

**Constraint: Pretrained BDD100K Weights Unavailable**

At the time of this work, **public pretrained BDD100K weights were not available** (links were removed or inaccessible).  
Therefore, I had to **train my own model**, initializing from **COCO‑pretrained weights** where possible.

---

**Compute & Time Constraints**

Hardware constraint:
- **GPU: RTX 2000 Ada, 8 GB VRAM**

Given the available time budget, training was limited to:
- **12 epochs**  
This was chosen as a practical compromise between quality and total training time.

---

**Model Architecture**

The training pipeline uses **Faster R‑CNN with FPN**, configurable by backbone:
- **Default backbone:** `ResNet‑50 + FPN`
- Optional backbone: `ResNet‑101 + FPN`

Key components:
- **Backbone + FPN** for multi‑scale feature extraction.
- **RPN** for region proposals.
- **ROI heads** for classification and box regression.

Implementation notes:
- Torchvision model API is used (`fasterrcnn_resnet50_fpn` or `resnet_fpn_backbone`).
- Final predictor head is replaced to match **BDD100K classes (10 + background)**.

---

**Training Methodology**

**Data pipeline**
- BDD100K label JSON is converted to COCO format.
- Images are loaded from `data/bdd100k_images_100k/...`.

**Transforms**
- `ToTensor`
- `RandomHorizontalFlip(p=0.5)` for training only.

**Optimization**
- **Optimizer:** SGD
  - Learning rate: **0.005**
  - Momentum: **0.9**
  - Weight decay: **0.0005**
- **LR Scheduler:** StepLR
  - Step size: **8 epochs**
  - Gamma: **0.1**
- **AMP (mixed precision):** optional, enabled when needed for speed/VRAM.

**Runtime notes**
- Batch size is constrained by 8 GB VRAM (typically **1–2**).
- DataLoader optimizations: `pin_memory`, `persistent_workers` enabled.

---

**Loss Functions**

The model is trained with the standard Faster R‑CNN multi‑task loss:
- **RPN objectness loss** (classification of anchors as object/background).
- **RPN box regression loss** (refines anchor boxes).
- **ROI classification loss** (classifies each proposal).
- **ROI box regression loss** (refines final boxes per class).

These losses are produced by the torchvision Faster R‑CNN model and are summed
each iteration to form the total optimization objective.

Below is the more formal mathematical view used in Faster R‑CNN.

Let:
- `i` index anchors (RPN) and `j` index ROI proposals (ROI head).
- `p_i` = predicted objectness probability for anchor `i`.
- `p_i*` = ground‑truth objectness label (1 for positive, 0 for negative).
- `t_i` = predicted box regression offsets for anchor `i`.
- `t_i*` = ground‑truth regression targets for anchor `i`.
- `u_j` = ground‑truth class label for ROI `j`.
- `p_j` = predicted class distribution for ROI `j`.
- `t_j` = predicted box regression offsets for ROI `j`.
- `t_j*` = ground‑truth regression targets for ROI `j`.

**1) RPN objectness (binary classification)**

Binary cross‑entropy over anchors:
```
L_rpn_cls = (1 / N_cls) * Σ_i [ - p_i* log(p_i) - (1 - p_i*) log(1 - p_i) ]
```

**2) RPN box regression (Smooth L1 / Huber)**

Applied only to positive anchors:
```
L_rpn_reg = (1 / N_reg) * Σ_i [ p_i* * SmoothL1(t_i - t_i*) ]
```

**3) ROI classification (multi‑class cross‑entropy)**
```
L_roi_cls = (1 / N_roi) * Σ_j [ - log p_j(u_j) ]
```

**4) ROI box regression (Smooth L1 / Huber)**

Applied only to foreground ROIs:
```
L_roi_reg = (1 / N_roi) * Σ_j [ 1[u_j > 0] * SmoothL1(t_j - t_j*) ]
```

**Total loss**
```
L_total = L_rpn_cls + L_rpn_reg + L_roi_cls + L_roi_reg
```

In practice, these are returned by the torchvision model and summed each step.

---

**Summary**

Due to the unavailability of pretrained BDD100K weights, the model was trained from COCO‑initialized backbones. Faster R‑CNN was chosen for its accuracy and robustness to small objects and clutter. Training was capped at **12 epochs** to fit the compute/time budget, using SGD + StepLR and lightweight augmentations. Evaluation follows COCO metrics and is tracked per epoch for reproducibility and diagnostics.
