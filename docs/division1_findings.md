# Division 1 Findings: BDD100K Data Analysis

This document summarizes the key findings from the BDD100K detection analysis produced by the Division 1 pipeline. All numbers below are computed from the generated summaries:
- `outputs/train/summary.json`
- `outputs/val/summary.json`

**Scope and Method**

The analysis parses the official BDD100K detection label files and computes:
- Per‑class box counts and image counts.
- Box area and aspect ratio distributions.
- Occlusion and truncation rates (from label attributes).
- Object size buckets (pixel area thresholds).
- Distance proxy buckets (box height / image height).
- Clutter levels (boxes per image).
- Rare class combinations and top‑K extreme samples (largest/smallest/widest/tallest).

Definitions used by the analysis:
- **Small/Medium/Large (pixel area)**:
  - Small: `< 32×32` pixels
  - Medium: `32×32 – 96×96` pixels
  - Large: `> 96×96` pixels
- **Distance proxy (box height ratio)**:
  - Far: `< 0.10`
  - Mid: `0.10 – 0.30`
  - Near: `> 0.30`
- **Clutter levels (boxes per image)**:
  - Low: `1–3`
  - Medium: `4–7`
  - High: `>= 8`

---

**Dataset Scale**

Train split:
- Images: **69,863**
- Total boxes: **1,286,871**
- Average boxes per image: **18.42**

Val split:
- Images: **10,000**
- Total boxes: **185,526**
- Average boxes per image: **18.55**

The train and val splits are closely aligned in scale and density.

---

**Class Frequency and Imbalance**

Top classes by box count (train):
- **car**: 713,211
- **traffic sign**: 239,686
- **traffic light**: 186,117
- **person**: 91,349
- **truck**: 29,971

Bottom classes by box count (train):
- **train**: 136
- **motor**: 3,002
- **rider**: 4,517
- **bike**: 7,210
- **bus**: 11,672

Val split shows the same pattern:
- Top: car, traffic sign, traffic light, person, truck.
- Bottom: train, motor, rider, bike, bus.

This imbalance is significant and directly impacts model performance for rare classes.

---

**Occlusion and Truncation**

Train split:
- Occluded: **47.3%** of all boxes
- Truncated: **6.9%** of all boxes

Val split:
- Occluded: **47.2%**
- Truncated: **6.8%**

Occlusion is common across both splits, so detection models must handle partial visibility.

---

**Object Size Distribution**

Train split:
- Small: **55.2%**
- Medium: **32.2%**
- Large: **12.7%**

Val split:
- Small: **55.4%**
- Medium: **32.2%**
- Large: **12.5%**

Small objects dominate the dataset, which suggests a need for strong multi‑scale features.

---

**Distance Proxy (Box Height Ratio)**

Train split:
- Far: **82.4%**
- Mid: **14.5%**
- Near: **3.1%**

Val split:
- Far: **82.6%**
- Mid: **14.3%**
- Near: **3.1%**

Most objects are far or mid‑distance, which again biases the dataset toward smaller boxes.

---

**Clutter Levels (Boxes per Image)**

Train split:
- Low clutter (1–3 boxes): **~0.0%** (4 images)
- Medium clutter (4–7 boxes): **11.7%**
- High clutter (>=8 boxes): **88.3%**

Val split:
- Low clutter: **~0.0%** (1 image)
- Medium clutter: **11.9%**
- High clutter: **88.1%**

The dataset is heavily cluttered. Most images contain many objects, increasing detection complexity and the likelihood of crowded overlaps.

---

**Class‑Level Shape Extremes**

For each class the pipeline tracks:
- **Large**: largest area boxes.
- **Small**: smallest area boxes.
- **Widest**: largest width/height ratio.
- **Tallest**: smallest width/height ratio.

These extremes are rendered as annotated images in:
- `outputs/train/visuals/`
- `outputs/val/visuals/`

These samples help identify edge‑case geometries and annotation outliers.

---

**Consistency Between Train and Val**

The following patterns are consistent across both splits:
- Class imbalance order.
- Size distribution (small objects dominate).
- Occlusion and truncation rates.
- Clutter levels (high clutter dominates).
- Distance proxy distribution (far/mid objects dominate).

This consistency indicates the train and val splits are well aligned for model development and evaluation.
