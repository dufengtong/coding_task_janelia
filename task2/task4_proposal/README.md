# Task 4 — Proposal: Improving Mitochondria Detection with Minimal Fine-Tuning

## Overview

The preceding tasks show that off-the-shelf DINO embeddings already capture meaningful mitochondrial structure. The question is how to bridge the gap from "semantically aware features" to accurate segmentation masks with minimal trainable parameters.

Below I outline four approaches, ordered from near-zero to moderate parameter count.

---

## Approach 1: Thresholding the Similarity Map (0 parameters)

**Idea:** Apply a fixed threshold to the per-pixel cosine similarity map from Task 3 to produce a binary mask.

**Method:** For a given query mitochondrion, compute the dense cosine similarity map. Sweep thresholds on a small held-out set of annotated slices and select the value that maximises F1 score. At inference, apply the threshold directly.

**Limitations:** Blurry boundaries (the similarity map is bilinearly upsampled from a coarse patch grid), sensitive to query choice, and no ability to generalise beyond the selected query.

**Use:** Zero-training baseline. Any learned approach should clearly outperform this.

---

## Approach 2: Linear Probe (~800 parameters)

**Idea:** Apply a 1×1 convolution on the dense pixel embeddings (D → 1 channel) as a per-pixel linear classifier.

**Architecture:**
- Input: dense embedding map (H × W × D), upsampled from patch grid as in Task 2
- Single 1×1 conv layer: D → 1, followed by sigmoid
- Trainable parameters: D + 1 ≈ 769 for ViT-B

**Training:**
- Loss: binary cross-entropy
- Optimiser: Adam, lr ~1e-3
- Data: a few hundred annotated 2D slices; backbone fully frozen

**Purpose:** Tests how linearly separable mitochondria are in DINO feature space. 

---

## Approach 3: Lightweight Upsampling Decoder (~800K parameters)

**Idea:** Take patch-level features from a single DINO layer and progressively upsample them with learned conv refinement, recovering sharper boundaries than bilinear interpolation alone.

**Architecture:** Starting from the patch grid (H/16 × W/16 × D), apply 4 upsampling stages:

| Stage | Operation | Channels | Output size |
|-------|-----------|----------|-------------|
| Project | 1×1 conv | D → 128 | H/16 × W/16 |
| Stage 1 | 2× upsample + 2× Conv(3×3) + BN + ReLU | 128 | H/8 × W/8 |
| Stage 2 | 2× upsample + 2× Conv(3×3) + BN + ReLU | 128 | H/4 × W/4 |
| Stage 3 | 2× upsample + 2× Conv(3×3) + BN + ReLU | 64 | H/2 × W/2 |
| Stage 4 | 2× upsample + 2× Conv(3×3) + BN + ReLU | 32 | H × W |
| Head | 1×1 conv + sigmoid | 32 → 1 | H × W |

**Training:**
- Loss: Dice + binary cross-entropy
- Optimiser: Adam, lr ~1e-4
- Data: same annotated slices as Approach 2, backbone fully frozen
- The decoder is small enough to train on a few hundred slices.

**Advantage over Approach 2:** Learns spatial upsampling filters rather than a flat linear classifier, recovering membrane-level boundary precision. 

---

## Approach 4: Fine-tune SAM on EM Data

**Idea:** Use SAM (Segment Anything Model) directly. Its image encoder already produces rich spatial features and its mask decoder is trained for promptable segmentation. Fine-tune only the mask decoder on EM mitochondria data — the image encoder stays frozen.

**Setup:**
- **Frozen:** SAM image encoder (ViT-H/L/B)
- **Trainable:** SAM mask decoder (~4M parameters) and optionally the prompt encoder
- **Prompts:** During training, simulate prompts from ground-truth masks, sample random foreground points inside each mitochondrion instance.

**Training:**
- Loss: focal loss + Dice loss on predicted masks
- Optimiser: AdamW, lr ~1e-5, backbone frozen
- Data: ground-truth instance masks from OpenOrganelle; a few hundred annotated slices might be sufficient given the small number of trainable parameters
- At inference: a point or box prompt can be used to segment individual mitochondria instances. 

**Advantage:** SAM's decoder is already optimised for promptable, instance-level segmentation. Fine-tuning it on EM data adapts the domain while preserving the general segmentation capability.

**Limitation:** SAM requires prompts at inference time, meaning a separate model is still needed to localise mitochondria before SAM can segment them.

---

## Summary

| Approach | Trainable Parameters | Training Data | Output |
|----------|---------------------|---------------|--------|
| 1. Threshold | 0 | None | Binary mask |
| 2. Linear probe | ~800 | Small (~100 slices) | Binary mask |
| 3. Upsampling decoder | ~800K | Moderate (~500 slices) | Binary mask |
| 4. Fine-tune SAM | ~4M (decoder only) | Moderate (~500 slices) | Promptable instance masks |

In general, there is a trade-off between parameter counts and the ability to model fine boundaries and richer structures. Approach 2 is a useful diagnostic for feature quality from the frozen backbone. Approach 3 is a practical strong baseline for automatic segmentation; Approach 4 is quite flexible, enabling interactive instance-level segmentation with minimal retraining but would need prompts at the inference time.
