# Task 3 — Embedding-Based Retrieval & Visualization

## Requirement

> 1. **Within-dataset retrieval:** Visualize how the query mitochondrion's embeddings compare
>    to the embeddings of other mitochondria *within* each dataset.
> 2. **Cross-dataset retrieval:** Visualize how the query mitochondrion's embeddings compare
>    to the embeddings of mitochondria *across* datasets.
> 3. **Multiple queries:** Describe how you would adapt the retrieval strategy with multiple
>    query mitochondria. What changes would you expect in the results?

---

## Pipeline

1. Load pre-computed patch embeddings (`.npz`) and bilinearly upsample to full pixel resolution.
2. Select a mitochondrion by instance ID from the segmentation mask.
3. Pool all dense embedding vectors within the mask → **mean query vector** `q`.
4. L2-normalise `q` and all pixel vectors. The dot product gives the **cosine similarity**:

$$\text{sim}(y,x) = \frac{\mathbf{d}_{y,x}^\top \mathbf{q}}{\|\mathbf{d}_{y,x}\| \cdot \|\mathbf{q}\|}$$

5. Visualize the similarity map and compare score distributions on the slices.

| Parameter | Value |
|-----------|-------|
| Query dataset | `jrc_mus-kidney` |
| Query mito ID | 286777 |
| Query slice | 6 |
| Model layer | 9 (default) — swept over 3, 6, 9, 12 |
| Image scale | 1× and 2× |
| Multi-query N | 10 (above-median size) |

---

## 1 — Within-Dataset Retrieval

### 1a — Layer comparison (scale 1×)

| | Layer 9 | Layer 12 |
|---|---|---|
| **Full slice** | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale1_layer9.png" width="100%"> | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale1_layer12.png" width="100%"> |
| **Cropped** | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale1_layer9.png" width="100%"> | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale1_layer12.png" width="100%"> |

**Score distributions (mito vs background):**

| Layer 9 | Layer 12 |
|---|---|
| <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale1_layer9.png" width="100%"> | <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale1_layer12.png" width="100%"> |

Layer 9 gives clear mito/background separation. Layer 12 saturates: both distributions collapse to nearly 1 with minimal discriminative power, as the final ViT block compresses all tokens toward a more universal representation.

---

### 1b — Layer sweep (scale 2×)

Input upsampled 2× increases the patch grid from ~14×14 to ~28×28, giving finer spatial resolution in the similarity maps.

| | Layer 3 | Layer 6 | Layer 9 | Layer 12 |
|---|---|---|---|---|
| **Full slice** | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale2_layer3.png" width="200"> | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale2_layer6.png" width="200"> | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale2_layer9.png" width="200"> | <img src="../outputs/task3/task3_corr_mito286777_slice6_jrc_mus-kidney_imagescale2_layer12.png" width="200"> |
| **Cropped** | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale2_layer3.png" width="200"> | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale2_layer6.png" width="200"> | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale2_layer9.png" width="200"> | <img src="../outputs/task3/task3_corr_crop_mito286777_slice6_jrc_mus-kidney_imagescale2_layer12.png" width="200"> |

**Score distributions:**

| Layer 3 | Layer 6 | Layer 9 | Layer 12 |
|---|---|---|---|
| <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale2_layer3.png" width="200"> | <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale2_layer6.png" width="200"> | <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale2_layer9.png" width="200"> | <img src="../outputs/task3/task3_hist_mito286777_slice6_jrc_mus-kidney_imagescale2_layer12.png" width="200"> |

Layers 6 and 9 both show strong separation. Layer 3 captures low-level texture but also seems to saturate; layer 9 gives better spatial alignment with organelle boundaries at 2× compared to 1×. Layer 12 again saturates. Layer 9 is used as the default for its combination of semantic content and spatial precision.

| Layer | Separation | Recommendation |
|-------|-----------|----------------|
| 3 | Strong (texture) | Good for texture-only queries |
| 6 | Good | Balanced |
| **9** | **Strong (semantic)** | **Default** |
| 12 | Poor (saturated) | Not suitable |

---

## 2 — Cross-Dataset Retrieval

The query vector from **kidney mito 286777** (slice 6, layer 9, scale 2×) is applied directly to a **liver** slice (slice 5, same settings).

| Full slice | Cropped |
|---|---|
| <img src="../outputs/task3/task3_corr_query-jrc_mus-kidney_mito286777_slice6_target-jrc_mus-liver_slice5_layer9.png" width="100%"> | <img src="../outputs/task3/task3_corr_crop_query-jrc_mus-kidney_mito286777_slice6_target-jrc_mus-liver_slice5_layer9.png" width="90%"> |

**Score distribution:**

<img src="../outputs/task3/task3_hist_query-jrc_mus-kidney_mito286777_slice6_target-jrc_mus-liver_slice5_layer9.png" width="50%">

Despite no fine-tuning and the query coming from a different tissue (kidney vs liver), the similarity map still highlights mitochondria in the liver slice. The score distributions show that the separation is still clear. This demonstrates that layer-9 DINO3 features encode mitochondria-specific patterns shared across cell types.

---

## 3 — Multiple Queries

When N > 1 queries are used, the mean of all query vectors is computed as a single pooled query `q_pool`:

$$\mathbf{q}_\text{pool} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{q}_i$$

The pooled vector is L2-normalised before computing the similarity map, as before. N = 10 above-median-size mitochondria are used.

### Within-dataset (kidney, N = 10)

| Full slice | Cropped |
|---|---|
| <img src="../outputs/task3/task3_multi_corr_query-jrc_mus-kidney_N10_slice6_layer9.png" width="100%"> | <img src="../outputs/task3/task3_multi_corr_crop_query-jrc_mus-kidney_N10_slice6_layer9.png" width="85%"> |

<img src="../outputs/task3/task3_multi_hist_query-jrc_mus-kidney_N10_slice6_layer9.png" width="60%">

### Cross-dataset (kidney N = 10 → liver)

| Full slice | Cropped |
|---|---|
| <img src="../outputs/task3/task3_multi_corr_query-jrc_mus-kidney_N10_slice6_target-jrc_mus-liver_slice5_layer9.png" width="100%"> | <img src="../outputs/task3/task3_multi_corr_crop_query-jrc_mus-kidney_N10_slice6_target-jrc_mus-liver_slice5_layer9.png" width="90%"> |

<img src="../outputs/task3/task3_multi_hist_query-jrc_mus-kidney_N10_slice6_target-jrc_mus-liver_slice5_layer9.png" width="60%">

Multi-query cross-dataset retrieval yields similar mito and background score distributions between the two datasets.

The result is not substantially better than a single query instance, possibly due to the quality of the selected mitochondria. One possible improvement is to compute the similarity map for each query mitochondrion individually and select the best match. This approach is more computationally intensive but can capture the morphological variety within the query set for better retrieval.