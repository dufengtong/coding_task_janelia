# EM Mitochondria Analysis

Solutions to the AI take-home challenge using EM data from [OpenOrganelle / Janelia COSEM](https://openorganelle.janelia.org).

## Setup

Create and activate the conda environment from the root of the repository:

```bash
conda env create -f environment.yml
conda activate emseg
```

## Task 1 — Data Acquisition

Four datasets were downloaded programmatically from the OpenOrganelle S3 bucket: `jrc_hela-3`, `jrc_cos7-11`, `jrc_mus-kidney`, `jrc_mus-liver`.

```bash
python task1_data_acquisition/download.py --all-mito --scale s3 --slices 10
```

See [task1_data_acquisition/task1.md](task1_data_acquisition/task1.md) for full results and visualizations.

---

## Task 2 — Feature Extraction with DINOv3

**Q1: Which patch size best captures mitochondrial ultrastructure?**

At scale s3, mitochondria span roughly 5–11 px (estimated from the median instance size across datasets). An **8 px effective patch** matches this range, so each token corresponds approximately to one mitochondrion. This is achieved by upsampling the input 2× before passing it to the ViT-B/16 model, with no model modification required.

**Q2: How to obtain dense per-pixel embeddings?**

DINOv3 produces one embedding per patch, giving a coarse grid. The patch grid is **bilinearly upsampled** to the full image resolution, yielding a dense `(H, W, 768)` embedding field. 

See [task2_dino_embeddings/task2.md](task2_dino_embeddings/task2.md) for PCA visualizations and implementation details.

---

## Task 3 — Embedding-Based Retrieval

The query mitochondrion's embedding (mean-pooled over its mask pixels, then L2-normalised) is compared to all pixel embeddings via cosine similarity.

**Q1: Within-dataset retrieval**

Layer 9 of the ViT clearly separates mitochondria from background, while layer 12 saturates and loses discriminative power. Layer 9 at 2× input scale is used as the default.

**Q2: Cross-dataset retrieval**

The kidney query vector applied directly to a liver slice still highlights mitochondria clearly. This shows that layer-9 DINOv3 features encode mitochondria-specific patterns that generalise across tissue types.

**Q3: Multiple queries**

With N query mitochondria, all query vectors are averaged into a single pooled vector before computing the similarity map. This suppresses instance-specific noise and retains only the shared structural signature. In practice, the pooled result is not substantially better than a single query, possibly due to variability in the selected instances. A potential improvement is to compute a separate similarity map for each query and select the best scoring one, which is more computationally intensive but better handles morphological diversity within the query set. This can be further visualized by comparing the individual similarity maps across queries, which reveals the variability and consistency of the retrieved regions.

See [task3_retrieval/task3.md](task3_retrieval/task3.md) for all similarity maps and score distributions.

---

## Task 4 — Proposal: Improving Detection with Minimal Fine-Tuning

Four approaches are proposed, ordered by parameter count:

| Approach | Trainable parameters | Key idea |
|----------|---------------------|----------|
| Similarity threshold | 0 | Threshold the cosine similarity map from Task 3 |
| Linear probe | ~800 | 1×1 conv on frozen dense embeddings |
| Upsampling decoder | ~800K | Learned conv decoder recovering sharp boundaries |
| Fine-tune SAM decoder | ~4M | Adapt SAM's mask decoder to EM domain; encoder frozen |

The linear probe is a useful diagnostic for feature quality. The upsampling decoder is the practical strong baseline for automatic segmentation. Fine-tuning SAM enables interactive instance-level segmentation with minimal retraining, though it requires a prompt at inference time.

See [task4_proposal/task4.md](task4_proposal/task4.md) for the full technical outline.

## Reproduction

Run the following steps in order:

**1. Download the data:**
```bash
python task1_data_acquisition/download.py --all-mito --scale s3 --slices 10
```

**2. Extract embeddings** (required before running the task 2 and task 3 notebooks):
```bash
# Scale 1× for task 2 exploration (all four datasets)
python task2_dino_embeddings/extract_embeddings.py jrc_mus-kidney_em_s3_10slices.npy --img-scale 1.0
python task2_dino_embeddings/extract_embeddings.py jrc_mus-liver_em_s3_10slices.npy --img-scale 1.0
python task2_dino_embeddings/extract_embeddings.py jrc_hela-3_em_s3_10slices.npy --img-scale 1.0
python task2_dino_embeddings/extract_embeddings.py jrc_cos7-11_em_s3_10slices.npy --img-scale 1.0

# Scale 2× for task 3 retrieval (kidney and liver only)
python task2_dino_embeddings/extract_embeddings.py jrc_mus-kidney_em_s3_10slices.npy --img-scale 2.0
python task2_dino_embeddings/extract_embeddings.py jrc_mus-liver_em_s3_10slices.npy --img-scale 2.0
```

**3. Run the notebooks in order:**
```bash
jupyter nbconvert --to notebook --execute --inplace task1_data_acquisition/task1_visualize_slices.ipynb
jupyter nbconvert --to notebook --execute --inplace task2_dino_embeddings/task2_dino_embeddings.ipynb
jupyter nbconvert --to notebook --execute --inplace task3_retrieval/task3_retrieval.ipynb
```