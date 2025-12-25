# Metric Definitions

This section provides mathematical definitions for all metrics implemented in MedEval.

## Segmentation Metrics

### Overlap Metrics

#### Dice Score (F1)

The Dice coefficient measures the overlap between two sets:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

- Range: [0, 1], higher is better
- Edge case: If both sets are empty, Dice = 1.0 (perfect match)

#### Jaccard Index (IoU)

The Jaccard index (Intersection over Union):

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

- Range: [0, 1], higher is better
- Relationship: $\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$

#### Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

#### Volumetric Similarity

$$\text{VS} = 1 - \frac{|V_A - V_B|}{V_A + V_B}$$

where $V_A$ and $V_B$ are the volumes of sets A and B.

### Surface Metrics

#### Hausdorff Distance

The (symmetric) Hausdorff distance:

$$\text{HD}(A, B) = \max\left(\max_{a \in S_A} \min_{b \in S_B} d(a, b), \max_{b \in S_B} \min_{a \in S_A} d(a, b)\right)$$

where $S_A$ and $S_B$ are the surfaces and $d$ is Euclidean distance.

- **HD95**: 95th percentile instead of maximum (more robust to outliers)

#### Average Symmetric Surface Distance (ASSD)

$$\text{ASSD} = \frac{1}{|S_A| + |S_B|}\left(\sum_{a \in S_A} \min_{b \in S_B} d(a,b) + \sum_{b \in S_B} \min_{a \in S_A} d(a,b)\right)$$

#### Surface Dice

Fraction of surface points within tolerance distance $\tau$:

$$\text{SD}(\tau) = \frac{|S_A^\tau| + |S_B^\tau|}{|S_A| + |S_B|}$$

where $S_A^\tau = \{a \in S_A : \min_{b \in S_B} d(a,b) \leq \tau\}$

### Calibration Metrics

#### Soft Dice

For probabilistic predictions $p$:

$$\text{Soft-Dice} = \frac{2 \sum_i p_i \cdot t_i}{\sum_i p_i + \sum_i t_i}$$

#### Brier Score

$$\text{Brier} = \frac{1}{N} \sum_i (p_i - t_i)^2$$

---

## Classification Metrics

### Discrimination

#### AUROC

Area Under the Receiver Operating Characteristic curve. Measures the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample.

#### AUPRC

Area Under the Precision-Recall Curve. More informative for imbalanced datasets.

#### Matthews Correlation Coefficient

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Range: [-1, 1], handles class imbalance well.

### Calibration

#### Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|$$

where bins $b$ are based on predicted confidence.

#### Adaptive ECE (AECE)

Same as ECE but with adaptive binning (equal samples per bin).

---

## Detection Metrics

### Mean Average Precision (mAP)

$$\text{mAP} = \frac{1}{|C|} \sum_{c \in C} \text{AP}_c$$

where AP is computed from precision-recall curve.

### FROC

Free-Response ROC plots sensitivity vs. average false positives per image.

---

## Registration Metrics

### Target Registration Error (TRE)

$$\text{TRE} = \sqrt{\sum_d (p_d - t_d)^2 \cdot s_d^2}$$

where $s_d$ is spacing in dimension $d$.

### Normalized Mutual Information

$$\text{NMI} = \frac{H(A) + H(B)}{H(A, B)}$$

where $H$ is entropy.

### Jacobian Determinant

For deformation field $\phi$, the Jacobian determinant at each point:

$$|J(\phi)| = \det\left(\frac{\partial \phi}{\partial x}\right)$$

- $|J| < 0$: indicates folding (physically impossible)
- $|J| = 1$: volume-preserving
- $|J| > 1$: expansion
- $0 < |J| < 1$: compression

---

## Common Pitfalls

### Class Imbalance

- **Dice/Jaccard**: Can be dominated by larger structures
- **Recommendation**: Report per-class metrics

### Anisotropic Spacing

- **Surface metrics**: Must use physical distances, not voxel distances
- **Recommendation**: Always pass spacing to surface metric functions

### Empty Predictions

- **Dice**: Both empty = 1.0, one empty = 0.0
- **HD**: Returns infinity if either surface is empty

### Calibration vs. Discrimination

- A model can have good AUROC but poor calibration
- Always report both when relevant

