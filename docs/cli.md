# CLI Reference

MedEval provides a command-line interface for batch evaluation.

## Basic Usage

```bash
medeval <command> [options]
```

## Commands

### evaluate

Evaluate metrics on a dataset from a manifest file.

```bash
medeval evaluate --manifest <path> --task <task> [options]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--manifest` | Path to CSV or JSON manifest file |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | segmentation | Task type: segmentation, classification, detection, registration |
| `--config` | - | Path to YAML/JSON configuration file |
| `--output` | results/ | Output directory for results |
| `-v, --verbose` | False | Enable verbose logging |

## Manifest Format

### CSV Format

```csv
prediction,target,spacing,patient_id,strata
/data/pred1.nii.gz,/data/gt1.nii.gz,"1.0,1.0,2.0",patient_001,site_A
/data/pred2.nii.gz,/data/gt2.nii.gz,"1.0,1.0,2.0",patient_002,site_B
```

### JSON Format

```json
[
  {
    "prediction": "/data/pred1.nii.gz",
    "target": "/data/gt1.nii.gz",
    "spacing": [1.0, 1.0, 2.0],
    "patient_id": "patient_001",
    "strata": "site_A"
  }
]
```

### Column Descriptions

| Column | Required | Description |
|--------|----------|-------------|
| `prediction` | Yes | Path to prediction file |
| `target` | Yes | Path to ground truth file |
| `spacing` | No | Physical spacing (comma-separated or list) |
| `patient_id` | No | Patient/study identifier |
| `strata` | No | Stratification label (e.g., site, scanner) |

## Configuration File

### YAML Format

```yaml
# Task type
task: segmentation

# Column mappings
columns:
  prediction: prediction
  target: target
  spacing: spacing
  patient_id: patient_id
  strata: strata

# Metric settings
metrics:
  threshold: 0.5
  ignore_index: null
  include_surface: true
  include_calibration: false

# Aggregation settings
aggregation:
  n_bootstrap: 1000
  confidence: 0.95
  seed: 42
```

### Configuration Options

#### Segmentation Metrics

| Option | Default | Description |
|--------|---------|-------------|
| `threshold` | 0.5 | Threshold for binary predictions |
| `ignore_index` | null | Label index to ignore |
| `include_surface` | true | Compute surface metrics |
| `include_calibration` | false | Compute calibration metrics |

#### Classification Metrics

| Option | Default | Description |
|--------|---------|-------------|
| `include_calibration` | true | Compute calibration metrics |

#### Aggregation

| Option | Default | Description |
|--------|---------|-------------|
| `n_bootstrap` | 1000 | Number of bootstrap samples |
| `confidence` | 0.95 | Confidence level for CI |
| `seed` | 42 | Random seed for reproducibility |

## Output Files

The `evaluate` command produces:

### results.csv

Per-case results with all computed metrics.

```csv
index,status,patient_id,strata,dice,jaccard,...
0,success,patient_001,site_A,0.85,0.74,...
1,success,patient_002,site_B,0.78,0.64,...
```

### summary.json

Aggregated results with confidence intervals.

```json
{
  "task": "segmentation",
  "n_samples": 100,
  "n_processed": 100,
  "n_errors": 0,
  "metrics": {
    "dice": {
      "mean": 0.82,
      "ci_lower": 0.79,
      "ci_upper": 0.85,
      "std": 0.08,
      "median": 0.83,
      "iqr": [0.77, 0.88]
    }
  },
  "stratified_metrics": {
    "dice": {
      "site_A": [0.84, 0.80, 0.88],
      "site_B": [0.80, 0.75, 0.85]
    }
  }
}
```

## Examples

### Basic Segmentation Evaluation

```bash
medeval evaluate \
  --manifest data/segmentation_manifest.csv \
  --task segmentation \
  --output results/segmentation/
```

### With Configuration

```bash
medeval evaluate \
  --manifest data/manifest.csv \
  --config configs/segmentation_config.yaml \
  --output results/ \
  --verbose
```

### Classification Task

```bash
medeval evaluate \
  --manifest data/classification_manifest.csv \
  --task classification \
  -v
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (invalid arguments, missing files, etc.) |

