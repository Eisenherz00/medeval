# Tutorials

## Getting Started

### Installation

```bash
pip install medeval
```

### Your First Evaluation

```python
import torch
from medeval.metrics.segmentation import dice_score

# Create sample data
pred = torch.rand(4, 1, 32, 32, 32) > 0.5
target = torch.rand(4, 1, 32, 32, 32) > 0.5

# Compute Dice
dice = dice_score(pred, target, reduction="mean-case")
print(f"Mean Dice: {dice.item():.4f}")
```

## Tutorial Index

1. [Segmentation Evaluation](segmentation.md)
2. [Classification Evaluation](classification.md)
3. [Working with Spacing](spacing.md)
4. [Confidence Intervals](confidence_intervals.md)
5. [Visualization](visualization.md)
6. [CLI Usage](cli_usage.md)
7. [MONAI Integration](monai_integration.md)

