#!/bin/bash
# MedEval CLI Usage Examples
# ==========================
#
# This script demonstrates how to use the medeval CLI for batch evaluation.

echo "============================================"
echo "MedEval CLI Usage Examples"
echo "============================================"

# Example 1: Basic segmentation evaluation
echo ""
echo "1. Basic Segmentation Evaluation"
echo "--------------------------------"
echo "Command:"
echo '  medeval evaluate --manifest data/manifest.csv --task segmentation --output results/'
echo ""

# Example 2: With configuration file
echo "2. Evaluation with Config File"
echo "------------------------------"
echo "Command:"
echo '  medeval evaluate --manifest data/manifest.csv --config config.yaml --output results/'
echo ""

# Example 3: Classification task
echo "3. Classification Evaluation"
echo "----------------------------"
echo "Command:"
echo '  medeval evaluate --manifest data/classification_manifest.csv --task classification -v'
echo ""

# Create example manifest
echo "Creating example manifest file..."
cat > /tmp/example_manifest.csv << 'EOF'
prediction,target,spacing,patient_id,strata
/path/to/pred1.nii.gz,/path/to/target1.nii.gz,"1.0,1.0,2.0",patient_001,site_A
/path/to/pred2.nii.gz,/path/to/target2.nii.gz,"1.0,1.0,2.0",patient_002,site_A
/path/to/pred3.nii.gz,/path/to/target3.nii.gz,"1.0,1.0,2.0",patient_003,site_B
EOF

echo "Example manifest created at /tmp/example_manifest.csv"
echo ""
cat /tmp/example_manifest.csv
echo ""

# Create example config
echo "Creating example config file..."
cat > /tmp/example_config.yaml << 'EOF'
# MedEval Configuration
task: segmentation

# Column mappings in manifest
columns:
  prediction: prediction
  target: target
  spacing: spacing
  patient_id: patient_id
  strata: strata

# Metric-specific settings
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
EOF

echo "Example config created at /tmp/example_config.yaml"
echo ""
cat /tmp/example_config.yaml
echo ""

# Show help
echo "4. View Help"
echo "------------"
echo "Command:"
echo '  medeval --help'
echo '  medeval evaluate --help'
echo ""

echo "============================================"
echo "For more examples, see the documentation at:"
echo "https://github.com/your-repo/medeval"
echo "============================================"

