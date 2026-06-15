# Image Reconstruction Pipeline

## Overview
This project implements image reconstruction using two inverse problem solving methods:
1. **Least Squares** - Conjugate Gradient optimization
2. **TV Regularization** - ADMM with Total Variation regularization

## Requirements
```
numpy
matplotlib
pandas
Pillow
scipy
scikit-image
```

Install with:
```bash
pip install -r requirements.txt
```

## Reproducing Results

### 1. Setup
```bash
git clone https://github.com/Samikshakotgire/Voxilgrids_imagerecon_assignment.git
cd Voxilgrids_imagerecon_assignment
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python main.py
```

This will:
- Load all 68 BSD68 test images
- Apply degradation (blur + downsample + noise)
- Reconstruct using both methods
- Save reconstruction images to `results/`
- Generate `all_images_results.csv` with metrics

### 3. Parameter Sweep
The pipeline also runs a parameter sweep testing different combinations of:
- Noise levels: 0.005, 0.01, 0.02
- Blur sigma: 0.5, 1.0, 1.5
- Downsample factors: 2, 3

Results saved to `results/results_sweep.csv`

## Output Files
- `results/*.png` - Reconstruction comparison images (68 files)
- `results/all_images_results.csv` - PSNR/SSIM metrics for each image
- `results/results_sweep.csv` - Parameter sweep analysis