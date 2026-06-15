# Image Reconstruction Results Report

## Summary
Pipeline processed **68 BSD68 test images** using two reconstruction methods:
- **Least Squares (Conjugate Gradient)**
- **TV Regularization (ADMM)**

## Results Overview

### Average Metrics Across All Images
| Method | Avg PSNR (dB) | Avg SSIM |
|--------|---------------|----------|
| Least Squares | 25.41 | 0.75 |
| TV Regularization | 17.49 | 0.53 |

### Key Observations

1. **Least Squares outperforms TV Regularization** in both PSNR and SSIM metrics
   - Higher PSNR indicates better reconstruction fidelity
   - Higher SSIM indicates better structural similarity to original

2. **Least Squares PSNR range**: 21.59 - 29.56 dB
   - Best: test017 (29.56 dB)
   - Worst: test011 (22.03 dB)

3. **TV Regularization PSNR range**: 14.37 - 20.71 dB
   - Best: test017 (20.71 dB)
   - Worst: test011 (14.37 dB)

4. **SSIM patterns** follow similar trends to PSNR, with Least Squares consistently achieving higher structural similarity

## Parameter Sweep Results
Tested combinations of noise levels (0.005, 0.01, 0.02), blur sigma (0.5, 1.0, 1.5), and downsample factors (2, 3).

Key findings:
- Lower noise levels yield better reconstruction quality
- Higher blur sigma degrades reconstruction performance
- Larger downsample factors (more aggressive downsampling) reduce quality

## Files Generated
- `results/01-68_testXXX_reconstruction.png` - Individual reconstruction comparisons
- `results/all_images_results.csv` - Complete metrics table
- `results/results_sweep.csv` - Parameter sweep analysis