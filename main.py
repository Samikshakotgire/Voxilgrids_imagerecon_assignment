import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import cg, LinearOperator
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r"D:\Projects-CORE\image_recon\BSD68" # Local path to dataset
OUTPUT_DIR = r"D:\Projects-CORE\image_recon\results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Data Loading
# ============================================================================
def load_bsd68_images(data_path=DATA_PATH, num_images=68):
    """Load BSD68 images from local folder."""
    images, image_names = [], []
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Please check instructions.")
        return [], []
    
    img_files = sorted(os.listdir(data_path))
    if num_images is not None:
        img_files = img_files[:num_images]
        
    for img_file in img_files:
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_path, img_file)
            img = Image.open(img_path).convert('L')  # Grayscale
            img = np.array(img).astype(np.float32) / 255.0
            
            # Resize to 128x128 for faster computation
            if img.shape[0] > 128:
                img = Image.fromarray((img * 255).astype(np.uint8))
                img = img.resize((128, 128), Image.Resampling.LANCZOS)
                img = np.array(img).astype(np.float32) / 255.0
            
            images.append(img)
            image_names.append(img_file.split('.')[0])
    return images, image_names

# ============================================================================
# 2. Forward Model
# ============================================================================
class ForwardModel:
    def __init__(self, blur_sigma=1.0, downsample_factor=2, noise_std=0.01):
        self.blur_sigma = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_std = noise_std
    
    def blur(self, x):
        return gaussian_filter(x, sigma=self.blur_sigma)
    
    def downsample(self, x):
        return x[::self.downsample_factor, ::self.downsample_factor]
    
    def forward(self, x, add_noise=True):
        y = self.blur(x)
        y = self.downsample(y)
        if add_noise:
            noise = np.random.normal(0, self.noise_std, y.shape)
            y = np.clip(y + noise, 0, 1)
        return y
    
    def adjoint_downsample(self, y, original_shape):
        z = np.zeros(original_shape)
        z[::self.downsample_factor, ::self.downsample_factor] = y
        return z
    
    def adjoint_blur(self, y):
        return gaussian_filter(y, sigma=self.blur_sigma)

# ============================================================================
# 3. Solvers
# ============================================================================
class LeastSquaresSolver:
    def __init__(self, forward_model, max_iter=100, tol=1e-5):
        self.fm = forward_model
        self.max_iter = max_iter
        self.tol = tol
    
    def apply_operator(self, x):
        return self.fm.downsample(self.fm.blur(x))
    
    def apply_adjoint(self, y, original_shape):
        upsampled = self.fm.adjoint_downsample(y, original_shape)
        return self.fm.adjoint_blur(upsampled)
    
    def solve(self, y, x_true_shape, verbose=False):
        def matvec(x_flat):
            x = x_flat.reshape(x_true_shape)
            Hx = self.apply_operator(x)
            result = self.apply_adjoint(Hx, x_true_shape)
            return result.flatten()
        
        b = self.apply_adjoint(y, x_true_shape).flatten()
        A = LinearOperator((np.prod(x_true_shape), np.prod(x_true_shape)), matvec=matvec)
        
        x_recon_flat, info = cg(A, b, maxiter=self.max_iter, atol=self.tol)
        if verbose:
            print(f"  CG converged: {info == 0}, iterations: {info if info > 0 else self.max_iter}")
        
        return np.clip(x_recon_flat.reshape(x_true_shape), 0, 1)

class ADMMTVSolver:
    def __init__(self, forward_model, lam=0.01, rho=1.0, max_iter=30):
        self.fm = forward_model
        self.lam = lam
        self.rho = rho
        self.max_iter = max_iter
    
    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def compute_gradients(self, x):
        gx = np.zeros_like(x)
        gx[:, :-1] = np.diff(x, axis=1)
        gx[:, -1] = x[:, 0] - x[:, -1]
        
        gy = np.zeros_like(x)
        gy[:-1, :] = np.diff(x, axis=0)
        gy[-1, :] = x[0, :] - x[-1, :]
        return gx, gy
    
    def solve(self, y, x_true_shape, verbose=False):
        x = np.zeros(x_true_shape)
        ux = np.zeros(x_true_shape)
        uy = np.zeros(x_true_shape)
        
        for _ in range(self.max_iter):
            residual = y - self.fm.downsample(self.fm.blur(x))
            grad_data = 2 * self.fm.adjoint_blur(self.fm.adjoint_downsample(residual, x_true_shape))
            grad_reg = 2 * self.rho * (ux + uy)
            
            x = np.clip(x + 0.1 * (grad_data + grad_reg), 0, 1)
            gx, gy = self.compute_gradients(x)
            ux = self.soft_threshold(gx, self.lam / self.rho)
            uy = self.soft_threshold(gy, self.lam / self.rho)
        return x

# ============================================================================
# 4. Metrics & Execution
# ============================================================================
def compute_metrics(x_true, x_recon):
    return {
        'MSE': np.mean((x_true - x_recon) ** 2),
        'PSNR': peak_signal_noise_ratio(x_true, x_recon, data_range=1.0),
        'SSIM': structural_similarity(x_true, x_recon, data_range=1.0)
    }

def main():
    print("Loading Dataset...")
    original_images, image_names = load_bsd68_images()
    if not original_images:
        return
    
    all_results = []
    
    for idx, (x_test, name) in enumerate(zip(original_images, image_names)):
        print(f"\n{'='*50}")
        print(f"Processing image {idx+1}/{len(original_images)}: {name}")
        print(f"{'='*50}")
        
        fm = ForwardModel(blur_sigma=1.0, downsample_factor=2, noise_std=0.01)
        y_test = fm.forward(x_test, add_noise=True)
        
        print("Solving with Least Squares...")
        ls_solver = LeastSquaresSolver(fm, max_iter=50)
        x_recon_ls = ls_solver.solve(y_test, x_test.shape, verbose=True)
        
        print("Solving with TV Regularization...")
        tv_solver = ADMMTVSolver(fm, lam=0.01, rho=1.0, max_iter=30)
        x_recon_tv = tv_solver.solve(y_test, x_test.shape, verbose=True)
        
        metrics_ls = compute_metrics(x_test, x_recon_ls)
        metrics_tv = compute_metrics(x_test, x_recon_tv)
        
        print(f"\nLeast Squares -> PSNR: {metrics_ls['PSNR']:.2f} dB, SSIM: {metrics_ls['SSIM']:.4f}")
        print(f"TV Reg        -> PSNR: {metrics_tv['PSNR']:.2f} dB, SSIM: {metrics_tv['SSIM']:.4f}")
        
        all_results.append({
            'image': name,
            'ls_psnr': metrics_ls['PSNR'], 'ls_ssim': metrics_ls['SSIM'],
            'tv_psnr': metrics_tv['PSNR'], 'tv_ssim': metrics_tv['SSIM']
        })
        
        # Visualization (Reconstruction)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes[0, 0].imshow(x_test, cmap='gray'); axes[0, 0].set_title('Original Image x')
        axes[0, 1].imshow(y_test, cmap='gray'); axes[0, 1].set_title('Degraded Image y')
        axes[0, 2].imshow(x_test, cmap='gray', alpha=0.5)
        axes[0, 2].imshow(y_test, cmap='gray', alpha=0.5); axes[0, 2].set_title('Overlay')
        
        axes[1, 0].imshow(x_recon_ls, cmap='gray'); axes[1, 0].set_title(f'Least Squares\nPSNR: {metrics_ls["PSNR"]:.2f} dB')
        axes[1, 1].imshow(x_recon_tv, cmap='gray'); axes[1, 1].set_title(f'TV Regularized\nPSNR: {metrics_tv["PSNR"]:.2f} dB')
        diff_tv = np.abs(x_test - x_recon_tv)
        im = axes[1, 2].imshow(diff_tv, cmap='hot'); axes[1, 2].set_title('TV Error Map')
        plt.colorbar(im, ax=axes[1, 2])
        
        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{idx+1:02d}_{name}_reconstruction.png'), dpi=150)
        plt.close()
        print(f"Saved visual comparison to {OUTPUT_DIR}/{idx+1:02d}_{name}_reconstruction.png")
    
    # Save all results to CSV
    df_results = pd.DataFrame(all_results)
    results_csv_path = os.path.join(OUTPUT_DIR, 'all_images_results.csv')
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nAll images results saved to {results_csv_path}")
    
    # Parameter Sweep (using first image)
    print("\nRunning parameter sweep...")
    results_sweep = []
    x_test = original_images[0]  # Use first image for sweep
    for noise in [0.005, 0.01, 0.02]:
        for blur in [0.5, 1.0, 1.5]:
            for downsample in [2, 3]:
                fm_p = ForwardModel(blur_sigma=blur, downsample_factor=downsample, noise_std=noise)
                y = fm_p.forward(x_test)
                
                x_ls = LeastSquaresSolver(fm_p, max_iter=50).solve(y, x_test.shape)
                x_tv = ADMMTVSolver(fm_p, lam=0.01, max_iter=20).solve(y, x_test.shape)
                
                mls = compute_metrics(x_test, x_ls)
                mtv = compute_metrics(x_test, x_tv)
                
                results_sweep.append({
                    'noise': noise, 'blur_sigma': blur, 'downsample': downsample,
                    'ls_psnr': mls['PSNR'], 'ls_ssim': mls['SSIM'],
                    'tv_psnr': mtv['PSNR'], 'tv_ssim': mtv['SSIM']
                })
    
    df = pd.DataFrame(results_sweep)
    csv_path = os.path.join(OUTPUT_DIR, 'results_sweep.csv')
    df.to_csv(csv_path, index=False)
    print(f"Parameter sweep complete. Results saved to {csv_path}")

if __name__ == "__main__":
    main()