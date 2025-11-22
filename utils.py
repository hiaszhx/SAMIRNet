import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, save_path):
    """
    Save model checkpoint
    
    Args:
        state: dict containing model state, optimizer state, etc.
        save_path: path to save checkpoint
    """
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: path to checkpoint
        model: model to load weights into
        optimizer: optimizer to load state into (optional)
    
    Returns:
        epoch: epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return epoch


def visualize_predictions(images, masks, predictions, save_path=None, num_samples=4):
    """
    Visualize predictions alongside ground truth
    
    Args:
        images: (B, C, H, W) input images
        masks: (B, 1, H, W) ground truth masks
        predictions: (B, 1, H, W) predicted masks
        save_path: path to save visualization
        num_samples: number of samples to visualize
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Image
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground Truth
        gt = masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = predictions[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()


def visualize_heatmap(image, heatmap, centers=None, save_path=None):
    """
    Visualize detection heatmap with centers
    
    Args:
        image: (H, W) or (C, H, W) input image
        heatmap: (H, W) detection heatmap
        centers: list of (x, y) center coordinates
        save_path: path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = image.transpose(1, 2, 0)
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Detection Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    
    if centers is not None:
        for cx, cy in centers:
            axes[2].plot(cx, cy, 'r*', markersize=15)
    
    axes[2].set_title('Overlay with Centers')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap visualization saved to {save_path}")
    
    plt.close()


def create_gaussian_heatmap(center, shape, sigma=2):
    """
    Create Gaussian heatmap for a single center point
    
    Args:
        center: (x, y) center coordinate
        shape: (H, W) output shape
        sigma: Gaussian kernel sigma
    
    Returns:
        heatmap: (H, W) Gaussian heatmap
    """
    H, W = shape
    cx, cy = center
    
    x = np.arange(0, W, 1, dtype=np.float32)
    y = np.arange(0, H, 1, dtype=np.float32)
    y = y[:, np.newaxis]
    
    # Gaussian kernel
    heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    
    return heatmap


def extract_centers_from_mask(mask, min_area=1):
    """
    Extract center points from binary mask
    
    Args:
        mask: (H, W) binary mask
        min_area: minimum area threshold for valid targets
    
    Returns:
        centers: list of (x, y) center coordinates
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    centers = []
    
    # Process each component (skip background label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            cx, cy = centroids[i]
            centers.append((cx, cy))
    
    return centers


def apply_nms_to_heatmap(heatmap, kernel_size=3, threshold=0.3):
    """
    Apply Non-Maximum Suppression to heatmap
    
    Args:
        heatmap: (H, W) detection heatmap
        kernel_size: NMS kernel size
        threshold: confidence threshold
    
    Returns:
        peaks: list of (x, y) peak coordinates
        scores: list of confidence scores
    """
    # Max pooling
    max_pooled = cv2.dilate(
        heatmap,
        np.ones((kernel_size, kernel_size)),
        iterations=1
    )
    
    # Find local maxima
    peaks_mask = (heatmap == max_pooled) & (heatmap > threshold)
    
    # Get coordinates
    y_coords, x_coords = np.where(peaks_mask)
    scores = heatmap[y_coords, x_coords]
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    peaks = [(x_coords[i], y_coords[i]) for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    return peaks, scores


def compute_model_complexity(model):
    """
    Compute model complexity (parameters and FLOPs)
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Dictionary containing complexity metrics
    """
    from thop import profile, clever_format
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute FLOPs
    input_tensor = torch.randn(1, 3, 256, 256)
    if next(model.parameters()).is_cuda:
        input_tensor = input_tensor.cuda()
    
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    complexity = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'params_str': params
    }
    
    return complexity


def save_results_to_csv(results, save_path):
    """
    Save evaluation results to CSV file
    
    Args:
        results: dict of results
        save_path: path to save CSV
    """
    import csv
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max'])
        
        # Data
        for metric_name, values in results.items():
            writer.writerow([
                metric_name,
                f"{values['mean']:.4f}",
                f"{values['std']:.4f}",
                f"{values['median']:.4f}",
                f"{values['min']:.4f}",
                f"{values['max']:.4f}"
            ])
    
    print(f"Results saved to {save_path}")


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file
    
    Args:
        log_file: path to training log
        save_path: path to save plot
    """
    # Parse log file
    epochs = []
    train_losses = []
    val_ious = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss' in line:
                # Parse epoch and loss
                parts = line.split()
                epoch = int(parts[1].strip('[]').split('/')[0])
                loss = float(parts[-1])
                
                epochs.append(epoch)
                train_losses.append(loss)
            
            elif 'Val IoU' in line:
                # Parse validation IoU
                parts = line.split()
                iou = float(parts[2])
                val_ious.append(iou)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation IoU
    ax2.plot(epochs, val_ious, 'r-', label='Val IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """
    
    def __init__(self, patience=20, min_delta=0.0001, mode='max'):
        """
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: 'max' or 'min' (whether higher or lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test Gaussian heatmap
    heatmap = create_gaussian_heatmap((128, 128), (256, 256), sigma=3)
    print(f"Heatmap shape: {heatmap.shape}, max: {heatmap.max():.4f}")
    
    # Test NMS
    peaks, scores = apply_nms_to_heatmap(heatmap, kernel_size=3, threshold=0.1)
    print(f"Found {len(peaks)} peaks")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg:.2f}")
    
    # Test EarlyStopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        stop = early_stopping(score)
        print(f"Epoch {i}: score={score:.2f}, stop={stop}")
        if stop:
            break
