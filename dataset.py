import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class IRSTDataset(Dataset):
    """
    IRST Dataset for BasicIRSTD framework
    Optimized: Caches centers to avoid repeated cv2 calculations
    """
    
    def __init__(self, base_dir, mode, dataset_name, transform=None, return_center=True):
        self.base_dir = base_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.transform = transform
        self.return_center = return_center
        
        # Paths
        self.dataset_dir = os.path.join(base_dir, dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.mask_dir = os.path.join(self.dataset_dir, 'masks')
        
        # Load image list
        list_file = os.path.join(
            self.dataset_dir, 
            'img_idx', 
            f'{mode}_{dataset_name}.txt'
        )
        
        with open(list_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_names)} images for {mode} from {dataset_name}")

        # Optimization: Pre-calculate centers for validation/test (where masks don't change)
        self.cached_centers = {}
        if self.return_center:
            print(f"Pre-calculating centers for {mode} set...")
            for img_name in self.image_names:
                try:
                    mask_path = self._find_image_path(self.mask_dir, img_name)
                    mask = np.array(Image.open(mask_path).convert('L'))
                    mask = (mask > 0).astype(np.float32)
                    self.cached_centers[img_name] = self._extract_centers(mask)
                except Exception as e:
                    print(f"Warning: Failed to load mask for {img_name}: {e}")
                    self.cached_centers[img_name] = np.zeros((0, 2), dtype=np.float32)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        img_path = self._find_image_path(self.image_dir, img_name)
        mask_path = self._find_image_path(self.mask_dir, img_name)
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # If transformed, we MUST re-calculate centers from the new mask
            if self.return_center:
                centers = self._extract_centers(mask)
                heatmap = self._generate_heatmap(mask.shape, centers)
        else:
            # No transform (Validation/Test): Use cached centers
            if self.return_center:
                centers = self.cached_centers.get(img_name, np.zeros((0, 2), dtype=np.float32))
                heatmap = self._generate_heatmap(mask.shape, centers)
        
        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.return_center:
            heatmap = torch.from_numpy(heatmap).unsqueeze(0)
            centers = torch.from_numpy(centers).float()
            
            return {
                'image': image,
                'mask': mask,
                'heatmap': heatmap,
                'centers': centers,
                'img_name': img_name
            }
        else:
            return {
                'image': image,
                'mask': mask,
                'img_name': img_name
            }
    
    def _find_image_path(self, directory, img_name):
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        if os.path.splitext(img_name)[1]:
            full_path = os.path.join(directory, img_name)
            if os.path.exists(full_path): return full_path
        for ext in extensions:
            full_path = os.path.join(directory, img_name + ext)
            if os.path.exists(full_path): return full_path
        raise FileNotFoundError(f"Cannot find image file for '{img_name}' in '{directory}'.")
    
    def _extract_centers(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        # connectivity=4 is slightly faster and sufficient for centers
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=4
        )
        
        centers = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 1:
                cx, cy = centroids[i]
                centers.append([cx, cy])
        
        if len(centers) == 0:
            centers = np.zeros((0, 2), dtype=np.float32)
        else:
            centers = np.array(centers, dtype=np.float32)
        return centers
    
    def _generate_heatmap(self, shape, centers, sigma=2):
        H, W = shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        if len(centers) == 0: return heatmap
        
        # Vectorized Gaussian generation for speed
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        for cx, cy in centers:
            # Standard Gaussian formula
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap

def get_loader(base_dir, mode, dataset_name, batch_size, num_workers=4):
    from torch.utils.data import DataLoader
    
    dataset = IRSTDataset(
        base_dir=base_dir,
        mode=mode,
        dataset_name=dataset_name,
        transform=None, # Add your transforms here for training
        return_center=True
    )

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        heatmaps = torch.stack([item['heatmap'] for item in batch])
        img_names = [item['img_name'] for item in batch]
        
        centers_list = [item['centers'] for item in batch]
        max_centers = max(len(c) for c in centers_list)
        if max_centers == 0: max_centers = 1
        
        centers_padded = []
        for centers in centers_list:
            num_centers = centers.shape[0]
            if num_centers == 0:
                pad_tensor = torch.zeros((max_centers, 2), dtype=torch.float32)
                centers_padded.append(pad_tensor)
            elif num_centers < max_centers:
                pad_tensor = centers[-1:].repeat(max_centers - num_centers, 1)
                centers_padded.append(torch.cat([centers, pad_tensor], dim=0))
            else:
                centers_padded.append(centers)
        
        centers_padded = torch.stack(centers_padded)
        
        return {
            'image': images,
            'mask': masks,
            'heatmap': heatmaps,
            'centers': centers_padded,
            'img_name': img_names
        }
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(mode == 'train')
    )
    return loader