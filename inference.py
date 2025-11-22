import argparse
import os
import time
import torch
import numpy as np
import cv2  # 引入OpenCV用于生成伪彩色热力图
from tqdm import tqdm
from PIL import Image

# 引入你项目中的模块
from dataset import get_loader
from net import NetworkWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SAMIRNet Inference with Heatmap")
    
    # 模型相关
    parser.add_argument("--model_name", default='SAMIRNet_Tiny', type=str, 
                        help="model_name: 'SAMIRNet_Tiny', 'DNANet', etc.")
    parser.add_argument("--pth_path", default='./log/NUDT-SIRST/SAMIRNet_Tiny/2025-11-18_20-33-44/best_model.pth', type=str, 
                        help="Path to the .pth checkpoint file")
    
    # 数据集相关
    parser.add_argument("--dataset_name", default='NUDT-SIRST', type=str, 
                        help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'")
    parser.add_argument("--base_dir", default='./datasets', type=str, 
                        help="Base directory of datasets")
    
    # 输出相关
    parser.add_argument("--save_dir", type=str, default='./results/', 
                        help="path to save inference results")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Threshold for binary mask")
    
    # 硬件
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    
    return parser.parse_args()

def save_heatmap(heatmap_tensor, save_path, img_name):
    """
    保存热力图
    heatmap_tensor: (1, H, W) value 0~1
    """
    # 1. 转为 numpy (H, W)
    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    
    # 2. 归一化到 0-255
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 3. 应用伪彩色 (COLORMAP_JET: 蓝背景->红高亮)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 4. 保存
    save_file = os.path.join(save_path, f"{img_name}_heatmap.png")
    cv2.imwrite(save_file, heatmap_color)

def inference():
    args = parse_args()

    # 1. 生成运行时间戳
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # 2. 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # 3. 加载数据
    test_loader = get_loader(
        base_dir=args.base_dir,
        mode='test',
        dataset_name=args.dataset_name,
        batch_size=1,
        num_workers=1
    )
    print(f"Loaded {len(test_loader)} images from {args.dataset_name}")

    # 4. 初始化模型
    model = NetworkWrapper(model_name=args.model_name, mode='test').to(device)

    # 5. 加载权重
    if os.path.isfile(args.pth_path):
        print(f"Loading checkpoint: {args.pth_path}")
        checkpoint = torch.load(args.pth_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.pth_path}")

    model.eval()

    # 6. 创建保存目录结构
    # 结构: results/数据集/模型/时间戳/masks
    #                          /heatmaps
    base_save_path = os.path.join(args.save_dir, args.dataset_name, args.model_name, run_timestamp)
    
    masks_save_path = os.path.join(base_save_path, 'masks')
    os.makedirs(masks_save_path, exist_ok=True)
    
    heatmaps_save_path = os.path.join(base_save_path, 'heatmaps')
    
    # 只有是 SAMIRNet 时才创建 heatmap 文件夹
    save_heatmap_flag = args.model_name.startswith('SAMIRNet')
    if save_heatmap_flag:
        os.makedirs(heatmaps_save_path, exist_ok=True)

    print(f"Results will be saved to: {base_save_path}")

    # 7. 推理循环
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing"):
            images = batch['image'].to(device)
            img_names = batch['img_name']
            
            pred_heatmap = None
            
            # 前向传播
            if args.model_name.startswith('SAMIRNet'):
                # SAMIRNet 返回 (heatmap, masks, iou)
                # 这里的 pred_heatmap 就是探测头的输出
                pred_heatmap, pred_masks, _ = model(images, gt_centers=None, gt_masks=None)
            else:
                # 其他模型只返回 mask
                pred_masks = model(images)

            # --- 保存 Mask ---
            preds = (pred_masks > args.threshold).float().cpu().numpy()
            
            for i in range(len(preds)):
                img_name = img_names[i]
                
                # 保存 Mask
                pred_mask = preds[i][0] * 255
                pred_img = Image.fromarray(pred_mask.astype(np.uint8))
                pred_img.save(os.path.join(masks_save_path, f"{img_name}.png"))
                
                # --- 保存 Heatmap (如果存在) ---
                if save_heatmap_flag and pred_heatmap is not None:
                    # 取出当前图片的 heatmap (1, H, W)
                    cur_heatmap = pred_heatmap[i] 
                    save_heatmap(cur_heatmap, heatmaps_save_path, img_name)

    print(f"\nInference Done!")
    print(f"Masks saved in: {masks_save_path}")
    if save_heatmap_flag:
        print(f"Heatmaps saved in: {heatmaps_save_path}")

if __name__ == '__main__':
    inference()