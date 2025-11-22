import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

from dataset import get_loader
from net import NetworkWrapper
from loss import SAMIRNetLoss, SoftIoULoss_Original

# --------------------------------------------------------------------------------
# 0. 尝试导入 thop 用于计算 FLOPs (如果未安装则忽略)
# --------------------------------------------------------------------------------
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# --------------------------------------------------------------------------------
# 1. 设置日志记录 (Logging)
# --------------------------------------------------------------------------------
def setup_logging(log_file):
    """设置日志，同时输出到控制台和文件"""
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    return logger

logger = logging.getLogger('train_logger')

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAMIRNet for IRST Detection')
    
    # Model
    parser.add_argument('--model_names', type=str, nargs='+', default=['SAMIRNet_Tiny'])
    
    # Dataset
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['NUDT-SIRST'])
    parser.add_argument('--base_dir', type=str, default='./datasets')
    parser.add_argument('--img_size', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--det_weight', type=float, default=0.3)
    parser.add_argument('--seg_weight', type=float, default=0.7)
    parser.add_argument('--iou_weight', type=float, default=0.1)
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')
    
    # Save
    parser.add_argument('--save_dir', type=str, default='./log')
    parser.add_argument('--save_interval', type=int, default=50)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    # GPU
    parser.add_argument('--gpu_id', type=str, default='0')
    
    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------------
# 2. FLOPs 和 Params 计算函数
# --------------------------------------------------------------------------------
def calculate_flops_params(model, img_size, device):
    """计算并打印模型的 FLOPs 和 参数量"""
    logger.info(f"{'-'*60}")
    logger.info("Model Complexity Check:")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  - Params: {total_params / 1e6:.4f} M")

    # 计算 FLOPs
    if THOP_AVAILABLE:
        try:
            # 构造一个 dummy input (1, 3, H, W)
            # 注意：如果你的模型输入是单通道，请将 3 改为 1
            input_dummy = torch.randn(1, 3, img_size, img_size).to(device)
            flops, params = profile(model, inputs=(input_dummy,), verbose=False)
            logger.info(f"  - FLOPs : {flops / 1e9:.4f} G")
        except Exception as e:
            logger.warning(f"  - FLOPs calc failed: {e}")
    else:
        logger.warning("  - FLOPs : 'thop' library not found. Install via 'pip install thop'")
    logger.info(f"{'-'*60}\n")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, args, n_epochs):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0.0
    epoch_det_loss = 0.0
    epoch_seg_loss = 0.0
    
    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{n_epochs}] Train', leave=False)
    
    for batch in loop:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        heatmaps = batch['heatmap'].to(device)
        centers = batch['centers'].to(device)
        
        # Forward
        if args.model_names[0].startswith('SAMIRNet'):
            pred_heatmap, pred_masks, pred_iou = model(
                images, 
                gt_centers=centers,
                gt_masks=masks
            )
            
            # Calculate loss
            total_loss, loss_dict = criterion(
                pred_heatmap, pred_masks, pred_iou,
                heatmaps, masks
            )
            
            epoch_det_loss += loss_dict['det_loss']
            epoch_seg_loss += loss_dict['seg_loss']
            
        else:
            pred_masks = model(images)
            total_loss = criterion(pred_masks, masks)
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
        # 进度条显示
        postfix_dict = {'loss': f"{total_loss.item():.4f}"}
        if args.model_names[0].startswith('SAMIRNet'):
            postfix_dict['det'] = f"{loss_dict['det_loss']:.4f}"
            postfix_dict['seg'] = f"{loss_dict['seg_loss']:.4f}"
            
        loop.set_postfix(**postfix_dict)

    loop.close()
    
    avg_loss = epoch_loss / len(train_loader)
    avg_det_loss = epoch_det_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_seg_loss = epoch_seg_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    return avg_loss, avg_det_loss, avg_seg_loss


def validate(model, val_loader, device, args):
    """Validate the model"""
    model.eval()
    
    total_inter = 0
    total_union = 0
    total_pred = 0
    total_gt = 0
    
    loop = tqdm(val_loader, desc='Validate', leave=False)
    
    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            if args.model_names[0].startswith('SAMIRNet'):
                _, pred_masks, _ = model(images, gt_centers=None, gt_masks=None)
            else:
                pred_masks = model(images)
            
            pred_binary = (pred_masks > 0.5).float()
            
            inter = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - inter
            pred_sum = pred_binary.sum()
            gt_sum = masks.sum()
            
            total_inter += inter
            total_union += union
            total_pred += pred_sum
            total_gt += gt_sum
    
    loop.close()
    
    iou = total_inter / (total_union + 1e-6)
    dice = (2 * total_inter) / (total_pred + total_gt + 1e-6)
    precision = total_inter / (total_pred + 1e-6)
    recall = total_inter / (total_gt + 1e-6)
    f1 = dice 
    
    return {
        'IoU': iou.item(),
        'Dice': dice.item(),
        'Precision': precision.item(),
        'Recall': recall.item(), # This is equivalent to Pd
        'F1': f1.item()
    }


def main():
    args = parse_args()
    
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name in args.model_names:
        for dataset_name in args.dataset_names:
            
            base_save_dir = os.path.join(args.save_dir, dataset_name, model_name)
            save_dir = os.path.join(base_save_dir, run_timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            log_path = os.path.join(save_dir, f'train_log.log')
            global logger
            logger = setup_logging(log_path)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {model_name} on {dataset_name}")
            logger.info(f"Run ID: {run_timestamp}")
            logger.info(f"{'='*80}\n")
            
            # --- 修改 1: 打印所有超参数 ---
            logger.info("Hyperparameters Configuration:")
            for key, value in vars(args).items():
                logger.info(f"  {key:<20}: {value}")
            logger.info(f"{'='*80}\n")
            
            writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
            
            train_loader = get_loader(args.base_dir, 'train', dataset_name, args.batch_size, args.num_workers)
            val_loader = get_loader(args.base_dir, 'test', dataset_name, args.batch_size, args.num_workers)
            
            model = NetworkWrapper(model_name, mode='train').to(device)
            
            # --- 修改 2: 计算并打印 Params 和 FLOPs ---
            calculate_flops_params(model, args.img_size, device)
            
            if model_name.startswith('SAMIRNet'):
                criterion = SAMIRNetLoss(args.det_weight, args.seg_weight, args.iou_weight)
            else:
                criterion = SoftIoULoss_Original()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            
            start_epoch = 0
            best_iou = 0.0
            best_pd = 0.0
            
            if args.resume is not None and os.path.exists(args.resume):
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_iou = checkpoint.get('best_iou', 0.0)
                logger.info(f"Resumed from epoch {start_epoch}")
            
            for epoch in range(start_epoch, args.epochs):
                epoch_start_time = time.time()
                
                train_loss, train_det_loss, train_seg_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, epoch, device, args, args.epochs
                )
                
                val_metrics = validate(model, val_loader, device, args)
                
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start_time
                
                # --- 修改 3: 优化日志打印格式 (包含 Det Loss, Seg Loss, IoU, Pd) ---
                # Pd (Probability of Detection) 等同于 Recall
                pd_score = val_metrics['Recall']
                
                log_str = f"Epoch [{epoch+1}/{args.epochs}] Time: {epoch_time:.2f}s | LR: {current_lr:.6f}\n"
                log_str += f"  [Train] Total: {train_loss:.4f}"
                if model_name.startswith('SAMIRNet'):
                    log_str += f" | Det: {train_det_loss:.4f} | Seg: {train_seg_loss:.4f}"
                
                log_str += f"\n  [Val]   IoU: {val_metrics['IoU']:.4f} | Pd: {pd_score:.4f} | F1: {val_metrics['F1']:.4f}"
                logger.info(log_str)
                
                # Tensorboard
                writer.add_scalar('Train/Loss', train_loss, epoch)
                if model_name.startswith('SAMIRNet'):
                    writer.add_scalar('Train/Det_Loss', train_det_loss, epoch)
                    writer.add_scalar('Train/Seg_Loss', train_seg_loss, epoch)
                    
                writer.add_scalar('Val/IoU', val_metrics['IoU'], epoch)
                writer.add_scalar('Val/Pd', pd_score, epoch)
                
                # Save checkpoint
                is_best = val_metrics['IoU'] > best_iou
                if is_best:
                    best_iou = val_metrics['IoU']
                    best_pd = pd_score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                        'best_pd': best_pd
                    }, os.path.join(save_dir, 'best_model.pth'))
                    logger.info(f"  >>> Best Model Saved (IoU: {best_iou:.4f}, Pd: {best_pd:.4f})")
                
                logger.info("-" * 60)
            
            writer.close()
            logger.info(f"Training Finished. Best IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
    os.system("/usr/bin/shutdown")