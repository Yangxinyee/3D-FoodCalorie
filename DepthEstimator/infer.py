import os
import yaml
import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from core.dataset import KITTI_2015
from core.networks.model_depth_pose import Model_depth_pose
from core.networks.structures.net_utils import warp_flow
from core.visualize.visualizer import Visualizer_debug
from core.evaluation.flowlib import flow_to_image

def load_model(path, model):
    data = torch.load(path)
    try:
        state_dict = data['model']
    except:
        state_dict = data['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load weights: {e}")
    return model

def disp2depth(disp, min_depth=0.01, max_depth=100.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_intrinsic_from_fov(width, height, fov_deg=60):
    """ Generate a pinhole camera intrinsic matrix from image size and field of view (FoV). """
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))  # assuming square pixels
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])

def get_intrinsic_from_fov(width, height, fov_deg=60):
    """ Generate a pinhole camera intrinsic matrix from image size and field of view (FoV). """
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))  # assuming square pixels
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])

def infer_single_image(img_path, model, training_hw, min_depth, max_depth, save_dir='./', fov=60):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]

    img_resized = cv2.resize(img, (training_hw[1], training_hw[0]), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(np.transpose(img_resized, [2,0,1])).float().cuda().unsqueeze(0) / 255.0

    with torch.no_grad():
        disp = model.infer_depth(img_t)
    disp = np.transpose(disp[0].cpu().numpy(), [1,2,0])  # (H, W, 1)
    disp_resized = cv2.resize(disp, (w,h), interpolation=cv2.INTER_NEAREST)

    # === Use estimated K for depth conversion ===
    K = get_intrinsic_from_fov(w, h, fov_deg=fov)
    fx = K[0, 0]
    depth = fx / (disp_resized + 1e-6)  # Use fx only (no cx/cy), assume depth ~ fx / disp

    visualizer = Visualizer_debug(dump_dir=save_dir)
    visualizer.save_depth_img(depth, name='depth_raw_pred')
    visualizer.save_disp_color_img(disp_resized, name='colorized_depth_pred')
    print(f'Depth prediction saved in {save_dir}, using fx={fx:.2f}')

def visualize_depth(depth_map_meters, min_depth=1e-3, max_depth=1.2, save_path='depth_color_pred.png'):
    """
    depth_map_meters: np.ndarray [H, W], float32, in meters
    """
    depth_clipped = np.clip(depth_map_meters, min_depth, max_depth)

    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Save image
    cv2.imwrite(save_path, depth_colored)

def infer_nutrition5k(img_path, model, training_hw, min_depth=1e-3, max_depth=1.2, save_dir='./', fov=60):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]

    img_resized = cv2.resize(img, (training_hw[1], training_hw[0]), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(np.transpose(img_resized, [2,0,1])).float() / 255.0
    img_t = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(img_t)
    img_t = img_t.unsqueeze(0).cuda()

    with torch.no_grad():
        disp = model.infer_depth(img_t)
    pred_depth = 1.0 / (disp + 1e-6)
    pred_depth = torch.clamp(pred_depth, min=min_depth, max=max_depth)
    print(torch.min(pred_depth), torch.max(pred_depth))
    print(torch.max(pred_depth) / torch.min(pred_depth))
    pred_depth = pred_depth.squeeze().cpu().numpy()

    pred_depth_resized = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_mm = np.clip(pred_depth_resized * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(save_dir, 'depth_raw_pred.png'), depth_mm)

    visualize_depth(pred_depth_resized, min_depth, max_depth, os.path.join(save_dir, 'depth_color_pred.png'))


def sliding_window_inference(image_path, model, input_size=(256, 832), orig_size=(480, 640), stride_ratio=0.5, save_patches=False, patch_save_dir="model_inputs", save_dir="model_outputs"):
    """
    Args:
        image_path (str): path to input RGB image
        model (torch.nn.Module): depth prediction model
        input_size (tuple): (H, W) input size expected by the model
        orig_size (tuple): original image size (H, W), default is (480, 640)
        stride_ratio (float): stride ratio w.r.t input height (e.g. 0.5 means 50% overlap)
        save_patches (bool): whether to save input patches for inspection
        patch_save_dir (str): directory to save patches

    Returns:
        depth_map_cropped (np.ndarray): predicted depth map of size (480, 640)
    """

    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    img_tensor = transform(img)  # [3, H, W]

    orig_H, orig_W = orig_size
    in_H, in_W = input_size

    # === Padding image to match model input size along width ===
    pad_h = max(0, in_H - orig_H)
    pad_w = max(0, in_W - orig_W)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom))  # [3, H_pad, W_pad]
    img_padded = img_padded.unsqueeze(0).cuda()  # [1, 3, H_pad, W_pad]

    _, _, H_pad, W_pad = img_padded.shape
    stride_h = int(in_H * stride_ratio)

    disp_full = torch.zeros((1, 1, H_pad, W_pad)).cuda()
    count_map = torch.zeros((1, 1, H_pad, W_pad)).cuda()

    if save_patches:
        os.makedirs(patch_save_dir, exist_ok=True)

    idx = 0
    for y in range(0, H_pad - in_H + 1, stride_h):
        patch = img_padded[:, :, y:y+in_H, :]  # [1, 3, 256, 832]

        if save_patches:
            patch_img = patch[0].permute(1, 2, 0).cpu().numpy()
            patch_img = (patch_img * 255).astype(np.uint8)
            Image.fromarray(patch_img).save(os.path.join(patch_save_dir, f'patch_{idx}.jpg'))

        with torch.no_grad():
            disp_patch = model.infer_depth(patch)  # [1, 1, 256, 832]
        disp_full[:, :, y:y+in_H, :] += disp_patch
        count_map[:, :, y:y+in_H, :] += 1
        idx += 1

    # === Normalize overlapped regions ===
    disp_full /= (count_map + 1e-6)

    # === Crop back to original image region ===
    disp_cropped = disp_full[:, :, pad_top:pad_top+orig_H, pad_left:pad_left+orig_W]
    disp_map = disp_cropped.squeeze().cpu().numpy()  # [H, W]
    
    K = get_intrinsic_from_fov(orig_W, orig_H, fov_deg=60)  # same as infer_single_image()
    fx = K[0, 0]
    depth_map = fx / (disp_map + 1e-6)  # [H, W]

    visualizer = Visualizer_debug(dump_dir=save_dir)
    disp = 1.0 / (disp_map + 1e-6)
    disp = np.clip(disp, 0, np.percentile(disp, 95))  # 可选增强可视化效果

    visualizer.save_depth_img(depth_map, name='depth_raw_pred')           # 保存灰度深度图
    visualizer.save_disp_color_img(disp_map, name='colorized_depth_pred')             # 保存彩色 disparity 图
    print(f"Saved depth_raw_pred & colorized_depth_pred to {save_dir}")


def batch_infer_directory(root_dir, model, training_hw, min_depth, max_depth, sliding_window=False):
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subdir in tqdm(subdirs, desc="Batch Depth Inference"):
        rgb_path = os.path.join(subdir, "rgb.png")
        if os.path.exists(rgb_path):
            if sliding_window:
                sliding_window_inference(rgb_path, model, training_hw, save_dir=subdir)
            else:
                infer_nutrition5k(rgb_path, model, training_hw, min_depth, max_depth, save_dir=subdir)
        else:
            print(f"Skipping {subdir}: rgb.png not found.")

def visualize_kitti_predictions(cfg, model, indices, output_dir):
    """
    Visualize predictions on KITTI dataset
    
    Args:
        cfg: Configuration object
        model: Pretrained model
        indices: List of image indices to visualize [int]
        output_dir: Output directory for saving visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = Visualizer_debug(dump_dir=output_dir)
    dataset = KITTI_2015(cfg.gt_2015_dir)
    model.eval()
    
    all_combined_imgs = []
    for idx in tqdm(indices):
        if idx >= len(dataset):
            raise ValueError(f"Index {idx} exceeds dataset range")
            
        img, K, K_inv = dataset[idx]
        img = img[None,:,:,:]
        K = K[None,:,:]
        K_inv = K_inv[None,:,:]
        img_h = int(img.shape[2] / 2)
        img1, img2 = img[:,:,:img_h,:], img[:,:,img_h:,:]
        img1, img2, K, K_inv = img1.cuda(), img2.cuda(), K.cuda(), K_inv.cuda()
        
        # Model inference
        with torch.no_grad():
            flow, disp1, disp2, Rt, _, _ = model.inference(img1, img2, K, K_inv)
        
        # Get original image
        original_img = img1[0].cpu().permute(1, 2, 0).numpy() * 255
        original_img = original_img.astype(np.uint8)
        
        # Get disparity (inverse depth)
        disp = disp1[0].detach().cpu().numpy()
        disp = disp.transpose(1, 2, 0).squeeze()
        
        # Save original image
        cv2.imwrite(os.path.join(output_dir, f'original_{idx}.png'), original_img)
        
        # Save colorized disparity using the visualizer
        visualizer.save_disp_color_img(disp, name=f'depth_{idx}')
        
        # Get flow visualization
        flow_np = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
        flow_viz = flow_to_image(flow_np)
        cv2.imwrite(os.path.join(output_dir, f'flow_{idx}.png'), flow_viz)
        
        # Generate occlusion mask
        with torch.no_grad():
            flow_rev, _, _, _, _, _ = model.inference(img2, img1, K, K_inv)
            flow_warped = warp_flow(flow_rev.cpu(), flow.cpu()).cuda()
            flow_diff = torch.abs(flow_warped + flow)

            # Calculate flow consistency bound 
            flow_mag = torch.sqrt(flow[0,0]**2 + flow[0,1]**2)
            consistency_bound = torch.max(0.05 * flow_mag, torch.ones_like(flow_mag) * 3.0)

            # Create mask based on consistency
            occ_mask = (torch.sqrt(flow_diff[0,0]**2 + flow_diff[0,1]**2) < consistency_bound).float()
            occ_mask_np = (occ_mask.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'occlusion_{idx}.png'), occ_mask_np)
                
        # Read colorized depth using OpenCV (as it was saved by the visualizer)
        colored_depth = cv2.imread(os.path.join(output_dir, f'depth_{idx}_depth.jpg'))
        colored_depth = cv2.resize(colored_depth, (original_img.shape[1], original_img.shape[0]))
        
        # Define spacing between rows
        spacing = 10  # pixels
        
        # Calculate combined image dimensions with spacing
        row_height = original_img.shape[0]
        combined_height = (row_height * 4) + (spacing * 3)  # 3 spaces between 4 rows
        combined_width = original_img.shape[1]
        
        # Create combined image with spacing (white background)
        combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Add images with spacing
        combined_img[0:row_height, :] = original_img
        combined_img[(row_height+spacing):(2*row_height+spacing), :] = colored_depth
        combined_img[(2*row_height+2*spacing):(3*row_height+2*spacing), :] = flow_viz
        
        # Convert occlusion mask to 3-channel for stacking
        occ_mask_3ch = cv2.cvtColor(occ_mask_np, cv2.COLOR_GRAY2BGR)
        combined_img[(3*row_height+3*spacing):, :] = occ_mask_3ch
        
        # Save individual combined visualization
        cv2.imwrite(os.path.join(output_dir, f'visualization_{idx}.png'), combined_img)
        
        # Store for grid visualization
        all_combined_imgs.append(combined_img)
        
        print(f"Visualization for index {idx} saved")
    
    # Create grid visualization with all images side by side
    if all_combined_imgs:
        # Define spacing between columns
        col_spacing = 10  # pixels
        
        # Calculate grid dimensions
        grid_height = all_combined_imgs[0].shape[0]
        grid_width = (all_combined_imgs[0].shape[1] * len(all_combined_imgs)) + (col_spacing * (len(all_combined_imgs) - 1))
        
        # Create grid image with white background
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Place all combined images side by side
        for i, img in enumerate(all_combined_imgs):
            start_col = i * (img.shape[1] + col_spacing)
            grid_img[:, start_col:(start_col + img.shape[1])] = img
        
        # Save grid visualization
        cv2.imwrite(os.path.join(output_dir, 'grid_visualization.png'), grid_img)
        print(f"Grid visualization with all {len(indices)} images saved")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="Inference script for depth estimation."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--task', type=str, default='demo', help='demo or single or batch.')
    arg_parser.add_argument('--image_path', type=str, default=None, help='Set this only when task==single. Depth demo for single image.')
    arg_parser.add_argument('--root_dir', type=str, default=None, help='Set this only when task==batch. Directory containing rgb.png images.')
    arg_parser.add_argument('--min_depth', type=float, default=0.01, help='Minimum depth value.')
    arg_parser.add_argument('--max_depth', type=float, default=100.0, help='Maximum depth value.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--result_dir', type=str, default=None, help='Set this only when task==demo. Directory for saving predictions')
    arg_parser.add_argument('--indices', type=str, default='0,5,10,15,25', help='Set this only when task==demo. Image indices to visualize, comma-separated, e.g., 0,5,10,15')

    args = arg_parser.parse_args()

    indices = [int(i) for i in args.indices.split(',')]
    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])
    cfg['model_dir'] = args.result_dir
    cfg['mode'] = 'depth'

    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)
    
    class pObject(object):
        def __init__(self):
            pass
    
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])

    model = Model_depth_pose(cfg_new)
    model.cuda()
    model = load_model(args.pretrained_model, model)
    model.eval()
    print('Model Loaded.')

    if args.task == 'single':
        infer_single_image(args.image_path, model, training_hw=cfg['img_hw'], min_depth=args.min_depth, max_depth=args.max_depth, save_dir=args.result_dir)
    elif args.task == 'demo':
        visualize_kitti_predictions(cfg_new, model, indices=indices, output_dir=args.result_dir)
    elif args.task == 'batch':
        batch_infer_directory(args.root_dir, model, training_hw=cfg['img_hw'], min_depth=args.min_depth, max_depth=args.max_depth)
    else:
        raise ValueError('Invalid task. Please use single, demo, or dict.')
