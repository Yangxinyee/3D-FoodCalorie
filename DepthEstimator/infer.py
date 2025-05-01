import os
import yaml
import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.dataset import KITTI_2015
from core.networks.model_depth_pose import Model_depth_pose
from core.networks.structures.net_utils import warp_flow
from core.visualize.visualizer import Visualizer_debug
from core.evaluation.flowlib import flow_to_image

def disp2depth(disp, min_depth=0.01, max_depth=100.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def infer_single_image(img_path, model, training_hw, save_dir='./'):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    img_resized = cv2.resize(img, (training_hw[1], training_hw[0]))
    img_t = torch.from_numpy(np.transpose(img_resized, [2,0,1])).float().cuda().unsqueeze(0) / 255.0
    disp = model.infer_depth(img_t)
    disp = np.transpose(disp[0].cpu().detach().numpy(), [1,2,0])
    disp_resized = cv2.resize(disp, (w,h))
    # depth = 1.0 / (1e-6 + disp_resized)
    _, depth = disp2depth(disp_resized)

    visualizer = Visualizer_debug(dump_dir=save_dir)
    visualizer.save_depth_img(depth, name='Depth')
    visualizer.save_disp_color_img(disp_resized, name='Colorized_Depth')
    print('Depth prediction saved in ' + save_dir)

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
    arg_parser.add_argument('--task', type=str, default='demo', help='demo or single or dict.')
    arg_parser.add_argument('--image_path', type=str, default=None, help='Set this only when task==infer or dict. Depth demo for single image.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--result_dir', type=str, default=None, help='directory for saving predictions')
    arg_parser.add_argument('--indices', type=str, default='0,5,10,15,25', help='Image indices to visualize, comma-separated, e.g., 0,5,10,15')

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
        print(attr)
        setattr(cfg_new, attr, cfg[attr])

    model = Model_depth_pose(cfg_new)

    model.cuda()
    weights = torch.load(args.pretrained_model)
    new_state_dict = {}
    for k, v in weights['model_state_dict'].items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    print('Model Loaded.')

    if args.task == 'single':
        infer_single_image(args.image_path, model, training_hw=cfg['img_hw'], save_dir=args.result_dir)
    elif args.task == 'demo':
        visualize_kitti_predictions(cfg, model, indices=indices, output_dir=args.result_dir)
    else:
        raise ValueError('Invalid task. Please use single, demo, or dict.')
