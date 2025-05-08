import os
import numpy as np
import argparse
import open3d as o3d
from PIL import Image
from tqdm import tqdm

# === Estimate Intrinsic Matrix ===
def get_intrinsic_from_fov(width, height, fov_deg=60):
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# === Generate point cloud from one dish directory ===
def generate_point_cloud(path):
    rgb_path = os.path.join(path, "rgb.png")
    depth_path = os.path.join(path, "depth_raw_pred.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"[Warning] Missing rgb or depth file in {path}, skipping.")
        return

    rgb = np.array(Image.open(rgb_path))
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32) / 10000.0  # To meters

    H, W = depth_raw.shape
    K = get_intrinsic_from_fov(W, H, fov_deg=60)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth_raw / fx
    Y = (v - cy) * depth_raw / fy
    Z = depth_raw

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    # Depth filtering (remove background / noise)
    mask = (Z.flatten() > 0.01) & (Z.flatten() < 1.2)
    points, colors = points[mask], colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(os.path.join(path, "pointCloud.ply"), pcd)

# === Batch process for all subdirectories ===
def batch_generate_point_clouds(root_dir):
    dish_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]
    for dish_dir in tqdm(dish_dirs, desc="Generating point clouds"):
        generate_point_cloud(dish_dir)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate point clouds from Nutrition5k RGB-D data")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing dish folders")
    args = parser.parse_args()

    batch_generate_point_clouds(args.root_dir)
    
