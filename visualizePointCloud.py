import argparse
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point clouds")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing point cloud")

    path = parser.parse_args().root_dir
    pcd = o3d.io.read_point_cloud(f"{path}/pointCloud.ply")

    o3d.visualization.draw_geometries([pcd])