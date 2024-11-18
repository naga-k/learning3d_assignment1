import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np

import pytorch3d
from starter.utils import get_device, get_points_renderer

def render_bridge(
        point_cloud_path="data/bridge_pointcloud.npz",
        image_size=256,
        background_color=(1, 1, 1),
        device=None,

):
    
    if device is None:
        device = get_device()
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/bridge_render.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()
    image = render_bridge(image_size=args.image_size)

    plt.imsave(args.output, image)