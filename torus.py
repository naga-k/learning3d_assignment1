import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytorch3d
from starter.utils import get_device, get_points_renderer
from PIL import Image, ImageDraw
import imageio
from tqdm.auto import tqdm

def render_torus(
        image_size=256,
        num_samples=200,
        num_frames=100,
        duration=6,
        output_file="outputs/torus_render_360.gif",
        device=None
    ):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R = 3
    r = 2
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)

    azimuths = torch.linspace(0, 360, num_frames)

    renders = []
    for azim in tqdm(azimuths, total=num_frames):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=10, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"azim: {azimuths[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/torus_render.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()
    image = render_torus(image_size=args.image_size)