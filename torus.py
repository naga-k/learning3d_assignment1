import argparse
import matplotlib.pyplot as plt
import mcubes
import torch
import numpy as np
import pytorch3d
from starter.utils import get_device, get_mesh_renderer, get_points_renderer
from PIL import Image, ImageDraw
import imageio
from tqdm.auto import tqdm

def render_torus_parametric(
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


# def render_torus_implicit(
#         image_size=256,
#         voxel_size=64,
#         num_frames=100,
#         duration=6,
#         output_file="outputs/torus_render_360.gif",
#         device=None
#     ):
#     """
#     Renders a sphere using an implicit function and uses a marching cubes algorithm to extract the surface.
#     """
    
#     if device is None:
#         device = get_device()

#     R = 3
#     r = 2
#     phi = torch.linspace(0, 2 * np.pi, voxel_size)
#     theta = torch.linspace(0, 2*np.pi, voxel_size)
#     # Densely sample phi and theta on a grid
#     Phi, Theta = torch.meshgrid(phi, theta)

#     x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
#     y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
#     z = r * torch.sin(Theta)
    
#     X, Y, Z = torch.meshgrid(x, y, z)

#     print("Shapes of X, Y, Z:", X.shape, Y.shape, Z.shape)
    

#     # Implicit function for torus: ((x² + y² + z²) + R² - r²)² = 4R²(x² + y²)
#     voxels = ((X**2 + Y**2 + Z**2 + R**2 - r**2)**2 - 4*(R**2)*(X**2 + Y**2)).cpu().numpy()

#     # Extract surface mesh using marching cubes
#     vertices, faces = mcubes.marching_cubes(voxels, 0)
    
#     # Normalize vertex coordinates to [-1, 1]
#     vertices = vertices / voxel_size * 2 - 1


#     # Convert to tensors
#     vertices = torch.tensor(vertices).float()
#     faces = torch.tensor(faces.astype(int))

#     print("Shapes of vertices, faces:", vertices.shape, faces.shape)

#     # Create colors based on normalized vertex positions
#     points = vertices.unsqueeze(0)  # Add batch dimension
#     color = (points - points.min()) / (points.max() - points.min())
    
#     # Create mesh with colored vertices
#     mesh = pytorch3d.structures.Meshes(
#         verts=[vertices],
#         faces=[faces],
#         textures=pytorch3d.renderer.TexturesVertex(color),
#     ).to(device)

#     # Set up renderer
#     renderer = get_mesh_renderer(image_size=image_size, device=device)
    
#     # Render 360 view
#     azimuths = torch.linspace(0, 360, num_frames)
#     renders = []
    
#     for azim in tqdm(azimuths):
#         R, T = pytorch3d.renderer.look_at_view_transform(dist=10, azim=azim)
#         cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
#         rend = renderer(mesh, cameras=cameras)
#         rend = rend[0, ..., :3].cpu().numpy()
#         renders.append(rend)

#     # Create gif
#     images = []
#     for i, r in enumerate(renders):
#         image = Image.fromarray((r * 255).astype(np.uint8))
#         draw = ImageDraw.Draw(image)
#         draw.text((20, 20), f"azim: {azimuths[i]:.2f}", fill=(255, 0, 0))
#         images.append(np.array(image))
    
#     imageio.mimsave(output_file, images, fps=(num_frames / duration))

def render_torus_implicit(
        image_size=256,
        voxel_size=64,
        num_frames=100,
        duration=6,
        output_file="outputs/torus_render_360.gif",
        device=None
    ):
    """
    Renders a torus using an implicit function and uses a marching cubes algorithm to extract the surface.
    """
    
    if device is None:
        device = get_device()

    R = 1.0  # Distance from the center of the tube to the center of the torus
    r = 0.4  # Radius of the tube

    # Create a 3D grid
    x = torch.linspace(-2, 2, voxel_size)
    y = torch.linspace(-2, 2, voxel_size)
    z = torch.linspace(-2, 2, voxel_size)
    X, Y, Z = torch.meshgrid(x, y, z)

    print("Shapes of X, Y, Z:", X.shape, Y.shape, Z.shape)

    # Implicit function for torus: ((x² + y² + z² + R² - r²)² - 4R²(x² + y²)) = 0
    voxels = ((X**2 + Y**2 + Z**2 + R**2 - r**2)**2 - 4*(R**2)*(X**2 + Y**2)).cpu().numpy()

    print("Shape of voxels:", voxels.shape)

    # Extract surface mesh using marching cubes
    vertices, faces = mcubes.marching_cubes(voxels, 0)
    
    # Normalize vertex coordinates to [-1, 1]
    vertices = vertices / (voxel_size - 1) * 4 - 2

    # Convert to tensors
    vertices = torch.tensor(vertices).float().to(device)
    faces = torch.tensor(faces.astype(int)).to(device)

    # Create colors based on normalized vertex positions
    points = vertices.unsqueeze(0)  # Add batch dimension
    color = (points - points.min()) / (points.max() - points.min())
    
    # Create mesh with colored vertices
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(color),
    ).to(device)

    # Set up renderer
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
    # Render 360 view
    azimuths = torch.linspace(0, 360, num_frames)
    renders = []
    
    for azim in tqdm(azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=30, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)

    # Create gif
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"azim: {azimuths[i].item():.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    
    imageio.mimsave(output_file, images, fps=(num_frames / duration))    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="parametric",
        choices=["parametric", "implicit"],
    )
    parser.add_argument("--output", type=str, default="outputs/torus_render.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()

    if args.render == "parametric":
        render_torus_parametric(image_size=args.image_size)
    elif args.render == "implicit":
        render_torus_implicit(image_size=args.image_size)
    else:
        raise ValueError("Invalid rendering method")
