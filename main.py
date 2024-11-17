import torch
import pytorch3d
from starter.utils import get_mesh_renderer, load_cow_mesh, get_device
import os
import imageio
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = get_device()

# Rendering your first mesh
renderer = get_mesh_renderer()

vertices, faces = load_cow_mesh(path="data/cow.obj")

vertices = vertices.unsqueeze(0)  # Add a batch dimension
faces = faces.unsqueeze(0)  # Add a batch dimension

texture_rgb = torch.ones_like(vertices)  # N x 3
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)  # important

meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,  # Correctly pass the textures object
)

cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=torch.eye(3).unsqueeze(0),
    T=torch.tensor([[0, 0, 3]]),
    fov=60,
)


lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]])

meshes = meshes.to(device)
cameras = cameras.to(device)
lights = lights.to(device)

rend = renderer(meshes, cameras=cameras, lights=lights)
image = rend[0, ..., :3].cpu().numpy()

output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.imsave(f"{output_dir}/cow_render.jpg", image)


distance = 2
elevation = 0
azimuth = 0
R,T = pytorch3d.renderer.cameras.look_at_view_transform(distance,elevation,azimuth)

def cow_360_render(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="outputs/cow_360.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    azimuths = torch.linspace(0, 360, num_frames)


    renders = []
    for azim in tqdm(azimuths, total=num_frames):
        R, T = pytorch3d.renderer.look_at_view_transform(dist = 2, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
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
        
    cow_360_render(
        image_size=256,
        num_frames=100,
        duration=6,
        device=None,
        output_file="outputs/cow_360.gif",
    )

