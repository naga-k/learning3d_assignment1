import torch
import pytorch3d
from starter.utils import get_mesh_renderer, get_device,load_cow_mesh
import imageio
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm

def retexture_cow_360_render(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="outputs/retexture_cow_360.gif",
):
    if device is None:
        device = get_device()

    vertices, faces = load_cow_mesh(path="data/cow.obj")

    vertices = vertices.unsqueeze(0)  # Add a batch dimension
    faces = faces.unsqueeze(0)  # Add a batch dimension
    
    # Extract z-coordinates
    z_coords = vertices[:, :, 2]

    # Calculate z_min and z_max
    z_min = z_coords.min()
    z_max = z_coords.max()

    # Calculate alpha for each vertex
    alpha = (z_coords - z_min) / (z_max - z_min)

    color1 = torch.tensor([0, 0, 1], dtype=torch.float32)
    color2 = torch.tensor([1, 0, 0], dtype=torch.float32)
    # Repeat color1 and color2 to match the shape of alpha
    color1 = color1.unsqueeze(0).unsqueeze(0).expand(vertices.shape[0], vertices.shape[1], -1)
    color2 = color2.unsqueeze(0).unsqueeze(0).expand(vertices.shape[0], vertices.shape[1], -1)

    alpha = alpha.unsqueeze(-1).expand(-1, -1, 3)

    print(color1.shape, color2.shape, alpha.shape)

    color = alpha * color2 + (1 - alpha) * color1
    textures = pytorch3d.renderer.TexturesVertex(color)  # important

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures,  # Correctly pass the textures object
    )

    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    azimuths = torch.linspace(0, 360, num_frames)

    renders = []
    for azim in tqdm(azimuths, total=num_frames):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
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
    retexture_cow_360_render(
        image_size=256,
        num_frames=100,
        duration=6,
        device=None,
        output_file="outputs/retexture_retexture_cow_360.gif",
    )