import torch
import pytorch3d
from starter.utils import get_mesh_renderer, get_device
import imageio
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm

def tetrahedron_360_render(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="outputs/tetrahedron_360.gif",
):
    if device is None:
        device = get_device()

    # Define the vertices
    vertices = torch.tensor([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ], dtype=torch.float32)  # Ensure vertices are of type Float

    # Define the faces using indices of the vertices
    faces = torch.tensor([
        [0, 1, 2],  # Face 1
        [0, 1, 3],  # Face 2
        [0, 2, 3],  # Face 3
        [1, 2, 3]   # Face 4
    ], dtype=torch.int64)  # Faces can remain as Long

    vertices = vertices.unsqueeze(0)  # Add a batch dimension
    faces = faces.unsqueeze(0)  # Add a batch dimension

    texture_rgb = torch.ones_like(vertices)  # N x 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb)  # important

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
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4, azim=azim)
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
    tetrahedron_360_render(
        image_size=256,
        num_frames=100,
        duration=6,
        device=None,
        output_file="outputs/tetrahedron_360.gif",
    )