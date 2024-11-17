import torch
import pytorch3d
from starter.utils import get_mesh_renderer, get_device
import imageio
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm

def cube_360_render(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="outputs/cube_360.gif",
):
    if device is None:
        device = get_device()

    # Define the vertices of the cube
    vertices = torch.tensor([
        [-1, -1, -1],  # Vertex 0
        [ 1, -1, -1],  # Vertex 1
        [ 1,  1, -1],  # Vertex 2
        [-1,  1, -1],  # Vertex 3
        [-1, -1,  1],  # Vertex 4
        [ 1, -1,  1],  # Vertex 5
        [ 1,  1,  1],  # Vertex 6
        [-1,  1,  1],  # Vertex 7
    ], dtype=torch.float32)

    # Define the faces of the cube using vertex indices
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4],  # Left face
    ], dtype=torch.int64)

    vertices = vertices.unsqueeze(0)  # Add a batch dimension
    faces = faces.unsqueeze(0)  # Add a batch dimension

        # Define RGB colors for each vertex
    vertex_colors = torch.tensor([
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0, 1],    # Magenta
        [0, 1, 1],    # Cyan
        [0.5, 0.5, 0.5],  # Gray
        [1, 1, 1],    # White
    ], dtype=torch.float32)

    # Expand dimensions to match the batch size
    vertex_colors = vertex_colors.unsqueeze(0)  # Shape: (1, 8, 3)

    textures = pytorch3d.renderer.TexturesVertex(vertex_colors)  # important

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
    cube_360_render(
        image_size=256,
        num_frames=100,
        duration=6,
        device=None,
        output_file="outputs/cube_360.gif",
    )