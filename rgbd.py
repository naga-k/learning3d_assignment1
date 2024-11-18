import pytorch3d.transforms
from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image, get_device, get_points_renderer
import numpy as np
import torch
import pytorch3d
import imageio
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def render_point_360(
        points = None,
        features = None,
        image_size=256,
        background_color=(1, 1, 1),
        device=None,
        num_frames = 100,
        duration = 10,
        output_file = "output/point_cloud.gif",
):
    
    if device is None:
        device = get_device()
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    points = points.to(device).unsqueeze(0)
    features = features.to(device).unsqueeze(0)


    rotation_transform = pytorch3d.transforms.RotateAxisAngle(angle=180, axis="X", degrees=True).to(device)
    points = rotation_transform.transform_points(points)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features)

    azimuths = torch.linspace(0, 360, num_frames)

    renders = []
    for azim in tqdm(azimuths, total=num_frames):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=6, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
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

    
    num_frames = 100
    duration = 10

    rgbd = load_rgbd_data()

    rgb_1, mask_1, depth_1, camera_1 = torch.Tensor(rgbd['rgb1']), torch.Tensor(rgbd['mask1']), torch.Tensor(rgbd['depth1']), rgbd['cameras1']
    rgb_2, mask_2, depth_2, camera_2 = torch.Tensor(rgbd['rgb2']), torch.Tensor(rgbd['mask2']), torch.Tensor(rgbd['depth2']), rgbd['cameras2']

    points_1, rgb_1 = unproject_depth_image(rgb_1, mask_1, depth_1, camera_1)
    points_2, rgb_2 = unproject_depth_image(rgb_2, mask_2, depth_2, camera_2)

    render_point_360(points_1, rgb_1, output_file="outputs/point_cloud_1.gif")
    render_point_360(points_2, rgb_2, output_file="outputs/point_cloud_2.gif")


    #union of points
    points_1  = points_1.to(get_device())
    points_2  = points_2.to(get_device())
    points_concat = torch.cat([points_1, points_2])

    rgb_1 = rgb_1.to(get_device())
    rgb_2 = rgb_2.to(get_device())
    rgb_concat = torch.cat([rgb_1, rgb_2])

    render_point_360(points_concat, rgb_concat, output_file="outputs/point_cloud_concat.gif")



