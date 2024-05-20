import numpy as np
import torch
from kornia.geometry.depth import depth_to_3d
import open3d as o3d
import argparse as ap
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import Optional
from einops import rearrange
from kornia import create_meshgrid

from kornia.core import Tensor, stack
from kornia.utils._compat import torch_meshgrid
from tqdm import tqdm

"""
Disclaimer, some of these functions have been taken from https://github.com/ethz-asl/virus_nerf and not implemented by me!
Any function with actually good documentation is implemented by them, the others are me.
"""

import warnings

warnings.filterwarnings("ignore")

def read_measurements(csv_file: str):
    """
    Return the time and measurements from the USS measurements csv file

    Args:
        csv_file: path to the csv file containing the measurements

    Returns:
        times: time stamps of the measurements
        measurements: measurements made by the USS
    """
    df = pd.read_csv(csv_file)
    times = df["time"].values
    measurements = df["meas"].values
    return times, measurements


def get_intrinsics(csv_file: str, ret_type=np.array, device="cpu"):
    """
    Read the intrinsic matrix from a csv file formatted as:
    fx, fy, cx, cy

    Args:
        csv_file: path to the csv file containing the intrinsic matrix
        ret_type: type of the return value

    Returns:
        K_mat: intrinsic matrix
    """
    df = pd.read_csv(csv_file)
    K_mat = np.eye(3) if ret_type == np.array else torch.eye(3).to(torch.device(device))
    K_mat[0, 0] = df["fx"].values[0]
    K_mat[1, 1] = df["fy"].values[0]
    K_mat[0, 2] = df["cx"].values[0]
    K_mat[1, 2] = df["cy"].values[0]
    return K_mat


def read_poses(csv_file: str, ret_type=np.array, device="cpu"):
    """
    Return the time and poses from the camera poses csv file

    Args:
        csv_file: path to the csv file containing the camera poses formatted as
                    time, x, y, z, qx, qy, qz, qw

    Returns:
        times: time stamps of the poses
        Ts: list of camera poses (N, 4, 4)

    """
    df = pd.read_csv(csv_file)
    times = df["time"].values
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    qxs = df["qx"].values
    qys = df["qy"].values
    qzs = df["qz"].values
    qws = df["qw"].values
    Ts = []
    assert len(qxs) == len(qys) == len(qzs) == len(qws)
    assert len(times) == len(x) == len(y) == len(z) == len(qxs)
    for ii in range(len(qxs)):
        assert np.isclose(
            np.linalg.norm([qxs[ii], qys[ii], qzs[ii], qws[ii]]), 1.0, atol=1e-3
        )
        qx, qy, qz, qw = qxs[ii], qys[ii], qzs[ii], qws[ii]
        r = R.from_quat([qx, qy, qz, qw])
        rotmat = r.as_matrix()
        T = np.eye(4) if ret_type == np.array else torch.eye(4).to(torch.device(device))
        T[:3, :3] = rotmat
        T[:3, 3] = [x[ii], y[ii], z[ii]]
        Ts.append(T)

    return times, (
        np.array(Ts)
        if ret_type == np.array
        else torch.tensor(Ts).to(torch.device(device))
    )


def visualize_poses(Ts):
    """
    Visualize the camera poses in 3D space

    Args:
        Ts: list of camera poses
    """
    coords = []
    for T in Ts:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        mesh.transform(T)
        coords.append(mesh)

    return coords


def project_single_depth(depth, T):
    """
    Function to take in a single depth measurement and convert it to a pointcloud projected from the pose T

    Args:
        depth: depth measurement
        T: pose of the camera (4,4)

    Returns:
        pointcloud: open3d.geometry.PointCloud
    """
    depth_pose_z = T[3, -1] + depth
    depth_pose = np.array([T[3, 0], T[3, 1], depth_pose_z])
    # create a pointcloud of it
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(depth_pose)
    return pointcloud


def project_single_depth_all(depths, Ts):
    """
    Project a list of depths to 3D space

    Args:
        depths: list of depths [float]
        Ts: list of poses (N, 4, 4)

    Returns:
        pointclouds: list of o3d pointclouds to be visualized later
    """
    depths_projs = np.eye(4)
    depths_projs = np.tile(depths_projs, (len(depths), 1, 1))
    depths_projs[:, 2, -1] = depths

    assert len(depths_projs) == len(Ts)
    # vectorizeeeeed, ty einstein
    depths_projs = np.einsum("ijk,ikl->ijl", Ts, depths_projs)

    depth_poses = depths_projs[:, :3, -1]

    # create a pointcloud of it
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(depth_poses)
    return pointcloud


def find_closest_indices(list1, list2):
    """
    ugh
    """
    # Determine the shorter and longer list
    if len(list1) < len(list2):
        shorter_list, longer_list = list1, list2
    else:
        shorter_list, longer_list = list2, list1

    # Convert lists to numpy arrays for efficient computation
    shorter_times = np.array(shorter_list)
    longer_times = np.array(longer_list)

    indices = []
    for time in shorter_times:
        # Calculate the absolute differences between the current time and all times in the longer list
        differences = np.abs(longer_times - time)
        # Find the index of the minimum difference
        closest_index = np.argmin(differences)
        indices.append(closest_index)

    return indices


def AoV2pixel(aov_sensor: list, camera_angle_of_view: list, H, W):
    """
    Convert the angle of view to width in pixels
    Args:
        aov_sensor: angle of view of sensor in width and hight; list
    Returns:
        num_pixels: width in pixels; int
    """
    img_wh = np.array([W, H])
    aov_sensor = np.array(aov_sensor)
    aov_camera = camera_angle_of_view

    num_pixels = img_wh * aov_sensor / aov_camera
    return np.round(num_pixels).astype(int)


def createMask(angle_of_view, camera_angle_of_view, H, W) -> torch.Tensor:
    """
    Create mask for ToF sensor.
    Returns:
        mask: mask for ToF sensor; tensor of shape (H*W,)
    """
    # define USS opening angle
    pix_wh = AoV2pixel(
        aov_sensor=angle_of_view, camera_angle_of_view=camera_angle_of_view, H=H, W=W
    )  # (2,)
    pix_wh = (pix_wh / 2.0).astype(np.int32)  # convert diameter to radius

    # create mask
    m1, m2 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    m1 = m1 - H / 2
    m2 = m2 - W / 2
    mask = (m1 / pix_wh[1]) ** 2 + (m2 / pix_wh[0]) ** 2 < 1  # (H, W), ellipse
    return mask  # (H*W,)


def project_view_frustum_all(depths, Ts, aov_sensor, aov_camera):
    num_points = createMask(aov_sensor, aov_camera, H=240, W=320).sum()

    points = np.eye(4)
    points = np.tile(points, (num_points, 1, 1))
    pcs = []
    for ii in range(len(depths)):
        points_ = generate_point_cloud(
            aov_sensor[0], aov_sensor[1], depths[ii], num_points
        )
        points__ = np.eye(4)
        points__[:3, -1] = points_[ii]
        points[ii] = np.dot(Ts[ii], points__)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3, -1])
        pcs.append(point_cloud)
        break
    return pcs


def my_generate_pc(depths, Ts, aov_sensor, aov_camera, H, W, K):
    mask = createMask(aov_sensor, aov_camera, H, W)
    pointclouds = []

    assert len(depths) == len(Ts)
    for ii, depth in enumerate(depths):
        depth_img = np.zeros((H, W))
        depth_img[mask] = depth

        depth_img = torch.tensor(depth_img).unsqueeze(0).unsqueeze(0).float()
        depth_img = depth_img.to(torch.device("cuda"))

        npyImageplaneX = (
            np.linspace((-0.5 * int(W)) + 0.5, (0.5 * int(W)) - 0.5, int(W))
            .reshape(1, int(W))
            .repeat(int(H), 0)
            .astype(np.float32)[:, :, None]
        )
        npyImageplaneY = (
            np.linspace((-0.5 * int(H)) + 0.5, (0.5 * int(H)) - 0.5, int(H))
            .reshape(int(H), 1)
            .repeat(int(W), 1)
            .astype(np.float32)[:, :, None]
        )
        npyImageplaneZ = np.full([int(H), int(W), 1], K[0, 0], np.float32)
        npyImageplane = np.concatenate(
            [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2
        )

        npyDepth = (
            depth_img.squeeze().cpu().numpy()
            / np.linalg.norm(npyImageplane, 2, 2)
            * K[0, 0]
        )
        K_torch = torch.tensor(K).unsqueeze(0).float().to(torch.device("cuda"))

        points = depth_to_3d(
            torch.tensor(npyDepth)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(torch.device("cuda")),
            K_torch,
        )
        points = points.squeeze().permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

        T_points = np.eye(4)
        T_points = np.tile(T_points, (len(points), 1, 1))
        T_points[:, :3, -1] = points
        T_points = torch.tensor(T_points).float().to(torch.device("cuda"))
        Ts_ = torch.tensor(Ts[ii]).float().to(torch.device("cuda")).unsqueeze(0)
        T_points_transformed = torch.matmul(Ts_, T_points)
        new_points = T_points_transformed.squeeze().cpu().numpy()[:, :3, -1]
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(new_points)
        pointclouds.append(pointcloud)

    return pointclouds


def generate_point_cloud(aov_horizontal, aov_vertical, depth, num_points_per_axis):
    """
    Generate a point cloud for a given view frustum with specified AoV and depth.

    Parameters:
        aov_horizontal (float): Horizontal angle of view in degrees.
        aov_vertical (float): Vertical angle of view in degrees.
        depth (float): Depth to project the frustum.
        num_points_per_axis (int): Number of points along each axis (azimuth and elevation).

    Returns:
        open3d.geometry.PointCloud: Generated point cloud.
    """
    # Convert angles from degrees to radians
    aov_horizontal_rad = np.deg2rad(aov_horizontal)
    aov_vertical_rad = np.deg2rad(aov_vertical)

    # Calculate the half-angles
    half_aov_horizontal = aov_horizontal_rad / 2.0
    half_aov_vertical = aov_vertical_rad / 2.0

    # Generate a set range of azimuth and elevation angles
    azimuth_range = np.linspace(
        -half_aov_horizontal, half_aov_horizontal, num_points_per_axis
    )
    elevation_range = np.linspace(
        -half_aov_vertical, half_aov_vertical, num_points_per_axis
    )

    # Create a mesh grid of azimuth and elevation angles
    azimuth, elevation = np.meshgrid(azimuth_range, elevation_range)

    # Flatten the mesh grid
    azimuth = azimuth.flatten()
    elevation = elevation.flatten()

    # Spherical to Cartesian conversion
    x = depth * np.cos(elevation) * np.sin(azimuth)
    y = depth * np.sin(elevation)
    z = depth * np.cos(elevation) * np.cos(azimuth)

    points = np.stack([x, y, z], axis=1)

    return points


def visualize_rays(rays_o, rays_d):
    ray_directions = (
        rays_d.cpu().numpy() / np.linalg.norm(rays_d.cpu().numpy(), axis=1)[:, None]
    )

    line_sets = []

    for i in range(rays_o.shape[0]):
        o = rays_o[i].cpu().numpy()
        d = ray_directions[i]

        points = np.vstack([o, o + 1.5 * d])
        lines = [[0, 1]]
        colors = [[0, 0, 0]]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)

        line_sets.append(line_set)
    return line_sets


def readDirections(csv_file: str):
    """
    Read camera intrinsics from the dataset.
    Args:
        dataset_dir: path to dataset directory; str
        data_dir: path to data directory; str
        cam_ids: list of camera ids; list of str
    Returns:
        img_wh: tuple of image width and height
        K_dict: camera intrinsic matrix dictionary; dict oftensor of shape (3, 3)
        directions_dict: ray directions dictionary; dict of tensor of shape (H*W, 3)
    """
    K = torch.tensor(get_intrinsics(csv_file)).to(torch.device("cuda"))
    H, W = 240, 320
    directions = get_ray_directions(H, W, K).to(torch.device("cuda"))  # (H*W, 3)
    B = 128
    # Sample only B rays
    idxs = torch.randperm(H * W)[:B]
    directions = directions[idxs]
    return directions


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a coordinate grid for an image. From ViRUS-Nerf (https://github.com/ethz-asl/virus_nerf)

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    # TODO: torchscript doesn't like `torch_version_ge`
    # if torch_version_ge(1, 13, 0):
    #     x, y = torch_meshgrid([xs, ys], indexing="xy")
    #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
    # TODO: remove after we drop support of old versions
    base_grid: Tensor = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def get_ray_directions(
    H, W, K, device="cpu", random=False, return_uv=False, flatten=True
):
    """
    Get ray directions for all pixels in camera coordinate [right down front]. From ViRUS NeRF (https://github.com/ethz-asl/virus_nerf)
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
    u, v = grid.to(torch.device("cuda")).unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = torch.stack(
            [
                (u - cx + torch.rand_like(u)) / fx,
                (v - cy + torch.rand_like(v)) / fy,
                torch.ones_like(u),
            ],
            -1,
        )
    else:  # pass by the center
        directions = torch.stack(
            [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1
        )
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image. From ViRUS NeRF (https://github.com/ethz-asl/virus_nerf)
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    # add a row of [0,0,0,1] to c2w
    if c2w.shape[0] == 4:
        c2w = c2w[:3]
    if c2w.ndim == 2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = (
            directions @ torch.tensor(c2w).to(torch.device("cuda"))[:, :3].float().T
        )
    else:
        rays_d = rearrange(directions, "n c -> n 1 c") @ rearrange(
            c2w[..., :3], "n a b -> n b a"
        )
        rays_d = rearrange(rays_d, "n 1 c -> n c")

    # The origin of all rays is the camera origin in world coordinate
    rays_o = torch.tensor(c2w).to(torch.device("cuda"))[..., 3].expand_as(rays_d)

    if rays_d.shape[1] == 4:
        rays_d = rays_d[:, :3]
        rays_o = rays_o[:, :3]

    return rays_o, rays_d


def main():
    print("Script for projecting the view frustum of the USS into 3D space")

    parser = ap.ArgumentParser(
        description="Project the view frustum of the USS into 3D space"
    )
    parser.add_argument(
        "--poses_csv",
        type=str,
        help="Path to the CSV file containing the poses of the left camera",
    )
    parser.add_argument(
        "--measurements_csv",
        type=str,
        help="Path to the csv containing the measurements made by the left USS",
    )
    parser.add_argument(
        "--k_file",
        type=str,
        help="Path to the csv file containing the camera intrinsics",
    )

    args = parser.parse_args()

    H, W = 240, 320
    aov_sensor = [35, 25]
    aov_camera = [60, 40]
    freq = 10

    meas_times, measurements = read_measurements(args.measurements_csv)
    pose_times, poses = read_poses(args.poses_csv)

    left_meshes = visualize_poses(poses[::freq])
    left_pointclouds = my_generate_pc(
        measurements[::freq],
        poses[::freq],
        aov_sensor=aov_sensor,
        aov_camera=aov_camera,
        H=H,
        W=W,
        K=get_intrinsics(args.k_file),
    )

    left_rays_o, left_rays_d = [], []
    left_line_sets = []
    for ii in tqdm(range(len(meas_times[::freq]))):
        rays_o, rays_d = get_rays(readDirections(args.k_file), poses[::freq][ii])
        left_rays_o.append(rays_o)
        left_rays_d.append(rays_d)
        left_line_set = visualize_rays(rays_o, rays_d)
        left_line_sets.append(left_line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    view_control = vis.get_view_control()

    # Set the view parameters to look from the top corner of the cube
    lookat = [0.5, 0.5, 0.5]  # Center of the cube
    up = [0, 0, 1]            # Up direction
    front = [-1, -1, -1]      # Direction from the top corner to the center

    view_control.set_lookat(lookat)
    view_control.set_up(up)
    view_control.set_front(front)

    for ii in range(len(left_pointclouds)):
        # Set the view parameters to look from the top corner of the cube
        lookat = [0.5, 0.5, 0.5]  # Center of the cube
        up = [0, 0, 1]            # Up direction
        front = [-1, -1, -1]      # Direction from the top corner to the center

        view_control.set_lookat(lookat)
        view_control.set_up(up)
        view_control.set_front(front)

        # Render the visualizer
        vis.add_geometry(left_pointclouds[ii])
        vis.add_geometry(left_meshes[ii])
        for jj in range(len(left_line_sets[ii])):
            vis.add_geometry(left_line_sets[ii][jj])
        vis.poll_events()
        vis.update_renderer()

    vis.run()


if __name__ == "__main__":
    main()
