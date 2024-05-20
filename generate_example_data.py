import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def generate_random_poses(n_poses, lower_xyz, upper_xyz):
    """
    Generate random poses for n_poses.
    """
    xyz = np.random.uniform(lower_xyz, upper_xyz, (n_poses, 3))
    qs = np.random.randn(n_poses, 4)
    qs /= np.linalg.norm(qs, axis=1)[:, None]
    time = np.linspace(0, 1, n_poses)[:, None]

    poses = np.concatenate([time, xyz, qs], axis=1)

    # Write to df
    df = pd.DataFrame(poses, columns=["time", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return df


def generate_spiral_poses(min_height, max_height, num_poses, radius, turns):
    # Generate height values
    heights = np.linspace(min_height, max_height, num_poses)

    # Generate angles
    angles = np.linspace(0, 2 * np.pi * turns, num_poses)

    # Calculate x, y, z coordinates
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)
    z_coords = heights

    # Calculate orientations
    orientations = np.vstack((np.cos(angles), np.sin(angles), np.zeros(num_poses))).T

    # Convert orientations to quaternions
    z_axis = np.array([0, 0, 1])
    quaternions = [
        R.from_rotvec(
            np.cross(z_axis, orient) * np.arccos(np.dot(z_axis, orient))
        ).as_quat()
        for orient in orientations
    ]

    # Create DataFrame
    data = {
        "time": np.linspace(0, 1, num_poses),
        "x": x_coords,
        "y": y_coords,
        "z": z_coords,
        "qx": [quat[0] for quat in quaternions],
        "qy": [quat[1] for quat in quaternions],
        "qz": [quat[2] for quat in quaternions],
        "qw": [quat[3] for quat in quaternions],
    }
    df = pd.DataFrame(data)

    return df


def generate_random_distance_measurements(n_poses):
    """
    Generate random distance measurements.
    """
    distances = np.random.uniform(0.8, 1, n_poses)
    time = np.linspace(0, 1, n_poses)[:, None]

    measurements = np.concatenate([time, distances[:, None]], axis=1)

    # Write to df
    df = pd.DataFrame(measurements, columns=["time", "meas"])
    return df


def generate_K_mat():
    """
    Generate camera intrinsic matrix. This is such a dumb function.
    """
    fx = 525.0 / 2
    fy = 525.0 / 2
    cx = 319.5 / 2
    cy = 239.5 / 2

    # write to df
    cols = ["fx", "fy", "cx", "cy"]
    K_df = pd.DataFrame([[fx, fy, cx, cy]], columns=cols)
    return K_df


def main():
    min_height = 0
    max_height = 2
    num_poses = 500
    radius = 1
    turns = 5

    df = generate_spiral_poses(min_height, max_height, num_poses, radius, turns)
    df.to_csv("sample_data/acoustic_projection/example_poses.csv", index=False)

    xyz = df[["x", "y", "z"]].values
    qs = df[["qx", "qy", "qz", "qw"]].values
    Rs = R.from_quat(qs).as_matrix()
    Ts = np.concatenate([Rs, xyz[:, :, None]], axis=2)
    Ts = np.concatenate(
        [Ts, np.array([0, 0, 0, 1])[None, None, :].repeat(num_poses, axis=0)], axis=1
    )

    meshes = []
    for i in range(num_poses):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(Ts[i])
        meshes.append(coord)

    o3d.visualization.draw_geometries(meshes)

    meas_df = generate_random_distance_measurements(num_poses)
    meas_df.to_csv(
        "sample_data/acoustic_projection/example_measurements.csv", index=False
    )

    K_df = generate_K_mat()
    K_df.to_csv("sample_data/acoustic_projection/example_K.csv", index=False)


if __name__ == "__main__":
    main()
