import torch


def contract_x2s(
    x,
    pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    alpha=0.8
):
    return world2contracted(x, pc_range_roi=pc_range, ratio=alpha)


def contract_s2x(
    s,
    pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    alpha=0.8
):
    return contracted2world(s, pc_range_roi=pc_range, ratio=alpha)


def world2contracted(xyz_world, pc_range_roi=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], ratio=0.8, eps=1e-6):
    """
    Convert 3D world coordinates to a contracted coordinate system based on a specified ROI.

    Args:
        xyz_world (torch.Tensor): Input tensor with shape [..., 3] representing 3D world coordinates.
        pc_range_roi (list, optional): List of 6 elements defining the ROI. Default is [-54, -54, -5, 54, 54, 3].
        eps (float): Small epsilon for numerical stability to prevent division by zero.

    Returns:
        torch.Tensor: Tensor with shape [..., 3] representing coordinates in the contracted system.
    """
    xyz_min = torch.tensor(pc_range_roi[:3]).to(xyz_world).reshape([1]*len(xyz_world.shape[:-1]) + [3])
    xyz_max = torch.tensor(pc_range_roi[3:]).to(xyz_world).reshape([1]*len(xyz_world.shape[:-1]) + [3])
    t = ratio / (1 - ratio)
    xyz_scaled = (2 * (xyz_world - xyz_min) / (xyz_max - xyz_min) - 1) * t
    xyz_abs = torch.abs(xyz_scaled)
    # Add eps to denominator to prevent division by zero when xyz_abs approaches (t - 1)
    denominator = xyz_abs + 1 - t + eps
    xyz_contracted = torch.where(
        xyz_abs <= t,
        xyz_scaled,
        xyz_scaled.sign() * (1.0 + t - 1.0 / denominator)
    )
    return xyz_contracted / (t + 1) # range: [-1, 1]


def contracted2world(xyz_contracted, pc_range_roi=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], ratio=0.8, eps=1e-6):
    """
    Convert 3D contracted coordinates back to the world coordinate system based on a specified ROI.

    Args:
        xyz_contracted (torch.Tensor): Input tensor with shape [..., 3] representing 3D contracted coordinates.
        pc_range_roi (list, optional): List of 6 elements defining the ROI. Default is [-54, -54, -5, 54, 54, 3].
        eps (float): Small epsilon for numerical stability to prevent division by zero.

    Returns:
        torch.Tensor: Tensor with shape [..., 3] representing coordinates in the world system.
    """
    xyz_min = torch.tensor(pc_range_roi[:3]).to(xyz_contracted).reshape([1]*len(xyz_contracted.shape[:-1]) + [3])
    xyz_max = torch.tensor(pc_range_roi[3:]).to(xyz_contracted).reshape([1]*len(xyz_contracted.shape[:-1]) + [3])
    t = ratio / (1 - ratio)
    xyz_ = xyz_contracted * (t + 1)
    xyz_abs = torch.abs(xyz_)
    # Add eps to denominator to prevent division by zero when xyz_abs approaches (t + 1)
    denominator = t + 1 - xyz_abs + eps
    xyz_scaled = torch.where(
        xyz_abs <= t,
        xyz_,
        xyz_.sign() * (t - 1.0 + 1.0 / denominator)
    ) / t
    xyz_world = 0.5 * (xyz_scaled + 1) * (xyz_max - xyz_min) + xyz_min
    return xyz_world


def rev_contract_depth(origins, directions, depths):
    """
        origins: contracted ray origins
        directions: contracted ray directions
        depths: contracted ray depths
    """
    pts = contract_s2x(origins + directions * depths)
    unscaled_depth = torch.norm(pts - contract_s2x(origins), dim=-1, keepdim=True)
    return unscaled_depth
