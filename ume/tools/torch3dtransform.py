import torch
import torch.nn.functional as F


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def r_t_to_T(r, t):
    batch_shape = r.shape[:-2]
    T = torch.zeros(*batch_shape, 4, 4, device=r.device, dtype=r.dtype)
    T[..., :3, :3] = r
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def pose9d_to_T(poses: torch.Tensor):
    r6d, t = poses[..., :6], poses[..., 6:]
    r = rotation_6d_to_matrix(r6d)
    out_poses = r_t_to_T(r, t)
    return out_poses


def T_to_pose9d(poses: torch.Tensor):
    r, t = poses[..., :3, :3], poses[..., :3, 3]
    r6d = matrix_to_rotation_6d(r)
    return torch.cat([r6d, t], dim=-1)


def T_to_rot6d_pos3d(poses: torch.Tensor):
    pose9d = T_to_pose9d(poses)
    return pose9d[..., :6], pose9d[..., 6:]


def rot6d_pos3d_to_T(rot6d, pos3d):
    pose9d = torch.cat([rot6d, pos3d], dim=-1)
    return pose9d_to_T(pose9d)
