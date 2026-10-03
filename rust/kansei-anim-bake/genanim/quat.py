"""Quaternion helpers on numpy arrays of (x, y, z, w), broadcasting over leading axes."""

import numpy as np


def mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def inv(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def rotate(q, v):
    """v rotated by q."""
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_matrix(m):
    """Quaternions of rotation matrices [..., 3, 3]."""
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(m.shape[0]):
        r = m[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return normalize(q).reshape(*shape, 4)


def to_matrix(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def yaw(angle):
    """Rotations by `angle` radians about +Y."""
    angle = np.asarray(angle, dtype=np.float64)
    z = np.zeros_like(angle)
    return np.stack([z, np.sin(angle / 2), z, np.cos(angle / 2)], -1)


def continuous(q):
    """Flip signs along axis 0 so consecutive quaternions stay in one hemisphere."""
    q = q.copy()
    for t in range(1, q.shape[0]):
        flip = np.sum(q[t] * q[t - 1], -1) < 0
        q[t][flip] *= -1
    return q


def slerp(a, b, t):
    """From a to b by t (broadcast), the short way."""
    d = np.sum(a * b, -1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    t = np.asarray(t, dtype=np.float64)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    wa = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    wb = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    return normalize(wa * a + wb * b)


def angle(a, b):
    """Angle between rotations, in radians."""
    d = np.abs(np.sum(a * b, -1))
    return 2 * np.arccos(np.clip(d, -1.0, 1.0))


def forward_kinematics(parents, local_t, local_r):
    """Global translations and rotations from local ones ([..., J, 3], [..., J, 4])."""
    gt, gr = np.empty_like(local_t), np.empty_like(local_r)
    for j, p in enumerate(parents):
        if p < 0:
            gt[..., j, :], gr[..., j, :] = local_t[..., j, :], local_r[..., j, :]
        else:
            gr[..., j, :] = mul(gr[..., p, :], local_r[..., j, :])
            gt[..., j, :] = gt[..., p, :] + rotate(gr[..., p, :], local_t[..., j, :])
    return gt, gr
