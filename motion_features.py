"""Build time-step differences for motion models from existing landmark recordings."""
import numpy as np


def model_inputs(rows, steps):
    poses = np.asarray(rows, dtype=np.float32).reshape(len(rows), steps, 2, 67)
    # Only compare a hand across frames when it was tracked in both.
    valid = (poses[:, 1:, :, -1:] > 0.5) & (poses[:, :-1, :, -1:] > 0.5)
    delta = np.diff(poses[..., :-1], axis=1) * valid
    return np.concatenate((poses.reshape(len(rows), -1), delta.reshape(len(rows), -1)), axis=1)
