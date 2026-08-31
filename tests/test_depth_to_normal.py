import numpy as np

from tools.depth_to_normal import depth_to_normal


def test_flat_depth_has_positive_z_normal():
    normal = depth_to_normal(np.ones((8, 8), dtype=np.float32))
    np.testing.assert_allclose(normal[0], 0)
    np.testing.assert_allclose(normal[1], 0)
    np.testing.assert_allclose(normal[2], 1)


def test_normals_are_unit_length():
    depth = np.arange(64, dtype=np.float32).reshape(8, 8)
    normal = depth_to_normal(depth)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=0), 1, atol=1e-6)

