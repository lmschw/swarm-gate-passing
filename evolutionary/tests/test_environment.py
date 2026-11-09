import numpy as np

from swarm_gate_passing.environment import GradientSensor, render_path_map


def test_render_path_map_shape_and_range():
    grid = render_path_map("sine_curve", world_width_m=10.0, world_height_m=10.0,
                            meters_per_pixel=0.05, path_width_m=0.6, path_kwargs={"freq": 2.0})
    assert grid.dtype == np.uint8
    assert grid.shape == (200, 200)
    assert grid.max() == 255
    assert grid.min() == 0


def test_render_path_map_unknown_shape_raises():
    try:
        render_path_map("not_a_shape", 10.0, 10.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_gradient_sensor_reads_bright_on_centerline():
    grid = render_path_map("zigzag", world_width_m=10.0, world_height_m=10.0, meters_per_pixel=0.05)
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)

    # sample every column's centerline (top-of-map convention: y measured from bottom)
    from swarm_gate_passing.environment.path_maps import zigzag_path
    width_px = grid.shape[1]
    height_px = grid.shape[0]
    xs_px = np.linspace(0, width_px - 1, 20)
    ys_px = zigzag_path(xs_px, width_px, height_px)

    xs_m = xs_px / width_px * 10.0
    ys_m = 10.0 - (ys_px / height_px * 10.0)  # convert row-space to bottom-up meters

    readings = sensor.read(xs_m, ys_m, add_noise=False)
    assert np.mean(readings) > 200  # near-peak brightness on the centerline


def test_gradient_sensor_out_of_bounds_reads_zero():
    grid = np.full((50, 50), 255, dtype=np.uint8)
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    readings = sensor.read(np.array([-5.0, 50.0]), np.array([-5.0, 50.0]), add_noise=False)
    assert np.all(readings == 0.0)


def test_gradient_sensor_vectorized_matches_scalar():
    grid = render_path_map("parabola", 10.0, 10.0)
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    xs = np.array([1.0, 5.0, 8.0])
    ys = np.array([2.0, 5.0, 9.0])
    vec = sensor.read(xs, ys, add_noise=False)
    scalars = np.array([sensor.read(x, y, add_noise=False) for x, y in zip(xs, ys)])
    assert np.allclose(vec, scalars)
