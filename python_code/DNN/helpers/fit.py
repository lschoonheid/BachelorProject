import math
import numpy as np
from typing import Any, Callable, Iterable
from scipy.optimize import curve_fit, minimize
from numpy.typing import NDArray


def poly_fit(X, Y, Z, deg=3, n=50, crop: float = 2):
    """Axis always `z`."""
    z_range = (min(Z), max(Z))
    z_span = z_range[1] - z_range[0]
    z_middle = np.mean(z_range)
    zlim = (z_middle - (z_span / 2 * crop), z_middle + (z_span / 2 * crop))

    # calculate polynomial
    zx = np.polyfit(Z, X, deg)
    fx = np.poly1d(zx)

    zy = np.polyfit(Z, Y, deg)
    fy = np.poly1d(zy)

    # calculate new x's and y's
    z_new = np.linspace(*zlim, n)
    x_new = fx(z_new)
    y_new = fy(z_new)
    return x_new, y_new, z_new


def define_circle(p1, p2, p3) -> tuple[tuple[float, float], float]:
    """Define a circle from three points. Returns (center_x, center_y), radius."""
    x = complex(p1[0], p1[1])  # type: ignore
    y = complex(p2[0], p2[1])  # type: ignore
    z = complex(p3[0], p3[1])  # type: ignore

    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    r = abs(c + x)
    c1, c2 = (-c.real, -c.imag)
    return ((c1, c2), r)


def validator_circle(
    x: float, y: float, center: tuple[float, float], r: float, abs_tolerance=0, rel_tolerance=0, **kwargs
) -> dict[str, bool | dict]:
    """Validate a point against a circle."""
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)  # distance from center of given circle
    diff = abs(dist - r)  # difference between distance and radius
    validated = diff <= abs_tolerance or diff <= rel_tolerance * r**2

    if kwargs.get("verbose") == True:
        print("on circle:", validated, "diff", diff, r, dist)

    return {
        "validated": validated,
        "diff": diff,
    }


def validator_line_xy(
    x: float,
    y: float,
    slope: float,
    intercept: float,
    abs_tolerance: float = 0,
    rel_tolerance: float = 0,
) -> dict[str, bool | float | dict]:
    """Validate a point against a line."""
    diff = abs(y - (slope * x + intercept))
    validated = diff <= abs_tolerance or diff <= rel_tolerance * abs(y)

    return {
        "validated": validated,
        "diff": diff,
    }


def _input_to_points_xy(
    *args: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    | tuple[tuple[float, float, float], tuple[float, float, float]]
    | tuple[Iterable, Iterable]
    | tuple[Iterable, Iterable, Iterable]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # Input [(x1, y1), (x2, y2), (x3, y3)]
    if len(args) == 3:
        p1, p2, p3 = args
    # Input [x1, x2, x3], [y1, y2, y3]
    elif len(args) == 2:
        X_arr, Y_arr = args
        assert (
            len(X_arr) == len(Y_arr) == 3
        ), "Input must be a tuple of three points or a tuple of two arrays of points."
        p1, p2, p3 = (X_arr[0], Y_arr[0]), (X_arr[1], Y_arr[1]), (X_arr[2], Y_arr[2])  # type: ignore
    else:
        raise ValueError("Input must be a tuple of three points or a tuple of two arrays of points.")
    return p1, p2, p3  # type: ignore


def get_validator_xy(
    *args: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    | tuple[tuple[float, float, float], tuple[float, float, float]]
    | tuple[Iterable, Iterable]
    | tuple[Iterable, Iterable, Iterable],
    thr_collinear=1.0e-6,
    **kwargs
):
    """Get validation function for a track in x, y plane."""
    p1, p2, p3 = _input_to_points_xy(*args)

    # Check if points are collinear
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])  # type: ignore
    if abs(det) < thr_collinear:
        # Since collinear, only need to check two points
        # TODO: expand to 3D
        slope, intercept = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)

        if kwargs.get("verbose") == True:
            print("Validator is line")
        return lambda x, y: validator_line_xy(x, y, slope, intercept, **kwargs)
    # Circle
    else:
        center, r = define_circle(p1, p2, p3)

        if kwargs.get("verbose") == True:
            print("Validator is circle")
        return lambda x, y: validator_circle(x, y, center, r, **kwargs)


def get_f(x: float, y: float, z: float, center: tuple[float, float], r: float, phi_0: float, z_0: float):
    """## Deprecated.
    Get frequency f of helix passing through point (x, y, z) and (0, 0, z_0) with parameters `center`, `r`, `phi_0` and `z_0`.
    """

    assert z != z_0, "z and z_0 must be different"

    center_x, center_y = center

    return (math.atan2(y - center_y, x - center_x) - phi_0) / (z - z_0)


def get_f_all(
    x_arr: Iterable[float],
    y_arr: Iterable[float],
    z_arr: Iterable[float],
    center: tuple[float, float],
    r: float,
    phi_0: float,
    z_0: float,
    single=True,
):
    """## Deprecated"""
    f_cand = []
    for x, y, z in zip(x_arr, y_arr, z_arr):
        if z == z_0:
            continue

        f_cand.append(get_f(x, y, z, center, r, phi_0, z_0))
    f = sum(f_cand) / len(f_cand)
    return f


def are_collinear(point1, point2, point3, threshold=1.0e-6):
    assert len(point1) == len(point2) == len(point3) == 3, "Points must be in 3D space."

    # Create vectors from point1 to point2 and from point1 to point3
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)

    # Compute the cross product of the vectors
    cross_product = np.cross(vector1, vector2)

    # Check if the cross product is the zero vector
    return np.linalg.norm(cross_product) < threshold


def line_equation(p1: tuple[float, float, float], p2: tuple[float, float, float]):
    """Get line equation from two points in 3D space.
    Returns
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Define line equation intercept + z * slope
    x = lambda z: x1 + (z - z1) * ((x2 - x1) / (z2 - z1))
    y = lambda z: y1 + (z - z1) * ((y2 - y1) / (z2 - z1))

    return x, y


def validator_line(
    x: float,
    y: float,
    z: float,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    abs_tolerance_line: float = 0,
    rel_tolerance: float = 0,
    **kwargs
) -> dict[str, bool | dict | Callable]:
    """Validate a point against a line in 3D space passing through `p1` and `p2`."""
    # Get equation in 3D space passing through p1 and p2
    x_eq, y_eq = line_equation(p1, p2)

    distance = lambda z: np.sqrt((x_eq(z) - x) ** 2 + (y_eq(z) - y) ** 2 + (z - z) ** 2)
    result = minimize(distance, x0=z)
    min_dist = result.fun
    z_min = result.x[0]

    validated = min_dist <= abs_tolerance_line  # ignores rel tolerance

    return {
        "validated": validated,
        "diff": min_dist,
        "z_min": z_min,
        "eq_yz": y_eq,
        "eq_xz": x_eq,
    }


def trig(x, A, B, f, phi_0, z_0, intercept):
    if A == 0:
        return B * np.cos(f * (x + z_0) + phi_0) + intercept
    if B == 0:
        return A * np.sin(f * (x + z_0) + phi_0) + intercept

    _sin = np.sin(f * (x + z_0) + phi_0)
    _cos = np.cos(f * (x + z_0) + phi_0)
    return A * _sin + B * _cos + intercept


def validator_trig(
    x: float,
    y: float,
    z: float,
    center: tuple[float, float],
    r: float,
    f: float,
    phi_0: float,
    z_0: float,
    abs_tolerance=0,
    rel_tolerance=0,
) -> dict[str, bool | dict | Callable]:
    """Validate against a sin and cos function."""
    # Define test functions
    center_x, center_y = center
    sin_test = lambda x: trig(x, r, 0, f, phi_0, z_0, center_y)  # yz plane
    cos_test = lambda x: trig(x, 0, r, f, phi_0, z_0, center_x)  # xz plane

    # Test sin in yz plane
    distance_sin = lambda z_t: np.sqrt((z_t - z) ** 2 + (sin_test(z_t) - y) ** 2)
    # Test cos in xz plane
    distance_cos = lambda z_t: np.sqrt((z_t - z) ** 2 + (cos_test(z_t) - x) ** 2)

    # Minimize functions to get distance
    result_yz = minimize(distance_sin, x0=z)
    result_xz = minimize(distance_cos, x0=z)

    # Get distance
    dist_yz = result_yz.fun
    dist_xz = result_xz.fun

    # Get z value of minimum
    z_min_yz = result_yz.x[0]
    z_min_xz = result_xz.x[0]
    z_min = (z_min_yz + z_min_xz) / 2

    # Get distance from z_min
    dist_z = abs(z - z_min)
    dist_x = abs(x - cos_test(z_min))
    dist_y = abs(y - sin_test(z_min))
    dist_xyz = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

    validated_yz = dist_yz <= abs_tolerance or dist_yz <= rel_tolerance * r**2
    validated_xz = dist_xz <= abs_tolerance or dist_xz <= rel_tolerance * r**2
    validated_xyz = dist_xyz <= abs_tolerance or dist_xyz <= rel_tolerance * r**2
    validated = validated_yz and validated_xz and validated_xyz

    return {
        "validated": validated,
        "validated_yz": validated_yz,
        "validated_xz": validated_xz,
        "validated_xyz": validated_xyz,
        "diff_yz": dist_yz,
        "diff_xz": dist_xz,
        "diff_z": dist_z,
        "diff_x": dist_x,
        "diff_y": dist_y,
        "diff_xyz": dist_xyz,
        "eq_yz": sin_test,
        "eq_xz": cos_test,
    }


def validator_helix(
    x: float,
    y: float,
    z: float,
    center: tuple[float, float],
    r: float,
    f: float,
    phi_0: float,
    z_0: float,
    abs_tolerance_r=0,
    abs_tolerance_trig=0,
    rel_tolerance=0,
    **kwargs
) -> dict[str, bool | float | dict | Any]:
    """Validate a point against a helix."""

    # Validate circle
    validation_circle: dict = validator_circle(x, y, center, r, abs_tolerance_r, rel_tolerance)  # type: ignore

    if kwargs.get("verbose") == True:
        print("Circle validation:", validation_circle)

    # Fast return, skip further checking if circle is not validated already
    if not validation_circle["validated"]:
        return {"validated": False, "validation_circle": validation_circle}

    # Validate trig
    validation_trig: dict = validator_trig(x, y, z, center, r, f, phi_0, z_0, abs_tolerance_trig, rel_tolerance)  # type: ignore

    if kwargs.get("verbose") == True:
        print("Trig validation:", validation_trig)

    validated = validation_circle["validated"] and validation_trig["validated"]

    diff = validation_trig["diff_xyz"]

    return {
        "validated": validated,
        "validation_circle": validation_circle,
        "validation_trig": validation_trig,
        "diff": diff,
    }


def closer_to_sine_or_cosine_symmetry(phi):
    distance_to_sine_symmetry: float = min(
        abs((phi - math.pi / 2) % math.pi), abs(math.pi - ((phi - math.pi / 2) % math.pi))
    )
    distance_to_cosine_symmetry: float = min(abs(phi % math.pi), abs(math.pi - (phi % math.pi)))

    if distance_to_sine_symmetry < distance_to_cosine_symmetry:
        return "sine"
    elif distance_to_cosine_symmetry < distance_to_sine_symmetry:
        return "cosine"
    else:
        return "equal"


def get_f_fit(X_arr, Y_arr, Z_arr, r, phi_0, z_0, center) -> float:
    """Get frequency `f` by fitting a sine and cosine function to the data."""
    center_x, center_y = center

    # Define test functions
    cos_test = lambda x, f: trig(x, 0, r, f, phi_0, z_0, center_x)
    sin_test = lambda x, f: trig(x, r, 0, f, phi_0, z_0, center_y)

    # Fit functions to data
    popt_zx, pcov = curve_fit(cos_test, Z_arr, X_arr, p0=0)
    popt_zy, pcov = curve_fit(sin_test, Z_arr, Y_arr, p0=0)

    # Get f
    f_zx = popt_zx[0]
    f_zy = popt_zy[0]
    f_zx_abs = abs(f_zx)
    f_zy_abs = abs(f_zy)
    f_abs = (f_zy_abs + f_zx_abs) / 2

    # If a function is symmetric around z_0, the sign of f is not well defined
    # The sign of f is best determined by the function that is least symmetric around z_0
    closest_symmetry = closer_to_sine_or_cosine_symmetry(phi_0)
    if closest_symmetry == "sine":
        # Cosine corresponds with zy, since sin is likely to be symmetric around z_0, choose sign of f_zx
        f_sign = np.sign(f_zx)
    elif closest_symmetry == "cosine":
        # Cosine corresponds with zx, since cos is likely to be symmetric around z_0, choose sign of f_zy
        f_sign = np.sign(f_zy)
    else:
        assert np.sign(f_zx) == np.sign(f_zy), "Signs of f_zx and f_zy should be equal."
        f_sign = np.sign(f_zx)

    f = f_sign * f_abs
    return f


def get_phi_0(center: tuple[float, float]):
    center_x, center_y = center
    return math.atan2(-center_y, -center_x)


def get_validator(
    *args: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    | tuple[Iterable, Iterable, Iterable]
    | tuple[NDArray, NDArray, NDArray],
    thr_collinear=1.0e-6,
    **kwargs
) -> Callable[[float, float, float], dict]:
    """Get validation function for a track in x, y, z plane.

    Input needs to be in the form of [(x1, x2, x3), (y1, y2, y3), (z1, z2, z3).

    Currently only supports 3 points as fit points.
    """
    X_arr, Y_arr, Z_arr = args
    p1 = (X_arr[0], Y_arr[0], Z_arr[0])
    p2 = (X_arr[1], Y_arr[1], Z_arr[1])
    p3 = (X_arr[2], Y_arr[2], Z_arr[2])

    if are_collinear(p1, p2, p3, threshold=thr_collinear):
        # Since collinear, only need to check two points

        if kwargs.get("verbose") == True:
            print("Validator is 3D line")
        return lambda x, y, z: validator_line(x, y, z, p1, p2, **kwargs)  # type: ignore
    # Circle
    else:
        if kwargs.get("verbose") == True:
            print("Validator is helix")

        # Get circle
        center, r = define_circle(p1, p2, p3)
        # Get phi_0
        phi_0 = get_phi_0(center)

        z_0 = 0  # TODO: lenience

        # Get f
        f = get_f_fit(X_arr, Y_arr, Z_arr, r, phi_0, z_0, center)

        validator = lambda x, y, z: validator_helix(x, y, z, center, r, f, phi_0, z_0, **kwargs)

        infodict = {
            "center": center,
            "r": r,
            "f": f,
            "phi_0": phi_0,
            "z_0": z_0,
        }
        return lambda x, y, z: validator(x, y, z) | infodict  # type: ignore
