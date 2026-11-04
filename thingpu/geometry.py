"""Points on the sphere and geodesic radii."""
import numpy as np

# Mean Earth radius (IUGG), in metres.
EARTH_RADIUS_M = 6_371_008.8

# Two records are neighbours when their distance is strictly below d, as in
# spThin and GeoThinneR. In double precision this factor turns the <= of the
# k-d tree query into a strict <.
STRICT = 1.0 - 1e-12


def to_xyz(lon_deg, lat_deg, radius=EARTH_RADIUS_M):
    """Longitude and latitude in degrees to (N, 3) Cartesian coordinates in metres."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    c = np.cos(lat)
    return np.stack([radius * c * np.cos(lon), radius * c * np.sin(lon), radius * np.sin(lat)], axis=1)


def chord_radius(distance_m, radius=EARTH_RADIUS_M):
    """Straight-line (chord) distance between two points that are distance_m apart on the sphere.

    A Euclidean radius query with this value returns exactly the records within
    the geodesic (great-circle) distance, so any spatial index can be used.
    """
    return 2.0 * radius * np.sin(np.asarray(distance_m, dtype=np.float64) / (2.0 * radius))


# The GPU compares squared distances in float32, so a pair whose distance is within
# about one float32 rounding of the radius can fall on either side of it. Checks of
# GPU results allow this relative margin around the radius (about 1 mm at 1 km).
F32_MARGIN = 1e-6


def gpu_coordinates(xyz):
    """The float32 coordinates the GPU receives: centred on their mean, which keeps a
    resolution of a few centimetres for a country instead of 0.5 m at Earth scale."""
    return np.ascontiguousarray((xyz - xyz.mean(axis=0)).astype(np.float32))


def query_frame(xyz, distance_m, precision="f64"):
    """Coordinates and inclusive query radius equivalent to "distance strictly below d".

    f64: the coordinates as given and the chord lowered by a relative 1e-12.
    f32: the GPU's float32 coordinates and its radius (the float32 chord lowered by
    one step), so that the CPU reproduces the decisions of the GPU exactly.
    """
    chord = float(chord_radius(distance_m))
    if precision == "f64":
        return xyz, chord * STRICT
    if precision == "f32":
        return (gpu_coordinates(xyz).astype(np.float64),
                float(np.nextafter(np.float32(chord), np.float32(0))))
    raise ValueError(f"unknown precision: {precision}")
