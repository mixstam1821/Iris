"""
make_sample_data.py — generates two synthetic NetCDF files so you can try
Iris immediately without needing a real satellite/reanalysis file on hand:

  sample_field.nc    — plain 2D variables (lat, lon)
  sample_field_nd.nc — N-D variables to exercise the dimension controls:
                        temperature(time, level, lat, lon)
                        cloud_fraction(time, lat, lon)

Run: python make_sample_data.py
"""

import numpy as np
import pandas as pd
import xarray as xr

lat = np.linspace(-90, 90, 91)
lon = np.linspace(-180, 180, 181)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# ── plain 2D file ────────────────────────────────────────────────────────
temperature_2d = (
    288
    - 0.6 * np.abs(lat_grid)
    + 4 * np.sin(np.radians(lon_grid) * 2)
    + np.random.normal(0, 1.5, lat_grid.shape)
)
cloud_fraction_2d = np.clip(
    0.5 + 0.3 * np.sin(np.radians(lat_grid) * 3) + np.random.normal(0, 0.1, lat_grid.shape),
    0, 1,
)

ds_2d = xr.Dataset(
    {
        "temperature": (
            ("lat", "lon"),
            temperature_2d.astype("float32"),
            {"units": "K", "long_name": "Synthetic Surface Temperature"},
        ),
        "cloud_fraction": (
            ("lat", "lon"),
            cloud_fraction_2d.astype("float32"),
            {"units": "1", "long_name": "Synthetic Cloud Fraction"},
        ),
    },
    coords={"lat": lat, "lon": lon},
)
ds_2d.to_netcdf("sample_field.nc")
print("Wrote sample_field.nc (2D)")

# ── N-D file: time + level, to exercise slider/dropdown controls ──────────
n_time = 8
n_level = 4
times = pd.date_range("2026-07-01", periods=n_time, freq="6h")
levels = np.array([1000, 850, 500, 200])  # hPa

temperature_4d = np.empty((n_time, n_level, lat.size, lon.size), dtype="float32")
for t in range(n_time):
    for lv in range(n_level):
        drift = 2.0 * np.sin(2 * np.pi * t / n_time)
        altitude_cooling = 0.04 * (1000 - levels[lv])
        temperature_4d[t, lv] = (
            288
            - 0.6 * np.abs(lat_grid)
            - altitude_cooling
            + drift
            + 4 * np.sin(np.radians(lon_grid) * 2 + t)
            + np.random.normal(0, 1.0, lat_grid.shape)
        )

cloud_fraction_3d = np.empty((n_time, lat.size, lon.size), dtype="float32")
for t in range(n_time):
    phase = 2 * np.pi * t / n_time
    cloud_fraction_3d[t] = np.clip(
        0.5 + 0.3 * np.sin(np.radians(lat_grid) * 3 + phase) + np.random.normal(0, 0.1, lat_grid.shape),
        0, 1,
    )

ds_nd = xr.Dataset(
    {
        "temperature": (
            ("time", "level", "lat", "lon"),
            temperature_4d,
            {"units": "K", "long_name": "Synthetic Air Temperature"},
        ),
        "cloud_fraction": (
            ("time", "lat", "lon"),
            cloud_fraction_3d,
            {"units": "1", "long_name": "Synthetic Cloud Fraction"},
        ),
    },
    coords={"time": times, "level": levels, "lat": lat, "lon": lon},
)
ds_nd.to_netcdf("sample_field_nd.nc")
print("Wrote sample_field_nd.nc (time, level, lat, lon)")
