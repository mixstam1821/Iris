"""
worker.py — background work off the GUI thread.

Qt's rule #1: never do slow I/O or heavy compute on the main thread, or the
whole UI (window dragging, button repaint, everything) freezes.

Workers:
- InspectWorker: opens a NetCDF file, lists all variables with >=2 dims,
  collects 1D coordinate arrays for every dimension it sees (so the UI can
  show real time/level values in dropdowns and sliders instead of bare
  indices), and — if it can find 1D lon/lat coordinate variables — also
  works out the geographic extent of the grid so the canvas can align
  coastlines against it.
- LoadWorker: given a variable name and an `indexers` dict selecting one
  index for every non-spatial dimension, pulls the resulting 2D slice into
  memory as a plain numpy array.
- ImageLoadWorker: loads one or more pre-rendered RGB(A) raster images
  (e.g. MTG true-color / natural-color PNGs exported from Xenia) from disk
  into plain uint8 numpy arrays, so they can be paged through / animated
  exactly like a NetCDF time dimension.

All workers run on a QThread via a QObject "worker" moved to that thread —
the standard Qt pattern, as opposed to subclassing QThread directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from PySide6.QtCore import QObject, QThread, Signal

# Names we'll look for when trying to detect geographic (lon/lat) coordinates
# on a regular lat/lon grid, so coastlines can be aligned against the data.
_LON_NAMES = ("lon", "longitude", "x")
_LAT_NAMES = ("lat", "latitude", "y")


@dataclass
class VariableInfo:
    name: str
    dims: tuple
    shape: tuple
    units: str = ""


@dataclass
class InspectResult:
    variables: list = field(default_factory=list)          # list[VariableInfo]
    dim_coords: dict = field(default_factory=dict)          # dim_name -> list[str]
    lonlat_extent: tuple | None = None                       # (lon_min, lon_max, lat_min, lat_max)
    lon_dim: str | None = None
    lat_dim: str | None = None


def _open_dataset(filepath: str) -> xr.Dataset:
    """Try CF-decoded first (gives real datetimes for time dims); fall back
    to raw values if the file has a calendar/units xarray can't decode."""
    try:
        return xr.open_dataset(filepath)
    except Exception:
        return xr.open_dataset(filepath, decode_times=False)


def _find_lonlat_extent(ds: xr.Dataset) -> tuple:
    """
    Best-effort detection of a regular 1D lon/lat grid so the canvas can
    set an imshow(extent=...) that lines up with real-world coastlines.

    Returns (extent, lon_dim, lat_dim) where extent is
    (lon_min, lon_max, lat_min, lat_max), or (None, None, None) if no
    plausible 1D lon/lat coordinate pair was found. Only handles the
    common case of a regular (non-rotated, non-projected) lat/lon grid —
    swath / geostationary data needs its own reprojected extent, supplied
    manually via the sidebar's "Geographic extent" fields instead.
    """
    lon_dim = lat_dim = None
    lon_vals = lat_vals = None

    for name in ds.coords:
        lname = str(name).lower()
        if lon_dim is None and lname in _LON_NAMES and ds.coords[name].ndim == 1:
            lon_dim, lon_vals = name, np.asarray(ds.coords[name].values, dtype="float64")
        if lat_dim is None and lname in _LAT_NAMES and ds.coords[name].ndim == 1:
            lat_dim, lat_vals = name, np.asarray(ds.coords[name].values, dtype="float64")

    if lon_vals is None or lat_vals is None or lon_vals.size < 2 or lat_vals.size < 2:
        return None, None, None

    extent = (
        float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals)),
        float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals)),
    )
    return extent, lon_dim, lat_dim


class InspectWorker(QObject):
    finished = Signal(object)  # InspectResult
    failed = Signal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            with _open_dataset(self.filepath) as ds:
                variables = []
                for name, da in ds.data_vars.items():
                    if da.ndim < 2:
                        continue
                    variables.append(
                        VariableInfo(
                            name=name,
                            dims=tuple(da.dims),
                            shape=tuple(da.shape),
                            units=str(da.attrs.get("units", "")),
                        )
                    )

                if not variables:
                    self.failed.emit("No 2D+ variables found in this file.")
                    return

                # Collect coordinate labels for every dimension referenced by
                # any of the chosen variables (small 1D arrays only).
                needed_dims = {d for v in variables for d in v.dims}
                dim_coords: dict[str, list] = {}
                for dim in needed_dims:
                    if dim in ds.coords:
                        try:
                            values = ds.coords[dim].values
                            dim_coords[dim] = [str(v)[:19] for v in values]
                        except Exception:
                            pass

                extent, lon_dim, lat_dim = _find_lonlat_extent(ds)

            self.finished.emit(
                InspectResult(
                    variables=variables,
                    dim_coords=dim_coords,
                    lonlat_extent=extent,
                    lon_dim=lon_dim,
                    lat_dim=lat_dim,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self.failed.emit(f"Could not open file:\n{exc}")


class LoadWorker(QObject):
    finished = Signal(np.ndarray, dict)  # (2D array, meta dict)
    failed = Signal(str)

    def __init__(self, filepath: str, variable: str, indexers: dict, request_id: int = 0):
        super().__init__()
        self.filepath = filepath
        self.variable = variable
        self.indexers = indexers or {}
        self.request_id = request_id

    def run(self):
        try:
            with _open_dataset(self.filepath) as ds:
                da = ds[self.variable]

                for dim, idx in self.indexers.items():
                    if dim in da.dims:
                        da = da.isel({dim: int(idx)})

                # Safety net: collapse any remaining extra dims (e.g. a dim
                # the UI didn't know about yet) by taking the first index.
                while da.ndim > 2:
                    da = da.isel({da.dims[0]: 0})

                if da.ndim != 2:
                    self.failed.emit(
                        f"'{self.variable}' resolved to {da.ndim}D, expected 2D."
                    )
                    return

                arr = np.asarray(da.values, dtype="float64")
                meta = {
                    "shape": arr.shape,
                    "units": str(da.attrs.get("units", "")),
                    "long_name": str(da.attrs.get("long_name", self.variable)),
                    "vmin": float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0,
                    "vmax": float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0,
                    "request_id": self.request_id,
                }
            self.finished.emit(arr, meta)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Could not load variable '{self.variable}':\n{exc}")


@dataclass
class ImageFrame:
    name: str
    array: np.ndarray  # HxWx3 or HxWx4, uint8


@dataclass
class ImageLoadResult:
    frames: list = field(default_factory=list)  # list[ImageFrame]


class ImageLoadWorker(QObject):
    """
    Loads one or more pre-rendered RGB(A) raster images — e.g. MTG true
    color / natural color PNGs exported from Xenia — into plain uint8
    numpy arrays. Multiple files become "frames" the UI can page through
    or animate exactly like a NetCDF time dimension.
    """

    finished = Signal(object)  # ImageLoadResult
    failed = Signal(str)

    def __init__(self, filepaths: list[str]):
        super().__init__()
        self.filepaths = list(filepaths)

    def run(self):
        try:
            from PIL import Image
        except ImportError:
            self.failed.emit(
                "Pillow is required to load PNG/JPEG images.\n"
                "Install it with:  pip install Pillow"
            )
            return

        frames = []
        errors = []
        # Sorted so a folder of e.g. MTG_2024...t1.png, t2.png, ... plays
        # back in chronological order regardless of OS listing order.
        for path in sorted(self.filepaths, key=lambda p: os.path.basename(p)):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGBA") if img.mode not in ("RGB", "RGBA") else img
                    arr = np.array(img)
                frames.append(ImageFrame(name=os.path.basename(path), array=arr))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{os.path.basename(path)}: {exc}")

        if not frames:
            self.failed.emit("Could not load any of the selected images:\n" + "\n".join(errors))
            return

        self.finished.emit(ImageLoadResult(frames=frames))


def run_in_thread(worker: QObject, owner: QObject) -> QThread:
    """
    Move `worker` onto a new QThread and start it. `owner` must keep a
    reference to the returned thread (and the worker) alive until finished/
    failed fires, or Qt will garbage-collect it mid-run.
    """
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.start()
    return thread