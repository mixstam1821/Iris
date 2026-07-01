"""
canvas.py — a matplotlib figure embedded as a native Qt widget.

FigureCanvasQTAgg is the standard bridge between matplotlib and Qt: it's a
real QWidget you can drop into any layout like a button or a label.

Things worth calling out:

- Every render fully clears the figure (`figure.clf()`) and rebuilds the
  axes + colorbar from scratch, using `constrained_layout=True`. Reusing the
  same axes across renders while repeatedly adding/removing a colorbar
  accumulates layout drift (the plot area visibly narrows a bit more on
  every update) — a full clear-and-rebuild is the robust fix.
- Zoom is scroll-wheel driven (zoom centered on the cursor), on top of
  matplotlib's own NavigationToolbar2QT (rectangle-zoom / pan / home) added
  in main_window.py — same "scroll to zoom" feel as a map.
- Even though every render rebuilds the axes from scratch, the *current
  zoom level is preserved* across renders that share the same array shape
  and geographic extent as the previous one (e.g. stepping through a time
  dimension, or animating). Only a genuinely new dataset/extent resets the
  view to the full frame. This is what keeps a zoomed-in view from
  snapping back out on every animation tick.
- Optional land-only coastlines can be overlaid on top of either a
  NetCDF field or an RGB image, using Natural Earth "land" polygons
  (cartopy/shapely), drawn as plain line rings — not full cartopy borders,
  rivers, or lakes, just the land/sea boundary. This only works when a
  geographic (lon/lat) extent is known for the current data: auto-detected
  for regular lat/lon NetCDF grids, or set manually in the sidebar for
  loaded RGB images.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Module-level cache of Natural Earth land geometries — loaded once (and
# cached to disk by cartopy itself after the first download) rather than
# re-parsed on every coastline redraw.
_LAND_GEOMETRIES = None


def _get_land_geometries():
    global _LAND_GEOMETRIES
    if _LAND_GEOMETRIES is None:
        import cartopy.io.shapereader as shpreader

        shp_path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMETRIES = list(shpreader.Reader(shp_path).geometries())
    return _LAND_GEOMETRIES


class FieldCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6, 5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

        self.ax = None
        self._image = None
        self._colorbar = None

        # Zoom-preservation bookkeeping.
        self._home_xlim = None
        self._home_ylim = None
        self._current_extent = None   # (lon_min, lon_max, lat_min, lat_max) or None
        self._current_shape = None    # array.shape of the last rendered frame

        # Coastlines.
        self._coastline_color = "#ffcc33"
        self._coastline_cache = None  # {"extent": ..., "lines": [(x, y), ...]}
        self.last_coastline_error: str | None = None
        self._last_render: dict | None = None

        self.show_empty_state()
        self.mpl_connect("scroll_event", self._on_scroll)

    # ── rendering ────────────────────────────────────────────────────────

    def show_empty_state(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.text(
            0.5, 0.5,
            "Open a NetCDF file or RGB image to begin",
            ha="center", va="center",
            fontsize=11, color="#888888",
            transform=self.ax.transAxes,
        )
        self._image = None
        self._colorbar = None
        self._current_extent = None
        self._current_shape = None
        self._last_render = None
        self.draw_idle()

    def _capture_zoom_if_reusable(self, new_extent, new_shape):
        """
        Returns the previous (xlim, ylim) to reapply on the freshly rebuilt
        axes, but only if the previous frame was showing the *same* extent
        and array shape — i.e. this is a same-dataset step (time slider,
        animation, next image frame) rather than a genuinely new dataset,
        which should reset to the full/home view instead.
        """
        if self.ax is None or self._image is None:
            return None
        if self._current_extent != new_extent or self._current_shape != new_shape:
            return None
        return self.ax.get_xlim(), self.ax.get_ylim()

    def plot_field(self, arr: np.ndarray, cmap: str, title: str, units: str,
                    extent=None, show_coastlines: bool = False):
        """Render a 2D scalar field (NetCDF variable slice) with a colorbar."""
        preserved_zoom = self._capture_zoom_if_reusable(extent, arr.shape)

        # Full clear + rebuild: avoids the progressive-shrink artifact that
        # comes from adding/removing a colorbar on axes reused across calls.
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)

        imshow_kwargs = {"cmap": cmap, "origin": "lower", "aspect": "auto"}
        if extent is not None:
            imshow_kwargs["extent"] = extent
        self._image = self.ax.imshow(arr, **imshow_kwargs)
        self.ax.set_title(title, fontsize=10)
        self._colorbar = self.figure.colorbar(self._image, ax=self.ax, label=units or None)

        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()
        if preserved_zoom is not None:
            self.ax.set_xlim(preserved_zoom[0])
            self.ax.set_ylim(preserved_zoom[1])

        self._current_extent = extent
        self._current_shape = arr.shape

        if show_coastlines:
            self._draw_coastlines(extent)
        else:
            self.last_coastline_error = None

        self._last_render = {
            "kind": "field", "arr": arr, "cmap": cmap,
            "title": title, "units": units, "extent": extent,
            "show_coastlines": show_coastlines,
        }
        self.draw_idle()

    def plot_image_rgb(self, arr: np.ndarray, title: str = "", extent=None,
                        show_coastlines: bool = False):
        """
        Render a pre-rendered RGB(A) raster (e.g. an MTG true-color /
        natural-color PNG exported from Xenia) — no colormap, no colorbar.
        `arr` is HxWx3 or HxWx4, uint8.
        """
        preserved_zoom = self._capture_zoom_if_reusable(extent, arr.shape)

        self.figure.clf()
        self.ax = self.figure.add_subplot(111)

        imshow_kwargs = {"origin": "upper", "aspect": "auto"}
        if extent is not None:
            imshow_kwargs["extent"] = extent
        self._image = self.ax.imshow(arr, **imshow_kwargs)
        if title:
            self.ax.set_title(title, fontsize=10)
        self._colorbar = None  # RGB images have no colorbar to manage

        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()
        if preserved_zoom is not None:
            self.ax.set_xlim(preserved_zoom[0])
            self.ax.set_ylim(preserved_zoom[1])

        self._current_extent = extent
        self._current_shape = arr.shape

        if show_coastlines:
            self._draw_coastlines(extent)
        else:
            self.last_coastline_error = None

        self._last_render = {
            "kind": "image", "arr": arr, "title": title, "extent": extent,
            "show_coastlines": show_coastlines,
        }
        self.draw_idle()

    def set_colormap(self, cmap: str):
        if self._image is None or self._last_render is None or self._last_render["kind"] != "field":
            return
        self._image.set_cmap(cmap)
        self.draw_idle()

    def set_clim(self, vmin: float, vmax: float):
        if self._image is None:
            return
        self._image.set_clim(vmin, vmax)
        self.draw_idle()

    # ── coastlines ───────────────────────────────────────────────────────

    def _draw_coastlines(self, extent):
        if self.ax is None:
            return
        if extent is None:
            self.last_coastline_error = (
                "No geographic extent known for this data — coastlines need "
                "lon/lat bounds (auto-detected for regular NetCDF grids, or "
                "set manually for images)."
            )
            return
        try:
            lines = self._get_cached_coastline_lines(extent)
        except ImportError:
            self.last_coastline_error = (
                "Coastlines need cartopy + shapely.\n"
                "Install with:  pip install cartopy shapely"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.last_coastline_error = f"Could not draw coastlines: {exc}"
            return

        self.last_coastline_error = None
        for x, y in lines:
            self.ax.plot(x, y, color=self._coastline_color, linewidth=0.7, zorder=5)

    def _get_cached_coastline_lines(self, extent):
        cache = self._coastline_cache
        if cache is not None and cache["extent"] == extent:
            return cache["lines"]
        lines = self._compute_coastline_lines(extent)
        self._coastline_cache = {"extent": extent, "lines": lines}
        return lines

    def _compute_coastline_lines(self, extent):
        """
        Clip Natural Earth land polygons to (a small padding around) the
        current extent and flatten every exterior/interior ring into plain
        (x, y) coordinate lists, so repeated redraws (e.g. one per
        animation frame) only pay the shapely intersection cost once per
        distinct extent, not once per frame.
        """
        from shapely.geometry import box as shapely_box

        lon_min, lon_max, lat_min, lat_max = extent
        pad = max(lon_max - lon_min, lat_max - lat_min) * 0.05 + 1.0
        bbox = shapely_box(lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)

        lines = []
        for geom in _get_land_geometries():
            if not geom.envelope.intersects(bbox):
                continue
            clipped = geom.intersection(bbox)
            if clipped.is_empty:
                continue
            polys = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
            for poly in polys:
                if poly.geom_type != "Polygon":
                    continue
                x, y = poly.exterior.xy
                lines.append((list(x), list(y)))
                for interior in poly.interiors:
                    xi, yi = interior.xy
                    lines.append((list(xi), list(yi)))
        return lines

    # ── zoom ─────────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if self.ax is None or event.inaxes != self.ax or self._image is None:
            return

        zoom_factor = 0.85 if event.button == "up" else 1 / 0.85

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor

        # Keep the point under the cursor fixed while zooming.
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))
        self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1 - rel_y))
        self.draw_idle()

    def reset_zoom(self):
        if self.ax is None or self._image is None:
            return
        self.ax.set_xlim(self._home_xlim)
        self.ax.set_ylim(self._home_ylim)
        self.draw_idle()