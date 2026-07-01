"""
main_window.py — the QMainWindow: layout, menus, dynamic dimension controls,
and the glue between the UI and the background workers in worker.py.

Two data sources, one canvas:

    "netcdf" mode — the original behaviour. Open a NetCDF file, pick a
    variable, get automatic controls for any extra (time/level/...) dims.
    A variable's last two dims are treated as the spatial (y, x) grid.

    "image" mode — open one or more pre-rendered RGB(A) rasters (e.g. MTG
    true-color / natural-color PNGs exported from Xenia). Multiple files
    become "frames" you can step through or animate with the same
    slider + Play/Pause pattern used for a NetCDF time dimension.

Geographic extent & coastlines:
    Both modes share a "Geo Extent" panel (lon min/max, lat min/max) and a
    "Show coastlines (land)" checkbox. For NetCDF files with a regular 1D
    lon/lat grid, the extent is auto-detected and the checkbox is turned on
    automatically; for images (which carry no embedded geo-referencing)
    the fields default to the full globe and are meant to be edited by hand
    to match whatever region the image actually covers. Coastlines are
    drawn from Natural Earth "land" polygons — land/sea boundary only, no
    borders/rivers/lakes — and only render when the extent checkbox is on.

Zoom persistence:
    Stepping a dimension slider, animating, or paging through image frames
    keeps the current zoom level (canvas.py only resets the view when the
    array shape or extent actually changes, i.e. a genuinely new dataset).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from iris.canvas import FieldCanvas
from iris.worker import (
    ImageFrame,
    ImageLoadResult,
    ImageLoadWorker,
    InspectResult,
    InspectWorker,
    LoadWorker,
    VariableInfo,
    run_in_thread,
)

COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu_r", "turbo", "jet", "gray",
]

PLAY_INTERVAL_MS = 500

# Default extent shown for freshly-loaded images with no known geo-referencing.
_DEFAULT_EXTENT = (-180.0, 180.0, -90.0, 90.0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris — NetCDF / RGB Image Viewer")
        self.resize(1160, 720)

        self._mode: str = "empty"  # "empty" | "netcdf" | "image"

        # NetCDF state.
        self._filepath: str | None = None
        self._variables: list[VariableInfo] = []
        self._dim_coords: dict = {}
        self._current_variable: VariableInfo | None = None
        self._current_array = None
        self._detected_extent: tuple | None = None

        # Image state.
        self._image_frames: list[ImageFrame] = []
        self._image_frame_index: int = 0

        self._threads = []       # keep worker threads alive until they finish
        self._dim_controls = {}  # dim_name -> {"widget": ..., "kind": "slider"/"combo"}
        self._request_counter = 0

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(PLAY_INTERVAL_MS)
        self._play_timer.timeout.connect(self._advance_playback)
        self._playing_dim = None  # NetCDF dim name being animated, when in netcdf mode

        self._build_menu()
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — open a NetCDF file or RGB image to begin.")

    # ── UI construction ──────────────────────────────────────────────────

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open NetCDF…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        open_images_action = QAction("Open &RGB Image(s)…", self)
        open_images_action.setShortcut("Ctrl+Shift+O")
        open_images_action.triggered.connect(self.open_images_dialog)
        file_menu.addAction(open_images_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # ── Sidebar ──
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)

        open_btn = QPushButton("Open NetCDF File…")
        open_btn.clicked.connect(self.open_file_dialog)
        sidebar_layout.addWidget(open_btn)

        open_images_btn = QPushButton("Open RGB Image(s)…")
        open_images_btn.clicked.connect(self.open_images_dialog)
        sidebar_layout.addWidget(open_images_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #666; font-size: 11px;")
        sidebar_layout.addWidget(self.file_label)

        # -- NetCDF: variable picker --
        self.var_group = QGroupBox("Variable")
        var_layout = QFormLayout(self.var_group)
        self.variable_combo = QComboBox()
        self.variable_combo.setEnabled(False)
        self.variable_combo.currentIndexChanged.connect(self._on_variable_selected)
        var_layout.addRow(self.variable_combo)
        sidebar_layout.addWidget(self.var_group)

        # -- NetCDF: dynamic dimension controls (time slider, level dropdown) --
        self.dims_group = QGroupBox("Dimensions")
        self.dims_layout = QVBoxLayout(self.dims_group)
        self.dims_group.setVisible(False)
        sidebar_layout.addWidget(self.dims_group)

        # -- Images: frame slider + play/pause, for paging/animating through
        #    multiple loaded RGB images the same way a time dim animates --
        self.image_frames_group = QGroupBox("Image Frames")
        image_frames_layout = QVBoxLayout(self.image_frames_group)

        frame_header = QHBoxLayout()
        frame_header.addWidget(QLabel("<b>frame</b>"))
        self.frame_name_label = QLabel("—")
        self.frame_name_label.setStyleSheet("color: #555;")
        frame_header.addWidget(self.frame_name_label)
        frame_header.addStretch(1)
        image_frames_layout.addLayout(frame_header)

        frame_controls_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_play_btn = QPushButton("▶ Play")
        self.frame_play_btn.setCheckable(True)
        self.frame_play_btn.toggled.connect(self._on_frame_play_toggled)
        frame_controls_row.addWidget(self.frame_slider, 1)
        frame_controls_row.addWidget(self.frame_play_btn)
        image_frames_layout.addLayout(frame_controls_row)

        self.image_frames_group.setVisible(False)
        sidebar_layout.addWidget(self.image_frames_group)

        # -- Display: colormap/clim (NetCDF only) + coastlines (both modes) --
        display_group = QGroupBox("Display")
        display_layout = QFormLayout(display_group)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        display_layout.addRow("Colormap", self.cmap_combo)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.valueChanged.connect(self._on_clim_changed)
        display_layout.addRow("Min", self.vmin_spin)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(3)
        self.vmax_spin.valueChanged.connect(self._on_clim_changed)
        display_layout.addRow("Max", self.vmax_spin)

        self.coastlines_checkbox = QCheckBox("Show coastlines (land)")
        self.coastlines_checkbox.toggled.connect(self._on_coastlines_toggled)
        display_layout.addRow(self.coastlines_checkbox)

        sidebar_layout.addWidget(display_group)

        # -- Geo extent (used for coastline alignment; auto-filled for
        #    regular lat/lon NetCDF grids, manual for images) --
        geo_group = QGroupBox("Geo Extent (for coastlines)")
        geo_layout = QFormLayout(geo_group)

        self.lon_min_spin = self._make_extent_spin(_DEFAULT_EXTENT[0])
        self.lon_max_spin = self._make_extent_spin(_DEFAULT_EXTENT[1])
        self.lat_min_spin = self._make_extent_spin(_DEFAULT_EXTENT[2])
        self.lat_max_spin = self._make_extent_spin(_DEFAULT_EXTENT[3])
        geo_layout.addRow("Lon min", self.lon_min_spin)
        geo_layout.addRow("Lon max", self.lon_max_spin)
        geo_layout.addRow("Lat min", self.lat_min_spin)
        geo_layout.addRow("Lat max", self.lat_max_spin)

        sidebar_layout.addWidget(geo_group)

        reset_zoom_btn = QPushButton("Reset Zoom")
        reset_zoom_btn.clicked.connect(self._on_reset_zoom)
        sidebar_layout.addWidget(reset_zoom_btn)

        info_group = QGroupBox("Info")
        info_layout = QFormLayout(info_group)
        self.shape_label = QLabel("—")
        self.units_label = QLabel("—")
        info_layout.addRow("Shape", self.shape_label)
        info_layout.addRow("Units", self.units_label)
        sidebar_layout.addWidget(info_group)

        sidebar_layout.addStretch(1)
        sidebar.setMaximumWidth(300)

        # ── Canvas + zoom/pan toolbar ──
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = FieldCanvas(self)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, canvas_container)
        canvas_layout.addWidget(self.nav_toolbar)
        canvas_layout.addWidget(self.canvas)

        splitter.addWidget(sidebar)
        splitter.addWidget(canvas_container)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _make_extent_spin(self, default_value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setDecimals(3)
        spin.setValue(default_value)
        spin.valueChanged.connect(self._on_extent_changed)
        return spin

    # ── Mode switching ───────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        is_netcdf = mode == "netcdf"
        is_image = mode == "image"

        self.var_group.setVisible(is_netcdf)
        self.dims_group.setVisible(is_netcdf and bool(self._dim_controls))
        self.image_frames_group.setVisible(is_image)

        # Colormap/clim only mean something for scalar NetCDF fields.
        self.cmap_combo.setEnabled(is_netcdf)
        self.vmin_spin.setEnabled(is_netcdf)
        self.vmax_spin.setEnabled(is_netcdf)

    # ── NetCDF: file loading ─────────────────────────────────────────────

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NetCDF file",
            "",
            "NetCDF files (*.nc *.nc4 *.netcdf);;All files (*)",
        )
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path: str):
        self._stop_playback()
        self._filepath = path
        self.file_label.setText(path)
        self.statusBar().showMessage(f"Inspecting {path}…")
        self.variable_combo.setEnabled(False)
        self.variable_combo.clear()
        self._clear_dim_controls()

        worker = InspectWorker(path)
        worker.finished.connect(self._on_inspect_finished)
        worker.failed.connect(self._on_worker_failed)
        thread = run_in_thread(worker, self)
        self._threads.append((thread, worker))

    def _on_inspect_finished(self, result: InspectResult):
        self._set_mode("netcdf")
        self._variables = result.variables
        self._dim_coords = result.dim_coords
        self._detected_extent = result.lonlat_extent

        if result.lonlat_extent is not None:
            self._set_extent_spins(result.lonlat_extent)
            self.coastlines_checkbox.blockSignals(True)
            self.coastlines_checkbox.setChecked(True)
            self.coastlines_checkbox.blockSignals(False)
            extent_note = f" Detected lon/lat extent from {result.lon_dim}/{result.lat_dim}."
        else:
            extent_note = " No regular lon/lat grid detected — set extent manually to use coastlines."

        self.variable_combo.clear()
        for v in self._variables:
            label = f"{v.name}  {v.shape}"
            self.variable_combo.addItem(label, userData=v.name)
        self.variable_combo.setEnabled(True)
        self.statusBar().showMessage(
            f"Found {len(self._variables)} variable(s)." + extent_note, 6000
        )
        self._cleanup_finished_threads()

        # Triggers _on_variable_selected via the signal already connected,
        # since index changes from -1 (empty) to 0.

    def _on_variable_selected(self, index: int):
        self._stop_playback()
        if index < 0 or not self._filepath:
            return
        var_name = self.variable_combo.itemData(index)
        var_info = next((v for v in self._variables if v.name == var_name), None)
        if var_info is None:
            return

        self._current_variable = var_info
        self._build_dim_controls(var_info)
        self._reload_slice()

    # ── NetCDF: dynamic dimension controls ───────────────────────────────

    def _clear_dim_controls(self):
        while self.dims_layout.count():
            item = self.dims_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._dim_controls = {}
        self.dims_group.setVisible(False)

    def _build_dim_controls(self, var_info: VariableInfo):
        self._stop_playback()
        self._clear_dim_controls()

        # Last two dims are treated as the spatial (row, col) grid; anything
        # before that needs a control so the user can pick a slice.
        extra_dims = list(zip(var_info.dims, var_info.shape))[:-2] if len(var_info.dims) > 2 else []

        if not extra_dims:
            self.dims_group.setVisible(False)
            return

        for dim_name, size in extra_dims:
            coords = self._dim_coords.get(dim_name)
            is_time = "time" in dim_name.lower()

            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)

            header = QHBoxLayout()
            header.addWidget(QLabel(f"<b>{dim_name}</b>"))
            value_label = QLabel(self._dim_value_label(dim_name, 0))
            value_label.setStyleSheet("color: #555;")
            header.addWidget(value_label)
            header.addStretch(1)
            row_layout.addLayout(header)

            if is_time:
                controls_row = QHBoxLayout()
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, max(size - 1, 0))
                slider.setValue(0)
                slider.valueChanged.connect(
                    lambda val, d=dim_name, lbl=value_label: self._on_dim_slider_changed(d, val, lbl)
                )
                play_btn = QPushButton("▶ Play")
                play_btn.setCheckable(True)
                play_btn.toggled.connect(lambda checked, d=dim_name: self._on_dim_play_toggled(d, checked))

                controls_row.addWidget(slider, 1)
                controls_row.addWidget(play_btn)
                row_layout.addLayout(controls_row)

                self._dim_controls[dim_name] = {
                    "kind": "slider",
                    "widget": slider,
                    "label": value_label,
                    "play_btn": play_btn,
                    "size": size,
                }
            else:
                combo = QComboBox()
                if coords:
                    combo.addItems(coords)
                else:
                    combo.addItems([f"index {i}" for i in range(size)])
                combo.currentIndexChanged.connect(
                    lambda val, d=dim_name, lbl=value_label: self._on_dim_combo_changed(d, val, lbl)
                )
                row_layout.addWidget(combo)

                self._dim_controls[dim_name] = {
                    "kind": "combo",
                    "widget": combo,
                    "label": value_label,
                    "size": size,
                }

            self.dims_layout.addWidget(row)

        self.dims_group.setVisible(self._mode == "netcdf")

    def _dim_value_label(self, dim_name: str, index: int) -> str:
        coords = self._dim_coords.get(dim_name)
        if coords and 0 <= index < len(coords):
            return coords[index]
        return f"index {index}"

    def _on_dim_slider_changed(self, dim_name: str, value: int, label: QLabel):
        label.setText(self._dim_value_label(dim_name, value))
        self._reload_slice()

    def _on_dim_combo_changed(self, dim_name: str, value: int, label: QLabel):
        if value < 0:
            return
        label.setText(self._dim_value_label(dim_name, value))
        self._reload_slice()

    def _current_indexers(self) -> dict:
        indexers = {}
        for dim_name, ctrl in self._dim_controls.items():
            if ctrl["kind"] == "slider":
                indexers[dim_name] = ctrl["widget"].value()
            else:
                indexers[dim_name] = max(ctrl["widget"].currentIndex(), 0)
        return indexers

    # ── Playback (time animation for NetCDF, frame animation for images) ──

    def _on_dim_play_toggled(self, dim_name: str, checked: bool):
        ctrl = self._dim_controls.get(dim_name)
        if ctrl is None:
            return
        if checked:
            self._playing_dim = dim_name
            ctrl["play_btn"].setText("⏸ Pause")
            self._play_timer.start()
        else:
            self._play_timer.stop()
            ctrl["play_btn"].setText("▶ Play")

    def _on_frame_play_toggled(self, checked: bool):
        if checked:
            self.frame_play_btn.setText("⏸ Pause")
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self.frame_play_btn.setText("▶ Play")

    def _advance_playback(self):
        """
        Single QTimer callback shared by both animation modes: NetCDF
        dimension-slider playback and image-frame playback. Advancing
        either just nudges the relevant slider — the zoom-preserving logic
        lives in canvas.py, so playback here never has to think about zoom.
        """
        if self._mode == "netcdf":
            dim_name = self._playing_dim
            ctrl = self._dim_controls.get(dim_name) if dim_name else None
            if ctrl is None or ctrl["kind"] != "slider":
                self._stop_playback()
                return
            slider = ctrl["widget"]
            next_val = slider.value() + 1
            if next_val > slider.maximum():
                next_val = 0
            slider.setValue(next_val)  # triggers reload via valueChanged
        elif self._mode == "image":
            if self.frame_slider.maximum() == 0:
                self._stop_playback()
                return
            next_val = self.frame_slider.value() + 1
            if next_val > self.frame_slider.maximum():
                next_val = 0
            self.frame_slider.setValue(next_val)  # triggers _on_frame_slider_changed
        else:
            self._stop_playback()

    def _stop_playback(self):
        self._play_timer.stop()
        for ctrl in self._dim_controls.values():
            if ctrl["kind"] == "slider" and "play_btn" in ctrl:
                ctrl["play_btn"].blockSignals(True)
                ctrl["play_btn"].setChecked(False)
                ctrl["play_btn"].setText("▶ Play")
                ctrl["play_btn"].blockSignals(False)
        self.frame_play_btn.blockSignals(True)
        self.frame_play_btn.setChecked(False)
        self.frame_play_btn.setText("▶ Play")
        self.frame_play_btn.blockSignals(False)
        self._playing_dim = None

    # ── NetCDF: slice loading ────────────────────────────────────────────

    def _reload_slice(self):
        if not self._filepath or self._current_variable is None:
            return

        self._request_counter += 1
        request_id = self._request_counter
        indexers = self._current_indexers()

        self.statusBar().showMessage(f"Loading '{self._current_variable.name}'…")

        worker = LoadWorker(self._filepath, self._current_variable.name, indexers, request_id)
        worker.finished.connect(self._on_load_finished)
        worker.failed.connect(self._on_worker_failed)
        thread = run_in_thread(worker, self)
        self._threads.append((thread, worker))

    def _on_load_finished(self, arr, meta: dict):
        if meta.get("request_id", -1) != self._request_counter:
            # A newer request has already been issued (e.g. user dragged the
            # slider further) — discard this now-stale result.
            self._cleanup_finished_threads()
            return

        self._current_array = arr

        self.shape_label.setText(str(meta["shape"]))
        self.units_label.setText(meta["units"] or "—")

        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(meta["vmin"])
        self.vmax_spin.setValue(meta["vmax"])
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

        self.canvas.plot_field(
            arr,
            cmap=self.cmap_combo.currentText(),
            title=meta["long_name"],
            units=meta["units"],
            extent=self._active_extent(),
            show_coastlines=self.coastlines_checkbox.isChecked(),
        )
        self._warn_if_coastline_error()
        self.statusBar().showMessage("Rendered.", 3000)
        self._cleanup_finished_threads()

    # ── Images: loading ──────────────────────────────────────────────────

    def open_images_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open RGB image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
        )
        if not paths:
            return
        self._load_images(paths)

    def _load_images(self, paths: list[str]):
        self._stop_playback()
        self.statusBar().showMessage(f"Loading {len(paths)} image(s)…")

        worker = ImageLoadWorker(paths)
        worker.finished.connect(self._on_images_loaded)
        worker.failed.connect(self._on_worker_failed)
        thread = run_in_thread(worker, self)
        self._threads.append((thread, worker))

    def _on_images_loaded(self, result: ImageLoadResult):
        self._set_mode("image")
        self._image_frames = result.frames
        self._image_frame_index = 0

        if len(self._image_frames) == 1:
            self.file_label.setText(self._image_frames[0].name)
        else:
            self.file_label.setText(f"{len(self._image_frames)} images loaded")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, len(self._image_frames) - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self._render_current_image_frame()
        self.statusBar().showMessage(
            f"Loaded {len(self._image_frames)} image(s).", 4000
        )
        self._cleanup_finished_threads()

    def _on_frame_slider_changed(self, value: int):
        if not self._image_frames:
            return
        self._image_frame_index = max(0, min(value, len(self._image_frames) - 1))
        self._render_current_image_frame()

    def _render_current_image_frame(self):
        if not self._image_frames:
            return
        frame = self._image_frames[self._image_frame_index]
        self.frame_name_label.setText(frame.name)
        self.shape_label.setText(str(frame.array.shape))
        self.units_label.setText("RGB")

        self.canvas.plot_image_rgb(
            frame.array, title=frame.name, extent=self._active_extent(),
            show_coastlines=self.coastlines_checkbox.isChecked(),
        )
        self._warn_if_coastline_error()

    # ── Geo extent / coastlines ─────────────────────────────────────────

    def _set_extent_spins(self, extent: tuple):
        for spin, value in zip(
            (self.lon_min_spin, self.lon_max_spin, self.lat_min_spin, self.lat_max_spin), extent
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _active_extent(self):
        """
        The extent passed to the canvas: only meaningful (and only used)
        when coastlines are enabled, since NetCDF field rendering otherwise
        deliberately stays in plain pixel-index axes.
        """
        if not self.coastlines_checkbox.isChecked():
            return None
        return (
            self.lon_min_spin.value(), self.lon_max_spin.value(),
            self.lat_min_spin.value(), self.lat_max_spin.value(),
        )

    def _on_extent_changed(self, _value: float):
        self._rerender_current()

    def _on_coastlines_toggled(self, _checked: bool):
        self._rerender_current()

    def _rerender_current(self):
        if self._mode == "netcdf" and self._current_array is not None:
            self.canvas.plot_field(
                self._current_array,
                cmap=self.cmap_combo.currentText(),
                title=self._current_variable.name if self._current_variable else "",
                units=self.units_label.text(),
                extent=self._active_extent(),
                show_coastlines=self.coastlines_checkbox.isChecked(),
            )
            self._warn_if_coastline_error()
        elif self._mode == "image" and self._image_frames:
            self._render_current_image_frame()

    def _warn_if_coastline_error(self):
        if self.canvas.last_coastline_error:
            self.statusBar().showMessage(self.canvas.last_coastline_error, 6000)

    # ── Shared worker plumbing ───────────────────────────────────────────

    def _on_worker_failed(self, message: str):
        self.statusBar().showMessage("Error.", 4000)
        QMessageBox.warning(self, "Iris", message)
        self._cleanup_finished_threads()

    def _cleanup_finished_threads(self):
        self._threads = [(t, w) for (t, w) in self._threads if t.isRunning()]

    # ── Display controls ─────────────────────────────────────────────────

    def _on_cmap_changed(self, name: str):
        self.canvas.set_colormap(name)

    def _on_clim_changed(self, _value: float):
        if self._current_array is None:
            return
        self.canvas.set_clim(self.vmin_spin.value(), self.vmax_spin.value())

    def _on_reset_zoom(self):
        self.canvas.reset_zoom()