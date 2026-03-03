"""
PlantBuilder Task Panel — FreeCAD right-side panel for picking a YAML
project, viewing its summary, and running the build workflow.
"""
from __future__ import annotations

import glob
import logging
import os
import traceback

from PySide2 import QtCore, QtGui, QtWidgets

from config.loader import load_config
from core.logging_setup import get_logger

logger = get_logger(__name__)

# Directories relative to this file
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_EXAMPLES_DIR = os.path.join(_SRC_DIR, "yaml_files", "examples")
_YAML_BASE_DIR = os.path.join(_SRC_DIR, "yaml_files", "base")


class PlantBuilderPanel:
    """FreeCAD Task Panel for PlantBuilder."""

    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("PlantBuilder")
        self.selected_path: str | None = None
        self._log_handler = None
        self._last_cuts_file = None
        # Maps seg_name → wafer list from the most recent "Build from Cut List" run.
        # Used by "Align Reconstruction" so it can access entry_mark_dir per wafer.
        self._last_rec_wafers: dict = {}
        # Accumulated rows for the error results table.
        self._result_rows: list = []
        self._apply_saved_log_level()
        self._build_ui()
        self._populate_dropdown()
        self._install_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)

        # --- Project File section ---
        pf_header = QtWidgets.QHBoxLayout()
        pf_header.addWidget(self._make_heading("Project File"))
        pf_header.addStretch()
        btn_prefs = QtWidgets.QPushButton("\u2699")
        btn_prefs.setFixedSize(26, 26)
        btn_prefs.setFlat(True)
        btn_prefs.setToolTip("Preferences")
        btn_prefs.clicked.connect(self._open_preferences)
        pf_header.addWidget(btn_prefs)
        layout.addLayout(pf_header)

        file_row = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        self.combo.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.combo.customContextMenuRequested.connect(self._on_combo_context_menu)
        file_row.addWidget(self.combo)
        self.btn_browse = QtWidgets.QPushButton("Browse\u2026")
        self.btn_browse.setToolTip("Open a .yml project file from disk")
        self.btn_browse.clicked.connect(self._on_browse)
        file_row.addWidget(self.btn_browse)
        layout.addLayout(file_row)

        # --- Project Info section ---
        layout.addWidget(self._make_heading("Project Info"))

        info_grid = QtWidgets.QFormLayout()
        self.lbl_name = QtWidgets.QLabel("-")
        self.lbl_segments = QtWidgets.QLabel("-")
        self.lbl_operations = QtWidgets.QLabel("-")
        info_grid.addRow("Name:", self.lbl_name)
        info_grid.addRow("Segments:", self.lbl_segments)
        info_grid.addRow("Operations:", self.lbl_operations)
        layout.addLayout(info_grid)

        # --- Workflow section ---
        layout.addWidget(self._make_heading("Workflow"))

        self.workflow_list = QtWidgets.QListWidget()
        self.workflow_list.setMaximumHeight(120)
        layout.addWidget(self.workflow_list)

        # --- Wafer Settings section ---
        layout.addWidget(self._make_heading("Wafer Settings"))

        ws_grid = QtWidgets.QFormLayout()
        self.lbl_cylinder = QtWidgets.QLabel("-")
        self.lbl_min_height = QtWidgets.QLabel("-")
        self.lbl_max_chord = QtWidgets.QLabel("-")
        ws_grid.addRow("Cylinder:", self.lbl_cylinder)
        ws_grid.addRow("Min Height:", self.lbl_min_height)
        ws_grid.addRow("Max Chord:", self.lbl_max_chord)
        layout.addLayout(ws_grid)

        # --- Action buttons ---
        action_row = QtWidgets.QHBoxLayout()

        self.btn_build = QtWidgets.QPushButton("Build")
        self.btn_build.setEnabled(False)
        self.btn_build.setMinimumHeight(32)
        font = self.btn_build.font()
        font.setBold(True)
        self.btn_build.setFont(font)
        self.btn_build.clicked.connect(self._run_build)
        action_row.addWidget(self.btn_build)

        self.btn_validate = QtWidgets.QPushButton("Validate")
        self.btn_validate.setEnabled(False)
        self.btn_validate.setMinimumHeight(32)
        self.btn_validate.setToolTip("Check config for errors before building")
        self.btn_validate.clicked.connect(self._on_validate_clicked)
        action_row.addWidget(self.btn_validate)

        self.btn_edit = QtWidgets.QPushButton("Edit Config")
        self.btn_edit.setEnabled(False)
        self.btn_edit.setMinimumHeight(32)
        self.btn_edit.setToolTip("Open the config editor dialog")
        self.btn_edit.clicked.connect(self._open_editor)
        action_row.addWidget(self.btn_edit)

        self.btn_new_config = QtWidgets.QPushButton("New Config")
        self.btn_new_config.setMinimumHeight(32)
        self.btn_new_config.setToolTip("Create a new config from template")
        self.btn_new_config.clicked.connect(self._new_config)
        action_row.addWidget(self.btn_new_config)

        layout.addLayout(action_row)

        # --- Build from Cut List ---
        rebuild_row = QtWidgets.QHBoxLayout()
        self.btn_rebuild = QtWidgets.QPushButton("Build from Cut List")
        self.btn_rebuild.setMinimumHeight(32)
        self.btn_rebuild.setToolTip(
            "Reconstruct 3D structure from an existing cut list .txt file"
        )
        self.btn_rebuild.clicked.connect(self._on_rebuild_from_cut_list)
        rebuild_row.addWidget(self.btn_rebuild)
        layout.addLayout(rebuild_row)

        # --- Wafer visibility range ---
        vis_row = QtWidgets.QHBoxLayout()
        vis_row.addWidget(QtWidgets.QLabel("Wafers:"))
        self.spin_wafer_from = QtWidgets.QSpinBox()
        self.spin_wafer_from.setMinimum(1)
        self.spin_wafer_from.setMaximum(9999)
        self.spin_wafer_from.setValue(1)
        self.spin_wafer_from.setFixedWidth(55)
        self.spin_wafer_from.setToolTip("First wafer to show (1-based)")
        vis_row.addWidget(self.spin_wafer_from)
        vis_row.addWidget(QtWidgets.QLabel("\u2013"))
        self.spin_wafer_to = QtWidgets.QSpinBox()
        self.spin_wafer_to.setMinimum(1)
        self.spin_wafer_to.setMaximum(9999)
        self.spin_wafer_to.setValue(9999)
        self.spin_wafer_to.setFixedWidth(55)
        self.spin_wafer_to.setToolTip("Last wafer to show (1-based; 9999 = all)")
        vis_row.addWidget(self.spin_wafer_to)
        self.btn_apply_range = QtWidgets.QPushButton("Show Range")
        self.btn_apply_range.setMinimumHeight(24)
        self.btn_apply_range.setToolTip(
            "Show only wafers in the specified range (both original and\n"
            "reconstructed); hide all others in the active document.")
        self.btn_apply_range.clicked.connect(self._on_apply_wafer_range)
        vis_row.addWidget(self.btn_apply_range)
        vis_row.addStretch()
        layout.addLayout(vis_row)

        # --- Align Reconstruction to a specific wafer's entry ellipse ---
        align_row = QtWidgets.QHBoxLayout()
        align_lbl = QtWidgets.QLabel("Align wafer:")
        align_row.addWidget(align_lbl)
        self.spin_align_wafer = QtWidgets.QSpinBox()
        self.spin_align_wafer.setMinimum(1)
        self.spin_align_wafer.setMaximum(9999)
        self.spin_align_wafer.setValue(1)
        self.spin_align_wafer.setFixedWidth(60)
        self.spin_align_wafer.setToolTip(
            "Wafer number (1-based) whose entry ellipse will be used for alignment"
        )
        align_row.addWidget(self.spin_align_wafer)
        self.btn_align = QtWidgets.QPushButton("Align Reconstruction")
        self.btn_align.setMinimumHeight(28)
        self.btn_align.setToolTip(
            "Move the reconstructed part so that the specified wafer's entry ellipse "
            "coincides with the original's. Run 'Build from Cut List' first."
        )
        self.btn_align.clicked.connect(self._on_align_reconstruction)
        align_row.addWidget(self.btn_align)
        align_row.addStretch()
        layout.addLayout(align_row)

        # --- Build Variance ---
        variance_row = QtWidgets.QHBoxLayout()
        variance_row.addWidget(QtWidgets.QLabel("Blade \u03c3\u00b0:"))
        self.dspin_sigma_blade = QtWidgets.QDoubleSpinBox()
        self.dspin_sigma_blade.setRange(0.0, 10.0)
        self.dspin_sigma_blade.setSingleStep(0.1)
        self.dspin_sigma_blade.setValue(0.50)
        self.dspin_sigma_blade.setDecimals(2)
        self.dspin_sigma_blade.setFixedWidth(65)
        self.dspin_sigma_blade.setToolTip(
            "1\u03c3 blade tilt uncertainty per cut (degrees)")
        variance_row.addWidget(self.dspin_sigma_blade)
        variance_row.addSpacing(8)
        variance_row.addWidget(QtWidgets.QLabel("Rot \u03c3\u00b0:"))
        self.dspin_sigma_rot = QtWidgets.QDoubleSpinBox()
        self.dspin_sigma_rot.setRange(0.0, 30.0)
        self.dspin_sigma_rot.setSingleStep(0.5)
        self.dspin_sigma_rot.setValue(2.0)
        self.dspin_sigma_rot.setDecimals(1)
        self.dspin_sigma_rot.setFixedWidth(65)
        self.dspin_sigma_rot.setToolTip(
            "1\u03c3 cylinder rotation uncertainty per cut (degrees)")
        variance_row.addWidget(self.dspin_sigma_rot)
        variance_row.addStretch()
        layout.addLayout(variance_row)

        # --- Error Results Table (collapsible) ---
        results_hdr = QtWidgets.QHBoxLayout()
        self.btn_toggle_results = QtWidgets.QPushButton("\u25b6 Error Results")
        self.btn_toggle_results.setCheckable(True)
        self.btn_toggle_results.setChecked(False)
        self.btn_toggle_results.setFlat(True)
        _rf = self.btn_toggle_results.font()
        _rf.setBold(True)
        self.btn_toggle_results.setFont(_rf)
        self.btn_toggle_results.toggled.connect(self._toggle_results_pane)
        results_hdr.addWidget(self.btn_toggle_results)
        results_hdr.addStretch()
        btn_clear_results = QtWidgets.QPushButton("Clear")
        btn_clear_results.setFixedSize(48, 22)
        btn_clear_results.clicked.connect(self._clear_results_table)
        results_hdr.addWidget(btn_clear_results)
        layout.addLayout(results_hdr)

        self._results_pane = QtWidgets.QWidget()
        self._results_pane.setVisible(False)
        rp_layout = QtWidgets.QVBoxLayout(self._results_pane)
        rp_layout.setContentsMargins(0, 0, 0, 0)

        # Column headers: Wfr | Seg | Aln | Ctrd | Axl | Lat | Nrm° | Spin° | Bld°Δ
        _COLS = ["Wfr", "Seg", "Aln", "Ctrd\u200bmm", "Axl\u200bmm",
                 "Lat\u200bmm", "Nrm\u00b0", "Spin\u00b0", "Bld\u00b0\u0394"]
        _COL_TIPS = [
            "Wafer — alignment target wafer number (1-based)",
            "Segment — segment name",
            "Alignment method:\n  6DOF = LCS-exact (position + spin constrained)\n  5DOF = face-only (spin unconstrained)",
            "Centroid distance — total distance between original and\nreconstructed exit face centers (mm)",
            "Axial offset — component along original exit face normal (mm);\npositive = reconstruction exit is too far along travel direction",
            "Lateral offset — in-plane component of centroid offset,\nperpendicular to face normal (mm)",
            "Normal angle — angle between original and reconstructed exit\nface normals (\u00b0); non-zero = blade angle or orientation error",
            "Major-axis spin — signed rotation from reconstructed to original\nmajor axis about face normal (\u00b0); non-zero = Rot\u00b0 error for this cut",
            "Blade \u0394 — (orig \u2212 rec) exit blade angle inferred from face geometry (\u00b0)",
        ]
        self._results_table = QtWidgets.QTableWidget(0, len(_COLS))
        self._results_table.setHorizontalHeaderLabels(_COLS)
        for _ci, _tip in enumerate(_COL_TIPS):
            self._results_table.horizontalHeaderItem(_ci).setToolTip(_tip)
        self._results_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self._results_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setMinimumHeight(120)
        self._results_table.setMaximumHeight(240)
        hh = self._results_table.horizontalHeader()
        hh.setStretchLastSection(True)
        mono = QtGui.QFont("Monospace")
        mono.setStyleHint(QtGui.QFont.TypeWriter)
        mono.setPointSize(8)
        self._results_table.setFont(mono)
        rp_layout.addWidget(self._results_table)

        self._lbl_variance = QtWidgets.QLabel("")
        self._lbl_variance.setWordWrap(True)
        _vf = QtGui.QFont("Monospace")
        _vf.setStyleHint(QtGui.QFont.TypeWriter)
        _vf.setPointSize(8)
        self._lbl_variance.setFont(_vf)
        rp_layout.addWidget(self._lbl_variance)

        layout.addWidget(self._results_pane)

        # --- Status section ---
        layout.addWidget(self._make_heading("Status"))

        self.lbl_status = QtWidgets.QLabel("Select a project to begin")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

        # --- Log section ---
        log_header_row = QtWidgets.QHBoxLayout()
        self.btn_toggle_log = QtWidgets.QPushButton("\u25b6 Log Output")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(False)
        self.btn_toggle_log.setFlat(True)
        font = self.btn_toggle_log.font()
        font.setBold(True)
        self.btn_toggle_log.setFont(font)
        self.btn_toggle_log.toggled.connect(self._toggle_log_pane)
        log_header_row.addWidget(self.btn_toggle_log)
        log_header_row.addStretch()
        layout.addLayout(log_header_row)

        self._log_pane_widget = QtWidgets.QWidget()
        self._log_pane_widget.setVisible(False)
        log_pane_layout = QtWidgets.QVBoxLayout(self._log_pane_widget)
        log_pane_layout.setContentsMargins(0, 0, 0, 0)

        self._txt_log = QtWidgets.QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        self._txt_log.setMinimumHeight(160)
        mono_font = QtGui.QFont("Monospace")
        mono_font.setStyleHint(QtGui.QFont.TypeWriter)
        mono_font.setPointSize(8)
        self._txt_log.setFont(mono_font)
        log_pane_layout.addWidget(self._txt_log)

        log_btn_row = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self._txt_log.clear)
        log_btn_row.addWidget(btn_clear)
        btn_copy_log = QtWidgets.QPushButton("Copy All")
        btn_copy_log.setToolTip("Copy all log text to clipboard")
        btn_copy_log.clicked.connect(self._copy_log)
        log_btn_row.addWidget(btn_copy_log)
        btn_open_log = QtWidgets.QPushButton("Open log file\u2026")
        btn_open_log.clicked.connect(self._open_log_file)
        log_btn_row.addWidget(btn_open_log)
        log_btn_row.addStretch()
        log_pane_layout.addLayout(log_btn_row)

        layout.addWidget(self._log_pane_widget)

        layout.addStretch()

    @staticmethod
    def _make_heading(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        return lbl

    # ------------------------------------------------------------------
    # FreeCAD Task Panel protocol
    # ------------------------------------------------------------------

    def getStandardButtons(self):
        return int(QtWidgets.QDialogButtonBox.Close)

    def reject(self):
        """Close button — dismiss the panel."""
        self._remove_log_handler()
        import FreeCADGui
        FreeCADGui.Control.closeDialog()

    # ------------------------------------------------------------------
    # Dropdown population
    # ------------------------------------------------------------------

    def _populate_dropdown(self):
        from gui.recent_files import load_recent
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("-- select a project --", None)

        recents = load_recent()
        if recents:
            self.combo.insertSeparator(self.combo.count())
            for path in recents:
                self.combo.addItem(os.path.basename(path), path)
            self.combo.insertSeparator(self.combo.count())

        yml_files = sorted(glob.glob(os.path.join(self._get_examples_dir(), "*.yml")))
        for path in yml_files:
            self.combo.addItem(os.path.basename(path), path)

        self.combo.blockSignals(False)
        self.combo.setCurrentIndex(0)
        self._clear_summary()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def _apply_saved_log_level(self):
        try:
            from gui.preferences_dialog import load_prefs, apply_log_level
            apply_log_level(load_prefs().get("log_level", "INFO"))
        except Exception:
            pass

    def _get_examples_dir(self) -> str:
        try:
            from gui.preferences_dialog import load_prefs
            d = load_prefs().get("examples_dir", "").strip()
            if d and os.path.isdir(d):
                return d
        except Exception:
            pass
        return _EXAMPLES_DIR

    def _open_preferences(self):
        from gui.preferences_dialog import PreferencesDialog
        dlg = PreferencesDialog(parent=self.form)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Refresh dropdown in case examples directory changed
            current = self.selected_path
            self._populate_dropdown()
            if current:
                idx = self.combo.findData(current)
                if idx != -1:
                    self.combo.blockSignals(True)
                    self.combo.setCurrentIndex(idx)
                    self.combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Selection change
    # ------------------------------------------------------------------

    def _on_selection_changed(self, index: int):
        path = self.combo.itemData(index)
        if path is None:
            self.selected_path = None
            self.btn_build.setEnabled(False)
            self.btn_validate.setEnabled(False)
            self.btn_edit.setEnabled(False)
            self._clear_summary()
            self._set_status("Select a project to begin")
            return
        self._load_summary(path)

    def _on_combo_context_menu(self, pos):
        from gui.recent_files import load_recent, clear_recent
        menu = QtWidgets.QMenu(self.combo)
        act_clear = menu.addAction("Clear recent files")
        act_clear.setEnabled(bool(load_recent()))
        act = menu.exec_(self.combo.mapToGlobal(pos))
        if act == act_clear:
            clear_recent()
            current = self.selected_path
            self._populate_dropdown()
            if current:
                idx = self.combo.findData(current)
                if idx != -1:
                    self.combo.blockSignals(True)
                    self.combo.setCurrentIndex(idx)
                    self.combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Browse for external YAML
    # ------------------------------------------------------------------

    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.form,
            "Select YAML Project",
            _EXAMPLES_DIR,
            "YAML files (*.yml *.yaml);;All files (*)",
        )
        if not path:
            return

        # Add to dropdown if not already present
        idx = self.combo.findData(path)
        if idx == -1:
            self.combo.addItem(os.path.basename(path), path)
            idx = self.combo.count() - 1
        self.combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Config loading & summary display
    # ------------------------------------------------------------------

    def _load_summary(self, path: str):
        self.selected_path = os.path.realpath(path)
        try:
            loaded = load_config(path, yaml_base_dir=_YAML_BASE_DIR)
            cfg = loaded.data
        except Exception as exc:
            self._clear_summary()
            self._set_status(f"Failed to load config: {exc}", error=True)
            logger.error(f"Config load error: {exc}")
            return

        # Project name
        metadata = cfg.get("metadata", {}) or {}
        self.lbl_name.setText(metadata.get("project_name", "(unnamed)"))

        # Workflow operations
        workflow = cfg.get("workflow", []) or []
        build_segments = [
            op for op in workflow if op.get("operation") == "build_segment"
        ]
        self.lbl_segments.setText(str(len(build_segments)))
        self.lbl_operations.setText(str(len(workflow)))

        # Workflow list
        self.workflow_list.clear()
        for op in workflow:
            op_type = op.get("operation", "?")
            name = op.get("name") or op.get("description", "")
            label = f"{op_type}: {name}" if name else op_type
            self.workflow_list.addItem(label)

        # Wafer settings — use first build_segment's settings
        self._display_wafer_settings(build_segments, cfg)

        from gui.recent_files import push_recent
        push_recent(path)

        self.btn_build.setEnabled(True)
        self.btn_validate.setEnabled(True)
        self.btn_edit.setEnabled(True)
        self._run_validation(loaded)

    def _display_wafer_settings(self, build_segments, cfg):
        if not build_segments:
            self.lbl_cylinder.setText("-")
            self.lbl_min_height.setText("-")
            self.lbl_max_chord.setText("-")
            return

        first_ws = build_segments[0].get("wafer_settings", {}) or {}

        # Check if all segments share the same values
        varies = False
        if len(build_segments) > 1:
            for seg in build_segments[1:]:
                other_ws = seg.get("wafer_settings", {}) or {}
                if other_ws != first_ws:
                    varies = True
                    break

        suffix = " (varies)" if varies else ""
        self.lbl_cylinder.setText(
            str(first_ws.get("cylinder_diameter", "-")) + suffix
        )
        self.lbl_min_height.setText(
            str(first_ws.get("min_height", "-")) + suffix
        )
        self.lbl_max_chord.setText(
            str(first_ws.get("max_chord", "-")) + suffix
        )

    def _clear_summary(self):
        self.lbl_name.setText("-")
        self.lbl_segments.setText("-")
        self.lbl_operations.setText("-")
        self.workflow_list.clear()
        self.lbl_cylinder.setText("-")
        self.lbl_min_height.setText("-")
        self.lbl_max_chord.setText("-")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, loaded_config, *, explicit=False):
        from gui.validator import validate_config
        issues = validate_config(loaded_config)

        errors   = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        if errors:
            self.btn_build.setEnabled(False)
            self._set_status(
                f"Config has {len(errors)} error(s), {len(warnings)} warning(s) — see log",
                error=True)
            self.btn_toggle_log.setChecked(True)
            self._toggle_log_pane(True)
            for issue in issues:
                levelno = logging.ERROR if issue.level == "error" else logging.WARNING
                self._append_log(str(issue), levelno)
        elif warnings:
            self.btn_build.setEnabled(True)
            self._set_status(
                f"Config has {len(warnings)} warning(s) — see log",
                warning=True)
            self.btn_toggle_log.setChecked(True)
            self._toggle_log_pane(True)
            for issue in issues:
                self._append_log(str(issue), logging.WARNING)
        else:
            self.btn_build.setEnabled(True)
            self._set_status("Ready to build")
            if explicit:
                self.btn_toggle_log.setChecked(True)
                self._toggle_log_pane(True)
                self._append_log("Validation passed — no issues found.", logging.INFO)

    def _on_validate_clicked(self):
        if not self.selected_path:
            return
        try:
            loaded = load_config(self.selected_path, yaml_base_dir=_YAML_BASE_DIR)
        except Exception as exc:
            self._set_status(f"Failed to load config: {exc}", error=True)
            return
        self._run_validation(loaded, explicit=True)

    # ------------------------------------------------------------------
    # Config editor
    # ------------------------------------------------------------------

    def _open_editor(self):
        from gui.config_editor import ConfigEditorDialog
        dlg = ConfigEditorDialog(parent=self.form, config_path=self.selected_path)
        dlg.exec_()
        # Refresh summary after dialog closes (config may have changed)
        if self.selected_path:
            self._load_summary(self.selected_path)

    def _new_config(self):
        from gui.config_editor import ConfigEditorDialog
        dlg = ConfigEditorDialog(parent=self.form)
        dlg.exec_()
        # If saved, refresh dropdown so new file appears
        if dlg._config_path:
            idx = self.combo.findData(dlg._config_path)
            if idx == -1:
                self.combo.addItem(os.path.basename(dlg._config_path), dlg._config_path)
                idx = self.combo.count() - 1
            self.combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _run_build(self):
        if not self.selected_path:
            self._set_status("No project selected", error=True)
            return

        import FreeCAD as App
        import FreeCADGui as Gui
        from driver import Driver

        # Disable controls while building
        self.btn_build.setEnabled(False)
        self.combo.setEnabled(False)
        self.btn_browse.setEnabled(False)

        self.btn_toggle_log.setChecked(True)
        self._toggle_log_pane(True)

        self._set_status("Building\u2026")
        self.progress.show()
        self.progress.setRange(0, 0)  # indeterminate spinner
        QtWidgets.QApplication.processEvents()

        try:
            driver = Driver(App, Gui, "curves")
            driver.load_configuration(self.selected_path)
            driver.workflow()

            # Track the generated cuts file so "Build from Cut List" defaults to it
            cuts_file = driver.output_files.get("cuts_file")
            if cuts_file:
                if not os.path.isabs(cuts_file):
                    base_dir = driver.output_files.get("working_directory", "")
                    cuts_file = os.path.join(base_dir, cuts_file)
                self._last_cuts_file = cuts_file

            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self._set_status("Build completed successfully", success=True)
            from gui.recent_files import push_recent
            push_recent(self.selected_path)

        except Exception as exc:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self._set_status(f"Build failed: {exc}", error=True)
            logger.error(f"Build failed:\n{traceback.format_exc()}")

        finally:
            # Re-enable controls
            self.btn_build.setEnabled(True)
            self.combo.setEnabled(True)
            self.btn_browse.setEnabled(True)

    # ------------------------------------------------------------------
    # Build from Cut List
    # ------------------------------------------------------------------

    def _on_apply_wafer_range(self):
        """Show only wafers in [from, to] (1-based); hide all others.

        Matches any document object whose name follows the pattern
        ``Wafer_*_<integer>`` — covers both original (``Wafer_{seg}_{i}``)
        and reconstructed (``Wafer_{seg}_Rec_{i}``) objects.
        """
        import re
        try:
            import FreeCAD as App
        except ImportError:
            self._set_status("FreeCAD not available", error=True)
            return

        doc = App.activeDocument()
        if doc is None:
            self._set_status("No active document", error=True)
            return

        from_1 = self.spin_wafer_from.value()
        to_1   = self.spin_wafer_to.value()
        from_0 = from_1 - 1   # convert to 0-based index
        to_0   = to_1   - 1

        _pat = re.compile(r'^Wafer_.*?_(\d+)$')
        changed = 0
        for obj in doc.Objects:
            m = _pat.match(obj.Name)
            if m is None:
                continue
            wafer_idx = int(m.group(1))
            visible = (from_0 <= wafer_idx <= to_0)
            vo = getattr(obj, 'ViewObject', None)
            if vo is not None:
                try:
                    vo.Visibility = visible
                    changed += 1
                except Exception:
                    pass

        doc.recompute()
        self._set_status(
            f"Showing wafers {from_1}\u2013{to_1} ({changed} objects updated)",
            success=True)

    def _on_rebuild_from_cut_list(self):
        """Open a file dialog and reconstruct 3D geometry from a cut list."""
        start_dir = (
            os.path.dirname(self._last_cuts_file) if self._last_cuts_file
            else os.path.expanduser("~/Documents")
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.form,
            "Select Cut List",
            start_dir,
            "Cut Lists (*.txt);;All Files (*)",
        )
        if not path:
            return

        import FreeCAD as App
        import FreeCADGui as Gui

        self._set_status("Reconstructing\u2026")
        self.progress.show()
        self.progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

        try:
            from cut_list_reconstruction import reconstruct_and_visualize
            rec_result = reconstruct_and_visualize(App, Gui, path)
            # Store wafers per segment for subsequent "Align Reconstruction" use.
            if isinstance(rec_result, dict):
                self._last_rec_wafers = rec_result
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self._set_status(
                f"Reconstruction complete: {os.path.basename(path)}",
                success=True,
            )
        except Exception as exc:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self._set_status(f"Reconstruction failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(
                self.form, "Reconstruction Error", str(exc)
            )

    # ------------------------------------------------------------------
    # Align Reconstruction
    # ------------------------------------------------------------------

    def _on_align_reconstruction(self):
        """Align the reconstructed part to a specified wafer's entry ellipse."""
        import FreeCAD as App

        doc = App.activeDocument()
        if doc is None:
            QtWidgets.QMessageBox.warning(
                self.form, "No Document", "No active FreeCAD document found.")
            return

        if not self._last_rec_wafers:
            QtWidgets.QMessageBox.warning(
                self.form,
                "No Reconstruction",
                "No reconstruction data available.\n"
                "Please run 'Build from Cut List' first.",
            )
            return

        wafer_index = self.spin_align_wafer.value()

        # Discover which segments are available
        seg_names = list(self._last_rec_wafers.keys())
        if not seg_names:
            QtWidgets.QMessageBox.warning(
                self.form, "No Segments", "No reconstructed segments found.")
            return

        self.btn_toggle_log.setChecked(True)
        self._toggle_log_pane(True)
        self._set_status(f"Aligning reconstruction to wafer {wafer_index}\u2026")
        QtWidgets.QApplication.processEvents()

        sigma_blade = self.dspin_sigma_blade.value()
        sigma_rot   = self.dspin_sigma_rot.value()

        errors = []
        successes = []
        try:
            from cut_list_reconstruction import (
                align_reconstruction_to_wafer,
                report_exit_ellipse_discrepancy,
            )
            for seg_name in seg_names:
                rec_wafers = self._last_rec_wafers[seg_name]
                try:
                    new_pl, method = align_reconstruction_to_wafer(
                        doc, seg_name, wafer_index, rec_wafers)
                    successes.append(
                        f"'{seg_name}': aligned wafer {wafer_index} [{method}]")
                    logger.info("Align OK: %s wafer %d — %s", seg_name, wafer_index, method)
                    # Report exit-ellipse discrepancy for the same wafer
                    try:
                        metrics = report_exit_ellipse_discrepancy(
                            doc, seg_name, wafer_index, rec_wafers,
                            sigma_blade_deg=sigma_blade, sigma_rot_deg=sigma_rot)
                        self._add_result_row(
                            seg_name, wafer_index, method, metrics)
                        build_s = metrics.get('build_sigma', {})
                        if build_s:
                            self._update_variance_label(
                                build_s.get('n_wafers', 0),
                                build_s.get('sigma_blade_deg', sigma_blade),
                                build_s.get('sigma_rot_deg', sigma_rot),
                                build_s.get('_radius_in', 1.0))
                    except Exception as rep_exc:
                        logger.warning("Exit report failed for '%s': %s",
                                       seg_name, rep_exc)
                except ValueError as exc:
                    errors.append(f"'{seg_name}': {exc}")
                    logger.error("Align failed for '%s': %s", seg_name, exc)
        except Exception as exc:
            self._set_status(f"Alignment error: {exc}", error=True)
            QtWidgets.QMessageBox.critical(
                self.form, "Alignment Error", str(exc))
            return

        if errors and not successes:
            msg = "Alignment failed:\n\n" + "\n".join(errors)
            self._set_status(f"Alignment failed — see log", error=True)
            QtWidgets.QMessageBox.critical(self.form, "Alignment Error", msg)
        elif errors:
            msg = ("Partial success:\n\n"
                   + "\n".join(successes)
                   + "\n\nFailed:\n"
                   + "\n".join(errors))
            self._set_status("Alignment partially succeeded — see log", warning=True)
            QtWidgets.QMessageBox.warning(self.form, "Alignment Partial", msg)
        else:
            summary = "; ".join(successes)
            self._set_status(
                f"Aligned to wafer {wafer_index}: {summary}", success=True)

    # ------------------------------------------------------------------
    # Error results table helpers
    # ------------------------------------------------------------------

    def _toggle_results_pane(self, checked: bool):
        self.btn_toggle_results.setText(
            "\u25bc Error Results" if checked else "\u25b6 Error Results")
        self._results_pane.setVisible(checked)

    def _clear_results_table(self):
        self._results_table.setRowCount(0)
        self._result_rows.clear()
        self._lbl_variance.setText("")

    def _update_variance_label(self, n_wafers: int, sigma_blade: float,
                               sigma_rot: float, radius_in: float):
        """Update the build variance impact label shown below the results table."""
        import math
        if n_wafers <= 0 or sigma_blade <= 0.0 and sigma_rot <= 0.0:
            self._lbl_variance.setText("")
            return
        sigma_b_rad    = math.radians(sigma_blade)
        sigma_normal   = sigma_blade * math.sqrt(n_wafers)
        sigma_spin     = sigma_rot   * math.sqrt(n_wafers)
        sigma_lat      = radius_in * 25.4 * math.sin(sigma_b_rad) * math.sqrt(n_wafers)
        self._lbl_variance.setText(
            f"Build \u03c3 (1\u03c3, N={n_wafers}):  "
            f"Normal \u00b1{sigma_normal:.1f}\u00b0  "
            f"Spin \u00b1{sigma_spin:.1f}\u00b0  "
            f"Lateral \u00b1{sigma_lat:.2f}\u202fmm")

    def _add_result_row(self, seg_name: str, wafer_index: int,
                        align_method: str, metrics: dict):
        """Append one row to the error results table.

        Columns: Wfr | Seg | Aln | Ctrd mm | Axl mm | Lat mm | Nrm° | Spin° | Bld°Δ
        Rows with large centroid (>1 mm) or large spin (>5°) are highlighted.
        """
        dist   = metrics.get('centroid_distance')
        axial  = metrics.get('centroid_axial')
        lat    = metrics.get('centroid_lateral')
        nrm    = metrics.get('normal_angle_deg')
        spin   = metrics.get('spin_angle_deg')
        b_orig = metrics.get('orig_blade_deg')
        b_rec  = metrics.get('rec_blade_deg')

        b_delta = ((b_orig - b_rec) if b_orig is not None and b_rec is not None
                   else None)

        aln_short = "6DOF" if "6DOF" in align_method else "5DOF"
        seg_short = seg_name[:10]

        def _fmt(v, fmt):
            return fmt.format(v) if v is not None else "-"

        values = [
            str(wafer_index),
            seg_short,
            aln_short,
            _fmt(dist,    "{:.3f}"),
            _fmt(axial,   "{:+.3f}"),
            _fmt(lat,     "{:.3f}"),
            _fmt(nrm,     "{:.2f}"),
            _fmt(spin,    "{:+.2f}"),
            _fmt(b_delta, "{:+.3f}"),
        ]

        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        # Determine row highlight colour
        warn = (dist is not None and dist > 1.0) or (spin is not None and abs(spin) > 5.0)
        crit = (dist is not None and dist > 5.0) or (spin is not None and abs(spin) > 20.0)
        if crit:
            bg = QtGui.QColor(255, 180, 180)   # red tint
        elif warn:
            bg = QtGui.QColor(255, 230, 160)   # amber tint
        else:
            bg = None

        for col, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if bg is not None:
                item.setBackground(bg)
            self._results_table.setItem(row, col, item)

        self._results_table.resizeColumnsToContents()
        self._results_table.scrollToBottom()

        # Auto-show the pane when the first row arrives
        if row == 0:
            self.btn_toggle_results.setChecked(True)
            self._toggle_results_pane(True)

        self._result_rows.append({'wafer': wafer_index, 'seg': seg_name,
                                   'method': align_method, **metrics})

    # ------------------------------------------------------------------
    # Log panel helpers
    # ------------------------------------------------------------------

    _LEVEL_COLORS = {
        logging.ERROR: "#cc0000",
        logging.WARNING: "#b36b00",
        25: "#5500aa",          # COORD level (purple)
        logging.DEBUG: "#888888",
    }

    def _install_log_handler(self):
        from gui.log_handler import QtLogHandler
        self._log_handler = QtLogHandler()
        self._log_handler.log_record_emitted.connect(self._append_log)
        logging.getLogger().addHandler(self._log_handler)

    def _remove_log_handler(self):
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    def _append_log(self, msg: str, levelno: int):
        color = self._LEVEL_COLORS.get(levelno)
        escaped = (msg.replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))
        html = f'<span style="color:{color}">{escaped}</span>' if color else escaped
        self._txt_log.append(html)
        sb = self._txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())
        QtWidgets.QApplication.processEvents()

    def _toggle_log_pane(self, checked: bool):
        self.btn_toggle_log.setText("\u25bc Log Output" if checked else "\u25b6 Log Output")
        self._log_pane_widget.setVisible(checked)

    def _copy_log(self):
        QtWidgets.QApplication.clipboard().setText(self._txt_log.toPlainText())

    def _open_log_file(self):
        import pathlib
        log_path = pathlib.Path.home() / ".plantbuilder" / "plantbuilder.log"
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_path)))

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, *, error: bool = False, success: bool = False,
                    warning: bool = False):
        self.lbl_status.setText(text)
        if error:
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        elif success:
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        elif warning:
            self.lbl_status.setStyleSheet("color: #b36b00; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("")
