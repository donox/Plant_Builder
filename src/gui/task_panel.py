"""
PlantBuilder Task Panel — FreeCAD right-side panel for picking a YAML
project, viewing its summary, and running the build workflow.
"""
from __future__ import annotations

import glob
import os
import traceback

from PySide2 import QtCore, QtGui, QtWidgets

from config.loader import load_config
from core.logging_setup import get_logger

logger = get_logger(__name__)

# Directories relative to this file
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.join(_SRC_DIR, "yaml_files", "examples")
_YAML_BASE_DIR = os.path.join(_SRC_DIR, "yaml_files", "base")


class PlantBuilderPanel:
    """FreeCAD Task Panel for PlantBuilder."""

    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("PlantBuilder")
        self.selected_path: str | None = None
        self._build_ui()
        self._populate_dropdown()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)

        # --- Project File section ---
        layout.addWidget(self._make_heading("Project File"))

        file_row = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
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
        import FreeCADGui
        FreeCADGui.Control.closeDialog()

    # ------------------------------------------------------------------
    # Dropdown population
    # ------------------------------------------------------------------

    def _populate_dropdown(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("-- select a project --", None)

        yml_files = sorted(glob.glob(os.path.join(_EXAMPLES_DIR, "*.yml")))
        for path in yml_files:
            self.combo.addItem(os.path.basename(path), path)

        self.combo.blockSignals(False)
        self.combo.setCurrentIndex(0)
        self._clear_summary()

    # ------------------------------------------------------------------
    # Selection change
    # ------------------------------------------------------------------

    def _on_selection_changed(self, index: int):
        path = self.combo.itemData(index)
        if path is None:
            self.selected_path = None
            self.btn_build.setEnabled(False)
            self.btn_edit.setEnabled(False)
            self._clear_summary()
            self._set_status("Select a project to begin")
            return
        self._load_summary(path)

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
        self.selected_path = path
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

        self.btn_build.setEnabled(True)
        self.btn_edit.setEnabled(True)
        self._set_status("Ready to build")

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

        self._set_status("Building\u2026")
        self.progress.show()
        self.progress.setRange(0, 0)  # indeterminate spinner
        QtWidgets.QApplication.processEvents()

        try:
            driver = Driver(App, Gui, "curves")
            driver.load_configuration(self.selected_path)
            driver.workflow()

            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self._set_status("Build completed successfully", success=True)

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
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, *, error: bool = False, success: bool = False):
        self.lbl_status.setText(text)
        if error:
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        elif success:
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("")
