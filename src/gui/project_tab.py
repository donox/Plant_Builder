"""
Project tab for the PlantBuilder Config Editor.

Displays and edits: metadata, params table, global settings, output files.
"""
from __future__ import annotations

from typing import Any, Dict

from PySide2 import QtCore, QtWidgets


class ProjectTab(QtWidgets.QWidget):
    """Widget for the 'Project' tab of the config editor."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QtWidgets.QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(inner)
        outer.setWidget(inner)

        wrapper = QtWidgets.QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(outer)

        # Use a two-column layout: left = metadata + global, right = params + output
        columns = QtWidgets.QHBoxLayout()
        left_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()

        # -- Metadata --
        meta_grp = QtWidgets.QGroupBox("Metadata")
        meta_form = QtWidgets.QFormLayout(meta_grp)
        self.txt_name = QtWidgets.QLineEdit()
        self.txt_version = QtWidgets.QLineEdit()
        self.txt_created = QtWidgets.QLineEdit()
        self.txt_description = QtWidgets.QLineEdit()
        meta_form.addRow("Name:", self.txt_name)
        meta_form.addRow("Version:", self.txt_version)
        meta_form.addRow("Created:", self.txt_created)
        meta_form.addRow("Description:", self.txt_description)
        left_col.addWidget(meta_grp)

        # -- Global Settings --
        gs_grp = QtWidgets.QGroupBox("Global Settings")
        gs_form = QtWidgets.QFormLayout(gs_grp)

        self.cmb_workflow_mode = QtWidgets.QComboBox()
        self.cmb_workflow_mode.addItems(["(none)", "first_pass", "second_pass"])
        gs_form.addRow("Workflow mode:", self.cmb_workflow_mode)

        self.chk_print_cuts = QtWidgets.QCheckBox()
        gs_form.addRow("Print cuts:", self.chk_print_cuts)

        self.chk_show_lcs = QtWidgets.QCheckBox()
        gs_form.addRow("Show LCS:", self.chk_show_lcs)

        self.chk_include_defs = QtWidgets.QCheckBox()
        gs_form.addRow("Include cut list definitions:", self.chk_include_defs)

        left_col.addWidget(gs_grp)
        left_col.addStretch()

        # -- Parameters table --
        params_grp = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_grp)
        self.tbl_params = QtWidgets.QTableWidget(0, 2)
        self.tbl_params.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_params.horizontalHeader().setStretchLastSection(True)
        self.tbl_params.setMinimumHeight(180)
        params_layout.addWidget(self.tbl_params)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_param = QtWidgets.QPushButton("+ Add")
        self.btn_rm_param = QtWidgets.QPushButton("- Remove")
        btn_row.addWidget(self.btn_add_param)
        btn_row.addWidget(self.btn_rm_param)
        btn_row.addStretch()
        params_layout.addLayout(btn_row)
        right_col.addWidget(params_grp)

        self.btn_add_param.clicked.connect(self._add_param_row)
        self.btn_rm_param.clicked.connect(self._remove_param_row)

        # -- Output Files --
        out_grp = QtWidgets.QGroupBox("Output Files")
        out_form = QtWidgets.QFormLayout(out_grp)

        self.txt_working_dir = QtWidgets.QLineEdit()
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self.txt_working_dir)
        self.btn_browse_dir = QtWidgets.QPushButton("Browse\u2026")
        self.btn_browse_dir.clicked.connect(self._browse_working_dir)
        dir_row.addWidget(self.btn_browse_dir)
        out_form.addRow("Working dir:", dir_row)

        self.txt_cuts_file = QtWidgets.QLineEdit()
        out_form.addRow("Cuts file:", self.txt_cuts_file)

        self.txt_place_file = QtWidgets.QLineEdit()
        out_form.addRow("Place file:", self.txt_place_file)

        self.txt_trace_file = QtWidgets.QLineEdit()
        out_form.addRow("Trace file:", self.txt_trace_file)

        right_col.addWidget(out_grp)
        right_col.addStretch()

        columns.addLayout(left_col)
        columns.addLayout(right_col)
        main_layout.addLayout(columns)

    # ------------------------------------------------------------------
    # Param table helpers
    # ------------------------------------------------------------------

    def _add_param_row(self, key: str = "", value: str = ""):
        row = self.tbl_params.rowCount()
        self.tbl_params.insertRow(row)
        self.tbl_params.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
        self.tbl_params.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    def _remove_param_row(self):
        row = self.tbl_params.currentRow()
        if row >= 0:
            self.tbl_params.removeRow(row)

    def _browse_working_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Working Directory", self.txt_working_dir.text()
        )
        if path:
            self.txt_working_dir.setText(path)

    # ------------------------------------------------------------------
    # populate / collect
    # ------------------------------------------------------------------

    def populate(self, raw_config: dict, resolved_config: dict):
        """Fill all fields from the raw and resolved config dicts."""
        # -- Metadata (use resolved values since they may come from params) --
        raw_meta = raw_config.get("metadata", {}) or {}
        res_meta = resolved_config.get("metadata", {}) or {}
        self.txt_name.setText(str(res_meta.get("project_name", "")))
        self.txt_version.setText(str(res_meta.get("version", "")))
        self.txt_created.setText(str(res_meta.get("created", "")))
        self.txt_description.setText(str(res_meta.get("description", "")))

        # Store raw metadata refs for roundtrip
        self._raw_metadata = raw_meta

        # -- Params --
        self.tbl_params.setRowCount(0)
        params = raw_config.get("params", {}) or {}
        self._raw_param_types: Dict[str, Any] = {}
        for key, val in params.items():
            self._raw_param_types[key] = val  # preserve original YAML type
            self._add_param_row(key, val)

        # -- Global Settings --
        gs = resolved_config.get("global_settings", {}) or {}
        wm = gs.get("workflow_mode")
        if wm in ("first_pass", "second_pass"):
            self.cmb_workflow_mode.setCurrentText(wm)
        else:
            self.cmb_workflow_mode.setCurrentIndex(0)

        self.chk_print_cuts.setChecked(bool(gs.get("print_cuts", False)))
        self.chk_show_lcs.setChecked(bool(gs.get("show_lcs", False)))
        self.chk_include_defs.setChecked(bool(gs.get("include_cut_list_definitions", False)))

        # -- Output Files --
        raw_of = raw_config.get("output_files", {}) or {}
        res_of = resolved_config.get("output_files", {}) or {}
        self._raw_output_files = raw_of
        self.txt_working_dir.setText(str(res_of.get("working_directory", "")))
        self.txt_cuts_file.setText(str(res_of.get("cuts_file", "")))
        self.txt_place_file.setText(str(res_of.get("place_file", "")))
        self.txt_trace_file.setText(str(res_of.get("trace_file", "")))

    def collect(self) -> Dict[str, Any]:
        """Gather all fields back into config sub-dicts."""
        # Params — preserve original YAML types when value text hasn't changed
        raw_types = getattr(self, "_raw_param_types", {})
        params: Dict[str, Any] = {}
        for row in range(self.tbl_params.rowCount()):
            key_item = self.tbl_params.item(row, 0)
            val_item = self.tbl_params.item(row, 1)
            if key_item and key_item.text().strip():
                key = key_item.text().strip()
                val_str = val_item.text() if val_item else ""
                # If the text matches the original raw value, keep original type
                if key in raw_types and str(raw_types[key]) == val_str:
                    params[key] = raw_types[key]
                else:
                    params[key] = _auto_type(val_str)

        # Metadata — preserve original ${param} refs where values match
        raw_meta = getattr(self, "_raw_metadata", {}) or {}
        metadata = self._collect_with_refs(
            raw_meta,
            {
                "project_name": self.txt_name.text(),
                "version": self.txt_version.text(),
                "created": self.txt_created.text(),
                "description": self.txt_description.text(),
            },
            params,
        )

        # Global settings
        wm_text = self.cmb_workflow_mode.currentText()
        workflow_mode = None if wm_text == "(none)" else wm_text
        global_settings = {
            "workflow_mode": workflow_mode,
            "print_cuts": self.chk_print_cuts.isChecked(),
            "show_lcs": self.chk_show_lcs.isChecked(),
            "include_cut_list_definitions": self.chk_include_defs.isChecked(),
        }

        # Output files — preserve refs
        raw_of = getattr(self, "_raw_output_files", {}) or {}
        output_files = self._collect_with_refs(
            raw_of,
            {
                "working_directory": self.txt_working_dir.text(),
                "cuts_file": self.txt_cuts_file.text(),
                "place_file": self.txt_place_file.text(),
                "trace_file": self.txt_trace_file.text(),
            },
            params,
        )

        return {
            "params": params,
            "metadata": metadata,
            "global_settings": global_settings,
            "output_files": output_files,
        }

    @staticmethod
    def _collect_with_refs(raw_section: dict, current_values: dict, params: dict) -> dict:
        """Preserve ${param} references when the resolved value hasn't changed."""
        import re
        out: dict = {}
        for key, current_val in current_values.items():
            raw_val = raw_section.get(key)
            if isinstance(raw_val, str):
                m = re.match(r"^\$\{([^}]+)\}$", raw_val.strip())
                if m:
                    param_name = m.group(1)
                    resolved = params.get(param_name)
                    if resolved is not None and str(resolved) == str(current_val):
                        out[key] = raw_val  # keep the ${ref}
                        continue
            out[key] = current_val
        return out


def _auto_type(s: str) -> Any:
    """Try to convert string value to int/float/bool, fallback to str."""
    s = s.strip()
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
