"""
PlantBuilder Config Editor — main QDialog.

Provides a tabbed editor for creating and modifying YAML project configs.
Preserves ${param} variable references when saving.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml
from PySide2 import QtCore, QtWidgets

from gui.project_tab import ProjectTab
from gui.workflow_tab import WorkflowTab


# Directory containing packaged base YAML files
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YAML_BASE_DIR = os.path.join(_SRC_DIR, "yaml_files", "base")
_EXAMPLES_DIR = os.path.join(_SRC_DIR, "yaml_files", "examples")

# Template for brand-new configs
_NEW_CONFIG_TEMPLATE = {
    "include": ["base/defaults.yml"],
    "params": {
        "project_name": "New Project",
        "out_dir": "",
        "cuts_file": "cutting_list.txt",
        "place_file": "placement_list.txt",
        "trace_file": "build_trace.log",
    },
    "metadata": {
        "project_name": "${project_name}",
    },
    "output_files": {
        "working_directory": "${out_dir}",
        "cuts_file": "${cuts_file}",
        "place_file": "${place_file}",
        "trace_file": "${trace_file}",
    },
    "workflow": [
        {
            "operation": "remove_objects",
            "description": "Clean up existing objects",
        },
    ],
}


class ConfigEditorDialog(QtWidgets.QDialog):
    """Full config editor dialog launched from the task panel."""

    def __init__(self, parent: QtWidgets.QWidget = None, config_path: str = None):
        super().__init__(parent)
        self.setWindowTitle("PlantBuilder Config Editor")
        self.resize(900, 700)
        self.setMinimumSize(700, 500)

        self._config_path: Optional[str] = config_path
        self._raw_config: dict = {}
        self._resolved_config: dict = {}
        self._include_list: list = []

        self._build_ui()

        if config_path:
            self._load_config(config_path)
        else:
            self._new_config()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # -- Path display --
        self.lbl_path = QtWidgets.QLabel("(new config)")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.lbl_path)

        # -- Tabs --
        self.tabs = QtWidgets.QTabWidget()
        self.project_tab = ProjectTab()
        self.workflow_tab = WorkflowTab()
        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.workflow_tab, "Workflow")
        layout.addWidget(self.tabs)

        # -- Button bar --
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_new = QtWidgets.QPushButton("New")
        self.btn_new.clicked.connect(self._new_config)
        btn_layout.addWidget(self.btn_new)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.clicked.connect(self._save)
        btn_layout.addWidget(self.btn_save)

        self.btn_save_as = QtWidgets.QPushButton("Save As\u2026")
        self.btn_save_as.clicked.connect(self._save_as)
        btn_layout.addWidget(self.btn_save_as)

        btn_layout.addStretch()

        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Load / New
    # ------------------------------------------------------------------

    def _load_config(self, path: str):
        """Load an existing config file and populate both tabs."""
        from config.loader import load_config, _read_yaml

        self._config_path = path
        self.lbl_path.setText(path)

        try:
            # 1) Read raw YAML (unresolved, preserves ${param} refs)
            self._raw_config = _read_yaml(path)

            # 2) Load resolved config via full loader (with includes + param resolution)
            loaded = load_config(path, yaml_base_dir=_YAML_BASE_DIR)
            self._resolved_config = loaded.data

            # 3) Remember the include list for roundtrip
            self._include_list = self._raw_config.get("include", []) or []

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Failed to load config:\n{exc}"
            )
            self._raw_config = {}
            self._resolved_config = {}
            return

        self._populate_tabs()

    def _new_config(self):
        """Start a new config from the built-in template."""
        import copy
        self._config_path = None
        self.lbl_path.setText("(new config)")
        self._raw_config = copy.deepcopy(_NEW_CONFIG_TEMPLATE)
        self._resolved_config = copy.deepcopy(_NEW_CONFIG_TEMPLATE)
        self._include_list = list(_NEW_CONFIG_TEMPLATE.get("include", []))
        self._populate_tabs()

    def _populate_tabs(self):
        self.project_tab.populate(self._raw_config, self._resolved_config)

        raw_wf = self._raw_config.get("workflow", []) or []
        res_wf = self._resolved_config.get("workflow", []) or []
        self.workflow_tab.populate(raw_wf, res_wf)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _collect_config(self) -> dict:
        """Gather all data from tabs into a single config dict."""
        proj = self.project_tab.collect()
        workflow = self.workflow_tab.collect()

        config: dict = {}

        # Preserve includes
        if self._include_list:
            config["include"] = list(self._include_list)

        if proj.get("params"):
            config["params"] = proj["params"]
        if proj.get("metadata"):
            config["metadata"] = proj["metadata"]

        # Global settings: start from raw (what was explicitly in the file),
        # then overlay user's UI changes — only write values that differ
        # from resolved include defaults so we don't bleed include values.
        raw_gs = dict(self._raw_config.get("global_settings", {}) or {})
        res_gs = self._resolved_config.get("global_settings", {}) or {}
        user_gs = proj.get("global_settings", {})
        for key, user_val in user_gs.items():
            if key in raw_gs:
                raw_gs[key] = user_val  # always keep explicitly-set keys
            elif user_val != res_gs.get(key):
                raw_gs[key] = user_val  # user changed from include default
        if raw_gs:
            config["global_settings"] = raw_gs

        if proj.get("output_files"):
            config["output_files"] = proj["output_files"]

        config["workflow"] = workflow

        return config

    def _save(self):
        """Save to the current path, or prompt Save As if no path."""
        if not self._config_path:
            self._save_as()
            return

        self._write_yaml(self._config_path)

    def _save_as(self):
        start_dir = (
            os.path.dirname(self._config_path) if self._config_path else _EXAMPLES_DIR
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Config As",
            start_dir,
            "YAML files (*.yml *.yaml);;All files (*)",
        )
        if not path:
            return
        if not (path.endswith(".yml") or path.endswith(".yaml")):
            path += ".yml"

        self._config_path = path
        self.lbl_path.setText(path)
        self._write_yaml(path)

    def _write_yaml(self, path: str):
        config = self._collect_config()
        try:
            with open(path, "w") as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Config saved to:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Failed to save:\n{exc}"
            )
