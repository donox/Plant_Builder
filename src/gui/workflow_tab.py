"""
Workflow tab for the PlantBuilder Config Editor.

Split horizontally: operation list on the left, detail editor on the right.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide2 import QtCore, QtWidgets

from gui.operation_editors import (
    OPERATION_TYPES,
    create_editor,
)


def _op_label(op: dict) -> str:
    """Build a short label for an operation dict."""
    op_type = op.get("operation", "?")
    name = op.get("name") or op.get("description", "")
    return f"{op_type}: {name}" if name else op_type


class WorkflowTab(QtWidgets.QWidget):
    """Widget for the 'Workflow' tab of the config editor."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # Parallel lists: _operations[i] is the raw dict, _resolved[i] is resolved
        self._operations: List[dict] = []
        self._resolved: List[dict] = []
        # Cache of editor widgets keyed by list-widget row index
        self._editors: Dict[int, QtWidgets.QWidget] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- Left side: operation list + buttons ---
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QtWidgets.QLabel("Operations"))
        self.lst_ops = QtWidgets.QListWidget()
        left_layout.addWidget(self.lst_ops)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+")
        self.btn_add.setToolTip("Add operation")
        self.btn_add.setFixedWidth(30)
        self.btn_remove = QtWidgets.QPushButton("\u2212")  # minus sign
        self.btn_remove.setToolTip("Remove selected operation")
        self.btn_remove.setFixedWidth(30)
        self.btn_up = QtWidgets.QPushButton("\u2191")
        self.btn_up.setToolTip("Move up")
        self.btn_up.setFixedWidth(30)
        self.btn_down = QtWidgets.QPushButton("\u2193")
        self.btn_down.setToolTip("Move down")
        self.btn_down.setFixedWidth(30)

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_up)
        btn_row.addWidget(self.btn_down)
        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # --- Right side: detail editor area ---
        self.editor_area = QtWidgets.QStackedWidget()
        # Index 0 = placeholder
        self._placeholder = QtWidgets.QLabel("Select an operation to edit")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.editor_area.addWidget(self._placeholder)
        splitter.addWidget(self.editor_area)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Connections
        self.lst_ops.currentRowChanged.connect(self._on_selection_changed)
        self.btn_add.clicked.connect(self._show_add_menu)
        self.btn_remove.clicked.connect(self._remove_operation)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)

    # ------------------------------------------------------------------
    # populate / collect
    # ------------------------------------------------------------------

    def populate(self, workflow: list, resolved_workflow: list):
        self._operations = [dict(op) for op in workflow]
        self._resolved = [dict(op) for op in resolved_workflow]
        self._editors.clear()

        # Remove any stacked editors beyond placeholder
        while self.editor_area.count() > 1:
            w = self.editor_area.widget(1)
            self.editor_area.removeWidget(w)
            w.deleteLater()

        self.lst_ops.blockSignals(True)
        self.lst_ops.clear()
        for op in self._operations:
            self.lst_ops.addItem(_op_label(op))
        self.lst_ops.blockSignals(False)

        if self._operations:
            self.lst_ops.setCurrentRow(0)
        else:
            self.editor_area.setCurrentIndex(0)

    def collect(self) -> List[dict]:
        """Collect data from all editors back into a workflow list."""
        # First, save the currently-visible editor back into _operations
        self._save_current_editor()
        return list(self._operations)

    # ------------------------------------------------------------------
    # Editor management
    # ------------------------------------------------------------------

    def _save_current_editor(self):
        """Persist the active editor's data back into _operations."""
        row = self.lst_ops.currentRow()
        editor = self._editors.get(row)
        if editor is not None and hasattr(editor, "collect"):
            self._operations[row] = editor.collect()
            self.lst_ops.item(row).setText(_op_label(self._operations[row]))

    def _on_selection_changed(self, row: int):
        if row < 0 or row >= len(self._operations):
            self.editor_area.setCurrentIndex(0)
            return

        # Save previous editor before switching
        for cached_row, cached_editor in self._editors.items():
            if cached_row != row and hasattr(cached_editor, "collect"):
                if cached_row < len(self._operations):
                    self._operations[cached_row] = cached_editor.collect()
                    self.lst_ops.item(cached_row).setText(
                        _op_label(self._operations[cached_row])
                    )

        # Get or create editor for this row
        editor = self._editors.get(row)
        if editor is None:
            op = self._operations[row]
            res = self._resolved[row] if row < len(self._resolved) else op
            op_type = op.get("operation", "")
            editor = create_editor(op_type)
            if hasattr(editor, "populate"):
                editor.populate(op, res)
            self._editors[row] = editor
            self.editor_area.addWidget(editor)

        self.editor_area.setCurrentWidget(editor)

    # ------------------------------------------------------------------
    # Add / Remove / Reorder
    # ------------------------------------------------------------------

    def _show_add_menu(self):
        menu = QtWidgets.QMenu(self)
        for op_type in OPERATION_TYPES:
            menu.addAction(op_type, lambda t=op_type: self._add_operation(t))
        menu.exec_(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))

    def _add_operation(self, op_type: str):
        new_op = {"operation": op_type}
        if op_type == "build_segment":
            new_op.update({
                "segment_type": "curve_follower",
                "name": "",
                "curve_spec": {"type": "linear", "parameters": {}},
                "wafer_settings": {},
                "segment_settings": {},
            })
        insert_at = self.lst_ops.currentRow() + 1
        if insert_at <= 0:
            insert_at = len(self._operations)

        self._operations.insert(insert_at, new_op)
        self._resolved.insert(insert_at, dict(new_op))
        self._rebuild_list(select=insert_at)

    def _remove_operation(self):
        row = self.lst_ops.currentRow()
        if row < 0:
            return
        self._operations.pop(row)
        if row < len(self._resolved):
            self._resolved.pop(row)
        # Discard cached editor
        editor = self._editors.pop(row, None)
        if editor is not None:
            self.editor_area.removeWidget(editor)
            editor.deleteLater()
        self._rebuild_list(select=min(row, len(self._operations) - 1))

    def _move_up(self):
        row = self.lst_ops.currentRow()
        if row <= 0:
            return
        self._save_current_editor()
        self._operations[row - 1], self._operations[row] = (
            self._operations[row],
            self._operations[row - 1],
        )
        if row < len(self._resolved) and row - 1 < len(self._resolved):
            self._resolved[row - 1], self._resolved[row] = (
                self._resolved[row],
                self._resolved[row - 1],
            )
        self._rebuild_list(select=row - 1)

    def _move_down(self):
        row = self.lst_ops.currentRow()
        if row < 0 or row >= len(self._operations) - 1:
            return
        self._save_current_editor()
        self._operations[row], self._operations[row + 1] = (
            self._operations[row + 1],
            self._operations[row],
        )
        if row + 1 < len(self._resolved) and row < len(self._resolved):
            self._resolved[row], self._resolved[row + 1] = (
                self._resolved[row + 1],
                self._resolved[row],
            )
        self._rebuild_list(select=row + 1)

    def _rebuild_list(self, select: int = -1):
        """Rebuild the QListWidget and clear cached editors."""
        # Clear editor cache — editors will be recreated on selection
        for editor in self._editors.values():
            self.editor_area.removeWidget(editor)
            editor.deleteLater()
        self._editors.clear()

        self.lst_ops.blockSignals(True)
        self.lst_ops.clear()
        for op in self._operations:
            self.lst_ops.addItem(_op_label(op))
        self.lst_ops.blockSignals(False)

        if 0 <= select < len(self._operations):
            self.lst_ops.setCurrentRow(select)
        elif self._operations:
            self.lst_ops.setCurrentRow(0)
        else:
            self.editor_area.setCurrentIndex(0)
