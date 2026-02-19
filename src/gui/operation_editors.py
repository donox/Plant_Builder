"""
Per-operation-type editor widgets for the PlantBuilder Config Editor.

Each editor is a QWidget subclass with populate(op_dict, resolved_dict) and
collect() -> dict methods.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from PySide2 import QtCore, QtWidgets

# ---------------------------------------------------------------------------
# Curve parameter definitions
# Each entry: (name, type_or_choices, default, label)
#   type_or_choices: float, int, bool, or list of str for combo box
# ---------------------------------------------------------------------------

CURVE_PARAMS: Dict[str, List[Tuple[str, Any, Any, str]]] = {
    "linear": [
        ("length", float, 100.0, "Length"),
        ("points", int, 100, "Points"),
    ],
    "circle": [
        ("radius", float, 10.0, "Radius"),
        ("points", int, 100, "Points"),
        ("plane", ["xy", "xz", "yz"], "xy", "Plane"),
    ],
    "helical": [
        ("radius", float, 10.0, "Radius"),
        ("pitch", float, 2.5, "Pitch per turn"),
        ("turns", float, 4.0, "Turns"),
        ("points", int, 100, "Points"),
        ("start_at_origin", bool, True, "Start at origin"),
    ],
    "sinusoidal": [
        ("length", float, 50.0, "Length"),
        ("amplitude", float, 5.0, "Amplitude"),
        ("frequency", float, 2.0, "Frequency"),
        ("points", int, 100, "Points"),
        ("axis", ["x", "y", "z"], "x", "Axis"),
    ],
    "spiral": [
        ("max_radius", float, 10.0, "Max radius"),
        ("min_radius", float, 5.0, "Min radius"),
        ("max_height", float, 10.0, "Max height"),
        ("turns", float, 2.0, "Turns"),
        ("points", int, 100, "Points"),
        ("plane", ["xy", "xz", "yz"], "xy", "Plane"),
    ],
    "figure_eight": [
        ("radius", float, 10.0, "Radius"),
        ("points", int, 100, "Points"),
        ("plane", ["xy", "xz", "yz"], "xy", "Plane"),
    ],
    "trefoil": [
        ("major_radius", float, 4.0, "Major radius"),
        ("tube_radius", float, 2.5, "Tube radius"),
        ("p", int, 2, "p (longitudinal wraps)"),
        ("q", int, 3, "q (meridional wraps)"),
        ("points", int, 150, "Points"),
        ("smooth_factor", float, 0.75, "Smooth factor"),
        ("jitter", float, 0.0, "Jitter"),
        ("optimize_spacing", bool, True, "Optimize spacing"),
        ("scale_z", float, 1.0, "Scale Z"),
    ],
    "overhand_knot": [
        ("scale", float, 1.0, "Scale"),
        ("points", int, 100, "Points"),
        ("increment", float, 1.0, "Increment"),
    ],
}

# Curve types that users can pick when creating/editing build_segment
EDITABLE_CURVE_TYPES = sorted(CURVE_PARAMS.keys())

_PARAM_RE = re.compile(r"^\$\{([^}]+)\}$")


def _is_param_ref(value: Any) -> Optional[str]:
    """Return the param name if *value* is a '${name}' reference, else None."""
    if isinstance(value, str):
        m = _PARAM_RE.match(value.strip())
        if m:
            return m.group(1)
    return None


# =========================================================================
# Reusable sub-widgets
# =========================================================================

class WaferSettingsForm(QtWidgets.QGroupBox):
    """Reusable form for wafer_settings — used by BuildSegment and CloseCurve."""

    _FIELDS = [
        ("cylinder_diameter", float, 2.0, "Cylinder diameter"),
        ("profile_density", float, 0.89, "Profile density"),
        ("min_height", float, 0.1, "Min height"),
        ("max_chord", float, 0.5, "Max chord"),
        ("min_inner_chord", float, 0.25, "Min inner chord"),
        ("max_wafer_count", int, 0, "Max wafer count (0=unlimited)"),
    ]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Wafer Settings", parent)
        form = QtWidgets.QFormLayout(self)
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._param_refs: Dict[str, str] = {}
        self._raw_data: dict = {}

        for name, typ, default, label in self._FIELDS:
            widget = self._make_spin(typ, default)
            form.addRow(label + ":", widget)
            self._widgets[name] = widget

    @staticmethod
    def _make_spin(typ, default):
        if typ is float:
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(0.0, 99999.0)
            w.setValue(float(default))
        else:
            w = QtWidgets.QSpinBox()
            w.setRange(0, 99999)
            w.setValue(int(default))
        return w

    def populate(self, raw: dict, resolved: dict):
        self._raw_data = dict(raw)
        self._param_refs.clear()
        for name, typ, default, _label in self._FIELDS:
            raw_val = raw.get(name)
            res_val = resolved.get(name, default)
            ref = _is_param_ref(raw_val)
            if ref:
                self._param_refs[name] = ref
                val = res_val
            else:
                val = raw_val if raw_val is not None else res_val
            w = self._widgets[name]
            if name == "max_wafer_count":
                w.setValue(int(val) if val is not None else 0)
            elif typ is float:
                w.setValue(float(val))
            else:
                w.setValue(int(val))

    def collect(self) -> dict:
        out = dict(self._raw_data)  # preserve unknown keys
        for name, typ, _default, _label in self._FIELDS:
            w = self._widgets[name]
            val = w.value()
            ref = self._param_refs.get(name)
            if ref is not None:
                out[name] = "${" + ref + "}"
            elif name == "max_wafer_count":
                if val > 0:
                    out[name] = int(val)
                elif name in out:
                    pass  # keep original value (e.g. null)
                # else: omit — wasn't in raw and user left at 0
            elif typ is float:
                out[name] = float(val)
            else:
                out[name] = int(val)
        return out


class SegmentSettingsForm(QtWidgets.QGroupBox):
    """Reusable form for segment_settings — used by BuildSegment and CloseCurve."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Segment Settings", parent)
        self._raw_data: dict = {}
        form = QtWidgets.QFormLayout(self)

        self.chk_show_lcs = QtWidgets.QCheckBox()
        form.addRow("Show LCS:", self.chk_show_lcs)

        self.chk_build_segment = QtWidgets.QCheckBox()
        self.chk_build_segment.setChecked(True)
        form.addRow("Build segment:", self.chk_build_segment)

        self.spin_rotate = QtWidgets.QDoubleSpinBox()
        self.spin_rotate.setRange(-360.0, 360.0)
        self.spin_rotate.setSuffix("\u00b0")
        self.spin_rotate.setDecimals(1)
        form.addRow("Rotate segment:", self.spin_rotate)

    def populate(self, d: dict):
        self._raw_data = dict(d)
        self.chk_show_lcs.setChecked(bool(d.get("show_lcs", False)))
        self.chk_build_segment.setChecked(bool(d.get("build_segment", True)))
        self.spin_rotate.setValue(float(d.get("rotate_segment", 0.0)))

    def collect(self) -> dict:
        out = dict(self._raw_data)  # preserve unknown keys
        out["show_lcs"] = self.chk_show_lcs.isChecked()
        out["build_segment"] = self.chk_build_segment.isChecked()
        out["rotate_segment"] = self.spin_rotate.value()
        return out


# =========================================================================
# Dynamic curve-parameter form builder
# =========================================================================

class CurveParameterForm(QtWidgets.QWidget):
    """Dynamically built form for a specific curve type's parameters."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._layout = QtWidgets.QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._param_refs: Dict[str, str] = {}
        self._curve_type: str = ""

    def rebuild(self, curve_type: str, raw_params: dict = None, resolved_params: dict = None):
        """Clear and rebuild the form for *curve_type*."""
        self._clear()
        self._curve_type = curve_type
        self._param_refs.clear()
        raw_params = raw_params or {}
        resolved_params = resolved_params or {}

        defs = CURVE_PARAMS.get(curve_type, [])
        for name, typ_or_choices, default, label in defs:
            raw_val = raw_params.get(name)
            res_val = resolved_params.get(name, default)
            ref = _is_param_ref(raw_val)
            if ref:
                self._param_refs[name] = ref
                val = res_val
            else:
                val = raw_val if raw_val is not None else res_val

            widget = self._make_widget(typ_or_choices, val)
            self._layout.addRow(label + ":", widget)
            self._widgets[name] = widget

    def collect(self) -> dict:
        out: dict = {}
        defs = CURVE_PARAMS.get(self._curve_type, [])
        for name, typ_or_choices, _default, _label in defs:
            w = self._widgets.get(name)
            if w is None:
                continue
            ref = self._param_refs.get(name)
            if ref is not None:
                out[name] = "${" + ref + "}"
            elif isinstance(typ_or_choices, list):
                out[name] = w.currentText()
            elif typ_or_choices is bool:
                out[name] = w.isChecked()
            elif typ_or_choices is int:
                out[name] = w.value()
            else:
                out[name] = w.value()
        return out

    # -- helpers --

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    @staticmethod
    def _make_widget(typ_or_choices, value):
        if isinstance(typ_or_choices, list):
            w = QtWidgets.QComboBox()
            w.addItems(typ_or_choices)
            idx = w.findText(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w
        if typ_or_choices is bool:
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            return w
        if typ_or_choices is int:
            w = QtWidgets.QSpinBox()
            w.setRange(0, 99999)
            w.setValue(int(value))
            return w
        # float
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(4)
        w.setRange(-99999.0, 99999.0)
        w.setValue(float(value))
        return w


# =========================================================================
# Per-operation editors
# =========================================================================

class RemoveObjectsEditor(QtWidgets.QWidget):
    """Editor for 'remove_objects' operations."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._raw_op: dict = {}
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Operation: remove_objects"))

        # Description
        desc_row = QtWidgets.QFormLayout()
        self.txt_description = QtWidgets.QLineEdit()
        desc_row.addRow("Description:", self.txt_description)
        layout.addLayout(desc_row)

        # Keep patterns
        grp = QtWidgets.QGroupBox("Keep Patterns")
        grp_layout = QtWidgets.QVBoxLayout(grp)
        self.lst_patterns = QtWidgets.QListWidget()
        self.lst_patterns.setMaximumHeight(120)
        grp_layout.addWidget(self.lst_patterns)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_pat = QtWidgets.QPushButton("+ Add")
        self.btn_rm_pat = QtWidgets.QPushButton("- Remove")
        btn_row.addWidget(self.btn_add_pat)
        btn_row.addWidget(self.btn_rm_pat)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)
        layout.addWidget(grp)

        self.btn_add_pat.clicked.connect(self._add_pattern)
        self.btn_rm_pat.clicked.connect(self._remove_pattern)

        layout.addStretch()

    def _add_pattern(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Add Keep Pattern", "Pattern:"
        )
        if ok and text.strip():
            item = QtWidgets.QListWidgetItem(text.strip())
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lst_patterns.addItem(item)

    def _remove_pattern(self):
        row = self.lst_patterns.currentRow()
        if row >= 0:
            self.lst_patterns.takeItem(row)

    def populate(self, raw: dict, resolved: dict = None):
        self._raw_op = dict(raw)
        self.txt_description.setText(raw.get("description", ""))
        self.lst_patterns.clear()
        for pat in raw.get("keep_patterns", []) or []:
            item = QtWidgets.QListWidgetItem(str(pat))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lst_patterns.addItem(item)

    def collect(self) -> dict:
        patterns = []
        for i in range(self.lst_patterns.count()):
            patterns.append(self.lst_patterns.item(i).text())
        out = dict(self._raw_op)  # preserve unknown keys
        out["operation"] = "remove_objects"
        desc = self.txt_description.text().strip()
        if desc:
            out["description"] = desc
        if patterns:
            out["keep_patterns"] = patterns
        elif "keep_patterns" in out:
            del out["keep_patterns"]
        return out


class SetPositionEditor(QtWidgets.QWidget):
    """Editor for 'set_position' operations."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._raw_op: dict = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Operation: set_position"))

        desc_row = QtWidgets.QFormLayout()
        self.txt_description = QtWidgets.QLineEdit()
        desc_row.addRow("Description:", self.txt_description)
        layout.addLayout(desc_row)

        # Position
        pos_grp = QtWidgets.QGroupBox("Position")
        pos_form = QtWidgets.QFormLayout(pos_grp)
        self.spins_pos = []
        for axis in ("X", "Y", "Z"):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-99999.0, 99999.0)
            s.setDecimals(3)
            pos_form.addRow(axis + ":", s)
            self.spins_pos.append(s)
        layout.addWidget(pos_grp)

        # Rotation
        rot_grp = QtWidgets.QGroupBox("Rotation")
        rot_form = QtWidgets.QFormLayout(rot_grp)
        self.spins_rot = []
        for axis in ("X", "Y", "Z"):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-360.0, 360.0)
            s.setDecimals(1)
            s.setSuffix("\u00b0")
            rot_form.addRow(axis + ":", s)
            self.spins_rot.append(s)
        layout.addWidget(rot_grp)

        layout.addStretch()

    def populate(self, raw: dict, resolved: dict = None):
        self._raw_op = dict(raw)
        self.txt_description.setText(raw.get("description", ""))
        pos = raw.get("position", [0, 0, 0])
        rot = raw.get("rotation", [0, 0, 0])
        for i in range(3):
            self.spins_pos[i].setValue(float(pos[i]) if i < len(pos) else 0.0)
            self.spins_rot[i].setValue(float(rot[i]) if i < len(rot) else 0.0)

    def collect(self) -> dict:
        out = dict(self._raw_op)  # preserve unknown keys
        out["operation"] = "set_position"
        out["position"] = [s.value() for s in self.spins_pos]
        out["rotation"] = [s.value() for s in self.spins_rot]
        desc = self.txt_description.text().strip()
        if desc:
            out["description"] = desc
        return out


class BuildSegmentEditor(QtWidgets.QWidget):
    """Editor for 'build_segment' operations (most complex)."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._raw_op: dict = {}

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        self._inner_layout = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        lay = self._inner_layout

        lay.addWidget(QtWidgets.QLabel("Operation: build_segment"))

        # Name / description
        top_form = QtWidgets.QFormLayout()
        self.txt_name = QtWidgets.QLineEdit()
        top_form.addRow("Name:", self.txt_name)
        self.txt_description = QtWidgets.QLineEdit()
        top_form.addRow("Description:", self.txt_description)
        lay.addLayout(top_form)

        # -- Curve section --
        self.curve_grp = QtWidgets.QGroupBox("Curve")
        curve_lay = QtWidgets.QVBoxLayout(self.curve_grp)

        type_row = QtWidgets.QFormLayout()
        self.cmb_curve_type = QtWidgets.QComboBox()
        self.cmb_curve_type.addItems(EDITABLE_CURVE_TYPES)
        type_row.addRow("Type:", self.cmb_curve_type)
        curve_lay.addLayout(type_row)

        self.curve_params_form = CurveParameterForm()
        curve_lay.addWidget(self.curve_params_form)

        lay.addWidget(self.curve_grp)

        self.cmb_curve_type.currentTextChanged.connect(self._on_curve_type_changed)

        # -- Wafer settings --
        self.wafer_form = WaferSettingsForm()
        lay.addWidget(self.wafer_form)

        # -- Segment settings --
        self.segment_form = SegmentSettingsForm()
        lay.addWidget(self.segment_form)

        # -- Connection --
        self.conn_grp = QtWidgets.QGroupBox("Connection")
        conn_form = QtWidgets.QFormLayout(self.conn_grp)
        self.spin_conn_angle = QtWidgets.QDoubleSpinBox()
        self.spin_conn_angle.setRange(-360.0, 360.0)
        self.spin_conn_angle.setDecimals(1)
        self.spin_conn_angle.setSuffix("\u00b0")
        conn_form.addRow("Rotation angle:", self.spin_conn_angle)
        lay.addWidget(self.conn_grp)

        lay.addStretch()

    def _on_curve_type_changed(self, curve_type: str):
        self.curve_params_form.rebuild(curve_type)

    def populate(self, raw: dict, resolved: dict = None):
        resolved = resolved or raw
        self._raw_op = raw

        self.txt_name.setText(raw.get("name", ""))
        self.txt_description.setText(raw.get("description", ""))

        # Curve
        raw_cs = raw.get("curve_spec", {}) or {}
        res_cs = resolved.get("curve_spec", {}) or {}
        curve_type = raw_cs.get("type", "linear")

        # If it's a non-editable type (existing_curve, closing_curve), still show
        # it but disable the combo
        if curve_type in EDITABLE_CURVE_TYPES:
            idx = self.cmb_curve_type.findText(curve_type)
            if idx >= 0:
                self.cmb_curve_type.blockSignals(True)
                self.cmb_curve_type.setCurrentIndex(idx)
                self.cmb_curve_type.blockSignals(False)
            self.cmb_curve_type.setEnabled(True)
        else:
            # Non-standard type — add temporarily
            self.cmb_curve_type.blockSignals(True)
            self.cmb_curve_type.addItem(curve_type)
            self.cmb_curve_type.setCurrentText(curve_type)
            self.cmb_curve_type.blockSignals(False)
            self.cmb_curve_type.setEnabled(False)

        raw_params = raw_cs.get("parameters", {}) or {}
        res_params = res_cs.get("parameters", {}) or {}
        self.curve_params_form.rebuild(curve_type, raw_params, res_params)

        # Wafer settings
        self.wafer_form.populate(
            raw.get("wafer_settings", {}) or {},
            resolved.get("wafer_settings", {}) or {},
        )

        # Segment settings
        self.segment_form.populate(raw.get("segment_settings", {}) or {})

        # Connection
        conn = raw.get("connection", {}) or {}
        self.spin_conn_angle.setValue(float(conn.get("rotation_angle", 0.0)))

    def collect(self) -> dict:
        out = dict(self._raw_op)  # preserve unknown keys
        curve_type = self.cmb_curve_type.currentText()
        out["operation"] = "build_segment"
        out["segment_type"] = out.get("segment_type", "curve_follower")
        out["name"] = self.txt_name.text().strip()

        # Merge curve_spec preserving unknown keys
        raw_cs = dict(self._raw_op.get("curve_spec", {}) or {})
        raw_cs["type"] = curve_type
        raw_cs["parameters"] = self.curve_params_form.collect()
        out["curve_spec"] = raw_cs

        out["wafer_settings"] = self.wafer_form.collect()
        out["segment_settings"] = self.segment_form.collect()

        desc = self.txt_description.text().strip()
        if desc:
            out["description"] = desc

        conn_angle = self.spin_conn_angle.value()
        if conn_angle != 0.0:
            out["connection"] = {"rotation_angle": conn_angle}
        elif "connection" not in self._raw_op:
            out.pop("connection", None)

        return out


class CloseCurveEditor(QtWidgets.QWidget):
    """Editor for 'close_curve' operations."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._raw_op: dict = {}

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        self._inner_layout = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        lay = self._inner_layout

        lay.addWidget(QtWidgets.QLabel("Operation: close_curve"))

        top_form = QtWidgets.QFormLayout()
        self.txt_name = QtWidgets.QLineEdit()
        top_form.addRow("Name:", self.txt_name)
        self.txt_description = QtWidgets.QLineEdit()
        top_form.addRow("Description:", self.txt_description)

        self.spin_max_angle = QtWidgets.QDoubleSpinBox()
        self.spin_max_angle.setRange(0.0, 360.0)
        self.spin_max_angle.setDecimals(1)
        self.spin_max_angle.setSuffix("\u00b0")
        self.spin_max_angle.setValue(90.0)
        top_form.addRow("Max closing angle:", self.spin_max_angle)

        self.spin_points = QtWidgets.QSpinBox()
        self.spin_points.setRange(2, 9999)
        self.spin_points.setValue(50)
        top_form.addRow("Points:", self.spin_points)

        self.spin_num_lcs = QtWidgets.QSpinBox()
        self.spin_num_lcs.setRange(1, 20)
        self.spin_num_lcs.setValue(3)
        top_form.addRow("LCS per end:", self.spin_num_lcs)

        self.spin_per_segment = QtWidgets.QSpinBox()
        self.spin_per_segment.setRange(2, 9999)
        self.spin_per_segment.setValue(100)
        top_form.addRow("Sampling per segment:", self.spin_per_segment)

        lay.addLayout(top_form)

        # Wafer & segment settings
        self.wafer_form = WaferSettingsForm()
        lay.addWidget(self.wafer_form)

        self.segment_form = SegmentSettingsForm()
        lay.addWidget(self.segment_form)

        lay.addStretch()

    def populate(self, raw: dict, resolved: dict = None):
        resolved = resolved or raw
        self._raw_op = dict(raw)
        self.txt_name.setText(raw.get("name", ""))
        self.txt_description.setText(raw.get("description", ""))
        self.spin_max_angle.setValue(float(raw.get("max_closing_angle", 90.0)))
        self.spin_points.setValue(int(raw.get("points", 50)))
        self.spin_num_lcs.setValue(int(raw.get("num_lcs_per_end", 3)))
        sampling = raw.get("sampling", {}) or {}
        self.spin_per_segment.setValue(int(sampling.get("per_segment", 100)))

        self.wafer_form.populate(
            raw.get("wafer_settings", {}) or {},
            resolved.get("wafer_settings", {}) or {},
        )
        self.segment_form.populate(raw.get("segment_settings", {}) or {})

    def collect(self) -> dict:
        out = dict(self._raw_op)  # preserve unknown keys
        out["operation"] = "close_curve"
        out["name"] = self.txt_name.text().strip()
        out["max_closing_angle"] = self.spin_max_angle.value()
        out["points"] = self.spin_points.value()
        out["num_lcs_per_end"] = self.spin_num_lcs.value()
        raw_sampling = dict(self._raw_op.get("sampling", {}) or {})
        raw_sampling["per_segment"] = self.spin_per_segment.value()
        out["sampling"] = raw_sampling
        out["wafer_settings"] = self.wafer_form.collect()
        out["segment_settings"] = self.segment_form.collect()
        desc = self.txt_description.text().strip()
        if desc:
            out["description"] = desc
        return out


class ExportCurveEditor(QtWidgets.QWidget):
    """Editor for 'export_curve' operations."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._raw_op: dict = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Operation: export_curve"))

        top_form = QtWidgets.QFormLayout()
        self.txt_name = QtWidgets.QLineEdit()
        top_form.addRow("Name:", self.txt_name)
        self.txt_description = QtWidgets.QLineEdit()
        top_form.addRow("Description:", self.txt_description)

        self.spin_per_segment = QtWidgets.QSpinBox()
        self.spin_per_segment.setRange(2, 9999)
        self.spin_per_segment.setValue(100)
        top_form.addRow("Sampling per segment:", self.spin_per_segment)
        layout.addLayout(top_form)

        # Segments list
        seg_grp = QtWidgets.QGroupBox("Segments")
        seg_lay = QtWidgets.QVBoxLayout(seg_grp)
        self.lst_segments = QtWidgets.QListWidget()
        self.lst_segments.setMaximumHeight(140)
        seg_lay.addWidget(self.lst_segments)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_seg = QtWidgets.QPushButton("+ Add")
        self.btn_rm_seg = QtWidgets.QPushButton("- Remove")
        btn_row.addWidget(self.btn_add_seg)
        btn_row.addWidget(self.btn_rm_seg)
        btn_row.addStretch()
        seg_lay.addLayout(btn_row)
        layout.addWidget(seg_grp)

        self.btn_add_seg.clicked.connect(self._add_segment)
        self.btn_rm_seg.clicked.connect(self._remove_segment)

        layout.addStretch()

    def _add_segment(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Add Segment", "Segment name:"
        )
        if ok and text.strip():
            item = QtWidgets.QListWidgetItem(text.strip())
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lst_segments.addItem(item)

    def _remove_segment(self):
        row = self.lst_segments.currentRow()
        if row >= 0:
            self.lst_segments.takeItem(row)

    def populate(self, raw: dict, resolved: dict = None):
        self._raw_op = dict(raw)
        self.txt_name.setText(raw.get("name", ""))
        self.txt_description.setText(raw.get("description", ""))
        sampling = raw.get("sampling", {}) or {}
        self.spin_per_segment.setValue(int(sampling.get("per_segment", 100)))

        self.lst_segments.clear()
        for seg_name in raw.get("segments", []) or []:
            item = QtWidgets.QListWidgetItem(str(seg_name))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lst_segments.addItem(item)

    def collect(self) -> dict:
        segments = []
        for i in range(self.lst_segments.count()):
            segments.append(self.lst_segments.item(i).text())
        out = dict(self._raw_op)  # preserve unknown keys
        out["operation"] = "export_curve"
        out["name"] = self.txt_name.text().strip()
        out["segments"] = segments
        raw_sampling = dict(self._raw_op.get("sampling", {}) or {})
        raw_sampling["per_segment"] = self.spin_per_segment.value()
        out["sampling"] = raw_sampling
        desc = self.txt_description.text().strip()
        if desc:
            out["description"] = desc
        return out


# =========================================================================
# Factory
# =========================================================================

OPERATION_TYPES = [
    "remove_objects",
    "build_segment",
    "set_position",
    "close_curve",
    "export_curve",
]

_EDITOR_MAP = {
    "remove_objects": RemoveObjectsEditor,
    "build_segment": BuildSegmentEditor,
    "set_position": SetPositionEditor,
    "close_curve": CloseCurveEditor,
    "close_loop": CloseCurveEditor,
    "export_curve": ExportCurveEditor,
}


def create_editor(op_type: str, parent: QtWidgets.QWidget = None) -> QtWidgets.QWidget:
    """Create the appropriate editor widget for *op_type*."""
    cls = _EDITOR_MAP.get(op_type)
    if cls is None:
        lbl = QtWidgets.QLabel(f"No editor for operation type: {op_type}")
        lbl.setWordWrap(True)
        return lbl
    return cls(parent)
