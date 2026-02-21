from __future__ import annotations
from dataclasses import dataclass
from typing import List

_KNOWN_OPERATIONS = {"remove_objects", "build_segment", "set_position",
                     "close_curve", "close_loop", "export_curve"}
_KNOWN_CURVE_TYPES = {"linear", "helical", "sinusoidal", "overhand_knot",
                      "circle", "spiral", "figure_eight", "trefoil", "custom"}


@dataclass
class ValidationIssue:
    level: str      # "error" | "warning"
    location: str   # e.g. "workflow[2].wafer_settings.max_chord"
    message: str

    def __str__(self):
        return f"[{self.level.upper():7}] {self.location}: {self.message}"


def validate_config(loaded_config) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    data = loaded_config.data
    _check_top_level(data, issues)
    _check_workflow(data.get("workflow") or [], issues)
    _check_output_files(data.get("output_files") or {}, issues)
    return issues


def _check_top_level(data, issues):
    for key in ("global_settings", "output_files", "workflow"):
        if key not in data:
            issues.append(ValidationIssue("error", key,
                f"Required top-level key '{key}' is missing"))
    if "workflow" in data and not isinstance(data["workflow"], list):
        issues.append(ValidationIssue("error", "workflow", "Must be a list"))


def _check_workflow(workflow, issues):
    if not isinstance(workflow, list):
        return
    segment_names: list[str] = []

    for i, op in enumerate(workflow):
        loc = f"workflow[{i}]"
        if not isinstance(op, dict):
            issues.append(ValidationIssue("error", loc, "Operation must be a dict"))
            continue
        op_name = op.get("operation")
        if not op_name:
            issues.append(ValidationIssue("error", loc, "Missing 'operation' field"))
            continue
        if op_name not in _KNOWN_OPERATIONS:
            issues.append(ValidationIssue("warning", f"{loc}.operation",
                f"Unknown operation '{op_name}'"))
            continue

        if op_name == "build_segment":
            _check_build_segment(op, loc, segment_names, issues)
            if op.get("name"):
                segment_names.append(op["name"])
        elif op_name == "export_curve":
            _check_export_curve(op, loc, segment_names, issues)
        elif op_name in ("close_curve", "close_loop"):
            _check_wafer_settings(op.get("wafer_settings"), f"{loc}.wafer_settings", issues)


def _check_build_segment(op, loc, segment_names, issues):
    name = op.get("name")
    if not name:
        issues.append(ValidationIssue("error", f"{loc}.name",
            "build_segment requires a 'name'"))
    elif name in segment_names:
        issues.append(ValidationIssue("error", f"{loc}.name",
            f"Duplicate segment name '{name}'"))

    if not op.get("segment_type"):
        issues.append(ValidationIssue("warning", f"{loc}.segment_type",
            "Missing 'segment_type' (expected 'curve_follower')"))

    curve_spec = op.get("curve_spec")
    if not curve_spec:
        issues.append(ValidationIssue("error", f"{loc}.curve_spec",
            "Missing 'curve_spec'"))
    else:
        _check_curve_spec(curve_spec, f"{loc}.curve_spec", issues)

    _check_wafer_settings(op.get("wafer_settings"), f"{loc}.wafer_settings", issues)


def _check_curve_spec(spec, loc, issues):
    ct = spec.get("type")
    if not ct:
        issues.append(ValidationIssue("error", f"{loc}.type",
            "Missing curve type"))
    elif ct not in _KNOWN_CURVE_TYPES:
        issues.append(ValidationIssue("warning", f"{loc}.type",
            f"Unknown curve type '{ct}'"))

    seg = spec.get("segment") or {}
    start = seg.get("start_fraction")
    end = seg.get("end_fraction")
    if start is not None and not (0.0 <= start <= 1.0):
        issues.append(ValidationIssue("error", f"{loc}.segment.start_fraction",
            f"Must be in [0, 1], got {start}"))
    if end is not None and not (0.0 <= end <= 1.0):
        issues.append(ValidationIssue("error", f"{loc}.segment.end_fraction",
            f"Must be in [0, 1], got {end}"))
    if start is not None and end is not None and start >= end:
        issues.append(ValidationIssue("error", f"{loc}.segment",
            f"start_fraction ({start}) must be less than end_fraction ({end})"))


def _check_export_curve(op, loc, segment_names, issues):
    segments = op.get("segments")
    if not segments:
        issues.append(ValidationIssue("error", f"{loc}.segments",
            "export_curve requires a non-empty 'segments' list"))
        return
    for seg_name in segments:
        if seg_name not in segment_names:
            issues.append(ValidationIssue("error", f"{loc}.segments",
                f"Segment '{seg_name}' referenced but not defined earlier in workflow"))


def _check_wafer_settings(ws, loc, issues):
    if not ws:
        return
    _POSITIVE_FIELDS = ("cylinder_diameter", "min_height", "max_chord", "min_inner_chord")
    for field in _POSITIVE_FIELDS:
        val = ws.get(field)
        if val is not None:
            if not isinstance(val, (int, float)):
                issues.append(ValidationIssue("error", f"{loc}.{field}",
                    f"Must be a number, got {type(val).__name__}"))
            elif val <= 0:
                issues.append(ValidationIssue("error", f"{loc}.{field}",
                    f"Must be > 0, got {val}"))
    pd = ws.get("profile_density")
    if pd is not None:
        if not isinstance(pd, (int, float)):
            issues.append(ValidationIssue("error", f"{loc}.profile_density",
                f"Must be a number, got {type(pd).__name__}"))
        elif not (0.0 < pd <= 1.0):
            issues.append(ValidationIssue("error", f"{loc}.profile_density",
                f"Must be in (0, 1], got {pd}"))
    mwc = ws.get("max_wafer_count")
    if mwc is not None and (not isinstance(mwc, int) or mwc <= 0):
        issues.append(ValidationIssue("error", f"{loc}.max_wafer_count",
            f"Must be a positive integer, got {mwc}"))


def _check_output_files(output_files, issues):
    if not output_files:
        return
    working_dir = output_files.get("working_directory")
    if working_dir and not str(working_dir).startswith("${"):
        import os
        if not os.path.isdir(working_dir):
            issues.append(ValidationIssue("warning", "output_files.working_directory",
                f"Directory '{working_dir}' does not exist"))
        elif not os.access(working_dir, os.W_OK):
            issues.append(ValidationIssue("error", "output_files.working_directory",
                f"Directory '{working_dir}' is not writable"))
