"""Core math/transform helpers for FreeCAD Placements and document management."""
from __future__ import annotations
from typing import Any
import FreeCAD


def make_transform_align(source_lcs: Any, target_lcs: Any) -> FreeCAD.Placement:
    """
    Compute transform that moves 'source_lcs' frame onto 'target_lcs' frame.
    Works with objects that have a .Placement attribute.
    """
    src = source_lcs.Placement if hasattr(source_lcs, "Placement") else source_lcs
    tgt = target_lcs.Placement if hasattr(target_lcs, "Placement") else target_lcs
    return src.inverse().multiply(tgt)


def to_local(placement_world: FreeCAD.Placement, base_lcs: Any) -> FreeCAD.Placement:
    """Convert world placement to local coordinates relative to base_lcs"""
    base = base_lcs.Placement if hasattr(base_lcs, "Placement") else base_lcs
    return base.inverse().multiply(placement_world)


def to_world(placement_local: FreeCAD.Placement, base_lcs: Any) -> FreeCAD.Placement:
    """Convert local placement to world coordinates using base_lcs"""
    base = base_lcs.Placement if hasattr(base_lcs, "Placement") else base_lcs
    return base.multiply(placement_local)


def ensure_group(doc, name, parent=None):
    """
    Return a group (create if missing). If parent is given, ensure the group
    is a child of parent. Uses Label for human-friendly matching.
    """
    # Find existing group by Label
    grp = None
    for obj in getattr(doc, "Objects", []):
        if obj.TypeId.startswith("App::DocumentObjectGroup") and getattr(obj, "Label", "") == name:
            grp = obj
            break

    # Create if missing
    if grp is None:
        grp = doc.addObject("App::DocumentObjectGroup", name)
        grp.Label = name

    # Parent linkage (if provided)
    if parent is not None:
        members = getattr(parent, "Group", []) or []
        if grp not in members:
            parent.addObject(grp)

    return grp


def add_to_group(group, obj):
    """
    Put obj into group if not already present. Works for Group/LinkGroup.
    """
    if group is None or obj is None:
        return obj
    members = getattr(group, "Group", []) or []
    if obj not in members:
        group.addObject(obj)
    return obj