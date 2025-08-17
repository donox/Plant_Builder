
"""Core math/transform helpers for FreeCAD Placements."""
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
    base = base_lcs.Placement if hasattr(base_lcs, "Placement") else base_lcs
    return base.inverse().multiply(placement_world)

def to_world(placement_local: FreeCAD.Placement, base_lcs: Any) -> FreeCAD.Placement:
    base = base_lcs.Placement if hasattr(base_lcs, "Placement") else base_lcs
    return base.multiply(placement_local)
