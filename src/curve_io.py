
import FreeCAD as App
import Part
from typing import List, Tuple
from core.core_utils import is_identity_placement


def _points_are_close(p1: App.Vector, p2: App.Vector, tol: float) -> bool:
    return (p1.sub(p2)).Length <= tol


def _order_edges_into_path(edges: List[Part.Edge], tol: float) -> List[Part.Edge]:
    """
    Given a list of edges, attempt to order them into a single continuous path.
    Edges must connect end-to-end within the given tolerance.

    Returns:
        Ordered list of edges.

    Raises:
        ValueError if a single continuous path cannot be constructed.
    """
    if not edges:
        raise ValueError("No edges provided to build a continuous path.")

    # Start with the first edge arbitrarily
    remaining = list(edges)
    ordered = [remaining.pop(0)]

    while remaining:
        extended = False
        start_point = ordered[0].Vertexes[0].Point
        end_point = ordered[-1].Vertexes[-1].Point

        for i, e in enumerate(remaining):
            v0 = e.Vertexes[0].Point
            v1 = e.Vertexes[-1].Point

            # Try to extend at the end
            if _points_are_close(end_point, v0, tol):
                ordered.append(e)
                remaining.pop(i)
                extended = True
                break
            elif _points_are_close(end_point, v1, tol):
                ordered.append(e.Reversed())
                remaining.pop(i)
                extended = True
                break

            # Try to extend at the start
            if _points_are_close(start_point, v1, tol):
                ordered.insert(0, e)
                remaining.pop(i)
                extended = True
                break
            elif _points_are_close(start_point, v0, tol):
                ordered.insert(0, e.Reversed())
                remaining.pop(i)
                extended = True
                break

        if not extended:
            # Could not attach any remaining edge to either end
            raise ValueError(
                f"Could not build a single continuous path from {len(edges)} edges; "
                f"{len(remaining)} edges did not connect within tolerance {tol}."
            )

    return ordered


def get_wire_from_label(doc: App.Document, label: str, tol: float = 1e-4) -> Part.Wire:
    """
    Find an object by label in the given document and return a single continuous Wire
    built from its edges.

    Behavior:
        - Requires exactly one object with the given Label.
        - If the object's shape already has a single wire, returns that.
        - Otherwise, attempts to order all edges into a continuous path and build a Wire.
        - Raises ValueError if this is not possible.

    Args:
        doc: FreeCAD document.
        label: Object Label to look for (must be unique).
        tol: Distance tolerance for connecting edge endpoints.

    Returns:
        Part.Wire representing a single continuous curve.

    Raises:
        ValueError: if no object or multiple objects are found, or no continuous wire
                    can be constructed.
    """
    if doc is None:
        raise ValueError("Document is None in get_wire_from_label.")

    matches = [obj for obj in doc.Objects if obj.Label == label]

    if not matches:
        raise ValueError(f"No object found with Label '{label}'.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple objects found with Label '{label}'. "
            "Labels must be unique for customcurve/fromlabel."
        )

    obj = matches[0]
    if not hasattr(obj, "Shape"):
        raise ValueError(f"Object '{label}' has no Shape attribute.")

    shape = obj.Shape

    # If the shape already has a single Wire, use it directly
    if len(shape.Wires) == 1:
        wire = shape.Wires[0]
        if wire.Edges:
            return wire
        else:
            raise ValueError(f"Wire on object '{label}' has no edges.")

    # Otherwise, try to construct a single continuous wire from all edges
    edges = list(shape.Edges)
    if not edges:
        raise ValueError(f"Object '{label}' has no edges to build a wire from.")

    ordered_edges = _order_edges_into_path(edges, tol)
    wire = Part.Wire(ordered_edges)

    # Basic sanity: ensure we have as many edges in the wire as we started with
    if len(wire.Edges) != len(edges):
        raise ValueError(
            f"Constructed wire has {len(wire.Edges)} edges but started from "
            f"{len(edges)} edges on object '{label}'."
        )

    return wire


def sample_points_on_wire(wire: Part.Wire, numsamples: int) -> List[App.Vector]:
    if numsamples < 2:
        raise ValueError(f"numsamples must be at least 2, got {numsamples}.")

    total_length = wire.Length
    if total_length <= 0:
        raise ValueError("Wire has non-positive length.")

    edges = list(wire.Edges)
    if not edges:
        raise ValueError("Wire has no edges to sample.")

    # Precompute cumulative edge lengths
    edge_lengths = [e.Length for e in edges]
    cum_lengths = []
    acc = 0.0
    for L in edge_lengths:
        acc += L
        cum_lengths.append(acc)

    points: List[App.Vector] = []

    for i in range(numsamples):
        # target length along whole wire
        target_len = total_length * (i / (numsamples - 1))

        # Find the edge that contains this length
        for edge_index, cum_L in enumerate(cum_lengths):
            start_L = 0.0 if edge_index == 0 else cum_lengths[edge_index - 1]
            if target_len <= cum_L or edge_index == len(edges) - 1:
                local_len = target_len - start_L
                edge = edges[edge_index]
                # Clamp to valid range for this edge
                if local_len < 0.0:
                    local_len = 0.0
                if local_len > edge.Length:
                    local_len = edge.Length
                param = edge.getParameterByLength(local_len)
                p = edge.valueAt(param)
                points.append(App.Vector(p.x, p.y, p.z))
                break

    return points




def transform_world_to_local(points: List[App.Vector],
                             baseplacement: App.Placement) -> List[App.Vector]:
    """
    Transform a list of world-coordinate points into the local frame defined
    by baseplacement.

    Args:
        points: List of App.Vector in world coordinates.
        baseplacement: App.Placement defining the local frame. Typically the
                       segment's baseplacement.

    Returns:
        List of App.Vector in local coordinates.
    """
    if baseplacement is None or is_identity_placement(baseplacement):
        return list(points)

    placement_inv = baseplacement.inverse()
    return [placement_inv.multVec(p) for p in points]


