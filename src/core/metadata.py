# src/core/metadata.py
from __future__ import annotations
from typing import Any, Dict, Optional


def apply_metadata(doc, metadata: Dict[str, Any], *, source_config: Optional[str] = None) -> None:
    """
    Apply metadata into the FreeCAD document.

    - Writes metadata to doc.Meta (File → Properties → Custom)
    - Creates/updates a visible 'ModelMetadata' object in the tree
    - Idempotent (safe to run multiple times)
    """
    if doc is None or not metadata:
        return

    # ---- Document Meta (Properties → Custom) ----
    for k, v in metadata.items():
        doc.Meta[str(k)] = "" if v is None else str(v)

    if source_config:
        doc.Meta["SourceConfig"] = str(source_config)

    # Optional but useful provenance
    doc.Meta["Generator"] = "PlantBuilder"
    doc.Meta["MetadataApplied"] = "true"

    # ---- Visible object in the tree ----
    meta_obj = doc.getObject("ModelMetadata")
    if meta_obj is None:
        meta_obj = doc.addObject("App::FeaturePython", "ModelMetadata")
        meta_obj.Label = "Model Metadata"

    def ensure_prop(name: str, prop_type: str, group: str, tooltip: str) -> None:
        if not hasattr(meta_obj, name):
            meta_obj.addProperty(prop_type, name, group, tooltip)

    ensure_prop("ProjectName", "App::PropertyString", "Metadata", "Project name")
    ensure_prop("Version", "App::PropertyString", "Metadata", "Model version")
    ensure_prop("Description", "App::PropertyString", "Metadata", "Model description")
    ensure_prop("Created", "App::PropertyString", "Metadata", "Creation date")
    ensure_prop("SourceConfig", "App::PropertyString", "Metadata", "Source YAML file")
    ensure_prop("Generator", "App::PropertyString", "Metadata", "Generating system")

    meta_obj.ProjectName = str(metadata.get("project_name", ""))
    meta_obj.Version = str(metadata.get("version", ""))
    meta_obj.Description = str(metadata.get("description", ""))
    meta_obj.Created = str(metadata.get("created", ""))
    meta_obj.SourceConfig = str(source_config or "")
    meta_obj.Generator = "PlantBuilder"

    doc.recompute()
