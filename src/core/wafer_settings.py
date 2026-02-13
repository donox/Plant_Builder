from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WaferSettings:
    cylinder_diameter: float = 2.0
    profile_density: float = 0.89
    min_height: float = 0.1
    max_chord: float = 0.5
    max_wafer_count: Optional[int] = None
    min_inner_chord: float = 0.25

    @classmethod
    def from_dict(cls, d: dict) -> "WaferSettings":
        """Create from YAML dict, ignoring unknown keys."""
        import inspect
        valid_keys = inspect.signature(cls).parameters
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @property
    def cylinder_radius(self) -> float:
        return self.cylinder_diameter / 2.0
