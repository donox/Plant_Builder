"""Parser for Plant Builder cutting list text files."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

try:
    from core.logging_setup import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class WaferSpec:
    """Specification for a single wafer from cutting list."""
    wafer_num: int
    wafer_type: str  # CE, EE, EC, CC
    lift_angle: Optional[float]  # degrees (None for last wafer)
    rotation_angle: Optional[float]  # degrees (None for last wafer)
    outside_height: float  # in model units (converted from inches)
    saw_position: Optional[float]  # degrees (for validation)
    chord_length: Optional[float] = None  # Calculated during reconstruction

    def __repr__(self):
        return (f"Wafer({self.wafer_num}: {self.wafer_type}, "
                f"lift={self.lift_angle:.2f}°, rot={self.rotation_angle:.2f}°, "
                f"outside={self.outside_height:.3f})")


@dataclass
class SegmentData:
    """Data for a complete segment from cutting list."""
    segment_name: str
    wafers: List[WaferSpec]
    total_cylinder_length: float

    def __repr__(self):
        return (f"Segment('{self.segment_name}', "
                f"{len(self.wafers)} wafers, "
                f"length={self.total_cylinder_length:.2f})")


class CutListParser:
    """Parse Plant Builder cutting list text files."""

    def __init__(self, units_per_inch: float = 25.4):
        """Initialize parser.

        Args:
            units_per_inch: Conversion factor (25.4 for mm, 1.0 for inches)
        """
        self.units_per_inch = units_per_inch


    def parse_file(self, filename: str) -> Dict[str, Any]:
        """Parse complete cutting list file.

        Args:
            filename: Path to cutting list text file

        Returns:
            Dictionary with:
                - project_name: str
                - bounds: Dict with x_min, x_max, y_min, y_max, z_min, z_max
                - segments: List[SegmentData]
        """
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Cut list file not found: {filename}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        result = {
            'project_name': '',
            'bounds': {},
            'segments': []
        }

        # Parse project name
        for line in lines:
            if 'Cut List for:' in line:
                result['project_name'] = line.split('Cut List for:')[1].strip()
                break

        # Parse bounds
        result['bounds'] = self._parse_bounds(lines)

        # Parse segments
        result['segments'] = self._parse_segments(lines)

        logger.info(f"Parsed cut list: {result['project_name']}")
        logger.info(f"  Found {len(result['segments'])} segment(s)")
        for seg in result['segments']:
            logger.info(f"    - {seg}")

        return result

    def _parse_bounds(self, lines: List[str]) -> Dict[str, float]:
        """Parse project bounds from lines."""
        bounds = {}

        for line in lines:
            if 'X-min:' in line:
                parts = line.split()
                bounds['x_min'] = float(parts[1])
                bounds['x_max'] = float(parts[3])
            elif 'Y-min:' in line:
                parts = line.split()
                bounds['y_min'] = float(parts[1])
                bounds['y_max'] = float(parts[3])
            elif 'Z-min:' in line:
                parts = line.split()
                bounds['z_min'] = float(parts[1])
                bounds['z_max'] = float(parts[3])

        return bounds

    def _parse_segments(self, lines: List[str]) -> List[SegmentData]:
        """Parse all segments from lines."""
        segments = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for segment header
            if "Segment '" in line and "' cuts" in line:
                segment_name = line.split("'")[1]
                segment_data = self._parse_segment(lines[i:], segment_name)
                if segment_data:
                    segments.append(segment_data)

            i += 1

        return segments

    def _parse_segment(self, lines: List[str], segment_name: str) -> Optional[SegmentData]:
        """Parse a single segment's wafer data.

        Args:
            lines: Lines starting from segment header
            segment_name: Name of the segment

        Returns:
            SegmentData or None if parsing fails
        """
        wafers = []
        total_length = 0.0

        # Find the table header
        header_idx = None
        for i, line in enumerate(lines):
            if 'Wafer\tType\tLift' in line or 'Wafer' in line and 'Type' in line:
                header_idx = i
                break

        if header_idx is None:
            logger.warning(f"Could not find wafer table for segment '{segment_name}'")
            return None

        # Parse wafer rows
        for i in range(header_idx + 1, len(lines)):
            line = lines[i].strip()

            # Stop at empty line or next segment
            if not line or 'Total Cylinder Length:' in line:
                if 'Total Cylinder Length:' in line:
                    # Extract total length
                    match = re.search(r'([\d.]+)', line)
                    if match:
                        total_length = float(match.group(1))
                break

            if line.startswith('Segment'):
                break

            # Parse wafer row
            wafer = self._parse_wafer_row(line)
            if wafer:
                wafers.append(wafer)

        if not wafers:
            logger.warning(f"No wafers found for segment '{segment_name}'")
            return None

        return SegmentData(
            segment_name=segment_name,
            wafers=wafers,
            total_cylinder_length=total_length
        )

    def _parse_wafer_row(self, line: str) -> Optional[WaferSpec]:
        """Parse a single wafer row from the table.

        Expected format:
        1	CE	3.93		0.41		1 7/16		0.00
        or
        20	EC	---		---		1 1/16		164.36
        """
        parts = line.split('\t')

        # Filter out empty parts
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 4:
            return None

        try:
            wafer_num = int(parts[0])
            wafer_type = parts[1]

            # Handle lift angle (may be --- for last wafer)
            if parts[2] == '---':
                lift_angle = None
            else:
                lift_angle = float(parts[2])

            # Handle rotation angle (may be --- for last wafer)
            if parts[3] == '---':
                rotation_angle = None
            else:
                rotation_angle = float(parts[3])

            # Parse outside height (format: "1 7/16")
            outside_height = self._parse_fractional_inches(parts[4])

            # Parse saw position (may be --- for last wafer)
            if len(parts) > 5 and parts[5] != '---':
                saw_position = float(parts[5])
            else:
                saw_position = None

            return WaferSpec(
                wafer_num=wafer_num,
                wafer_type=wafer_type,
                lift_angle=lift_angle,
                rotation_angle=rotation_angle,
                outside_height=outside_height,
                saw_position=saw_position
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse wafer row: {line} ({e})")
            return None

    def _parse_fractional_inches(self, text: str) -> float:
        """Parse fractional inch measurement to model units.

        Examples:
            "1 7/16" -> 1.4375 inches -> 36.5125 mm
            "2 0/16" -> 2.0 inches -> 50.8 mm
        """
        parts = text.split()

        if len(parts) == 1:
            # Just a number
            inches = float(parts[0])
        elif len(parts) == 2:
            # Whole number and fraction
            whole = int(parts[0])

            # Parse fraction
            if '/' in parts[1]:
                num, denom = parts[1].split('/')
                fraction = int(num) / int(denom)
            else:
                fraction = 0.0

            inches = whole + fraction
        else:
            raise ValueError(f"Cannot parse fractional inches: {text}")

        # Convert to model units
        return inches # * self.units_per_inch