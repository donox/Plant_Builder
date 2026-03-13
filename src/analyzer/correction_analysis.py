"""correction_analysis.py

Identifies optimal correction points for a PlantBuilder cut list.

A correction point is created by splitting a wafer at its midpoint with a
perpendicular (blade=0) cut.  The flat circular joint allows free rotation of
the cylinder before making the next cut, so any accumulated spin error can be
zeroed out without affecting the shape.

The spin error grows as a random walk: σ_rot × √k after k cuts.  Placing a
correction after k cuts resets the walk, so the problem is to choose split
wafers that minimise the longest un-corrected section.

Usage (standalone)
------------------
    from cut_list_reconstruction import parse_cut_list
    from analyzer.correction_analysis import analyse_correction_points, format_correction_report

    segments = parse_cut_list("path/to/cuts.txt")
    for seg in segments:
        result = analyse_correction_points(seg['rows'], sigma_rot_deg=1.15)
        print(format_correction_report(seg['name'], result))
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CorrectionCandidate:
    """A single wafer whose midpoint split gives a one-correction schedule."""
    wafer_index:     int    # 0-based index into rows list
    cut_number:      int    # cut number as printed in the cut list (1-based)
    split_pos:       float  # effective correction position in wafer-count units (= wafer_index + 0.5)
    sec_before:      float  # section length before correction (wafers)
    sec_after:       float  # section length after correction (wafers)
    max_err_deg:     float  # σ_rot × max(√sec_before, √sec_after)
    max_blade_deg:   float  # max(theta_entry, theta_exit) — jig registration quality
    theta_entry_deg: float
    theta_exit_deg:  float


@dataclass
class TwoCorrectionCandidate:
    """A pair of wafers whose midpoint splits give a two-correction schedule."""
    wafer_a:     int
    wafer_b:     int
    cut_a:       int
    cut_b:       int
    sec1:        float  # section before first correction
    sec2:        float  # section between corrections
    sec3:        float  # section after second correction
    max_err_deg: float  # σ_rot × max(√sec1, √sec2, √sec3)
    max_blade_a: float  # max blade at wafer_a
    max_blade_b: float  # max blade at wafer_b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blade_angles(row: dict) -> Tuple[float, float]:
    """Return (theta_entry_deg, theta_exit_deg) for a cut list row."""
    blade = row.get('blade_deg', 0.0)
    return row.get('theta_entry_deg', blade), row.get('theta_exit_deg', blade)


def _max_err(sigma: float, *section_lengths: float) -> float:
    return sigma * max(math.sqrt(max(s, 0.0)) for s in section_lengths)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_correction_points(
    rows: list,
    sigma_rot_deg: float = 1.15,
    top_n: int = 5,
) -> dict:
    """Analyse a parsed cut list segment for optimal correction point wafers.

    Parameters
    ----------
    rows : list of row dicts from :func:`cut_list_reconstruction.parse_cut_list`
    sigma_rot_deg : 1σ rotation error per cut (degrees)
    top_n : number of top candidates to return for each correction count

    Returns
    -------
    dict with keys:
        'n_wafers'            : int
        'sigma_rot_deg'       : float
        'baseline_max_err_deg': float  (no corrections)
        'single'              : list[CorrectionCandidate]   (top_n)
        'double'              : list[TwoCorrectionCandidate] (top_n)
    """
    N = len(rows)
    if N < 2:
        return {}

    baseline = sigma_rot_deg * math.sqrt(N)

    # ── Single correction candidates ──────────────────────────────────────────
    singles: List[CorrectionCandidate] = []
    for i, row in enumerate(rows):
        entry, exit_ = _blade_angles(row)
        p  = i + 0.5          # effective split position (wafers)
        s1 = p
        s2 = N - p
        singles.append(CorrectionCandidate(
            wafer_index     = i,
            cut_number      = row.get('cut', i + 1),
            split_pos       = p,
            sec_before      = s1,
            sec_after       = s2,
            max_err_deg     = _max_err(sigma_rot_deg, s1, s2),
            max_blade_deg   = max(entry, exit_),
            theta_entry_deg = entry,
            theta_exit_deg  = exit_,
        ))

    # Sort: minimise max_err first; prefer higher blade angle as tiebreaker.
    singles.sort(key=lambda c: (round(c.max_err_deg, 4), -c.max_blade_deg))

    # ── Double correction candidates ──────────────────────────────────────────
    doubles: List[TwoCorrectionCandidate] = []
    for i in range(N):
        entry_a, exit_a = _blade_angles(rows[i])
        mb_a = max(entry_a, exit_a)
        for j in range(i + 1, N):
            entry_b, exit_b = _blade_angles(rows[j])
            s1 = i + 0.5
            s2 = j - i          # gap between the two corrections (integer wafers)
            s3 = N - j - 0.5
            doubles.append(TwoCorrectionCandidate(
                wafer_a     = i,
                wafer_b     = j,
                cut_a       = rows[i].get('cut', i + 1),
                cut_b       = rows[j].get('cut', j + 1),
                sec1        = s1,
                sec2        = s2,
                sec3        = s3,
                max_err_deg = _max_err(sigma_rot_deg, s1, s2, s3),
                max_blade_a = mb_a,
                max_blade_b = max(entry_b, exit_b),
            ))

    doubles.sort(key=lambda c: (round(c.max_err_deg, 4),
                                -(c.max_blade_a + c.max_blade_b)))

    return {
        'n_wafers':             N,
        'sigma_rot_deg':        sigma_rot_deg,
        'baseline_max_err_deg': baseline,
        'single':               singles[:top_n],
        'double':               doubles[:top_n],
    }


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_correction_report(seg_name: str, result: dict) -> str:
    """Return a human-readable correction-point analysis report."""
    if not result:
        return "No data."

    N        = result['n_wafers']
    sigma    = result['sigma_rot_deg']
    baseline = result['baseline_max_err_deg']
    W = 72

    lines = []
    lines.append("=" * W)
    lines.append(f"Correction Point Analysis — '{seg_name}'  ({N} wafers)")
    lines.append(f"σ_rot = {sigma:.2f}°  |  No corrections: "
                 f"max 1σ spin error = {baseline:.2f}°")
    lines.append("=" * W)

    # ── Single correction ────────────────────────────────────────────────────
    ideal_single = sigma * math.sqrt(N / 2.0)
    lines.append("")
    lines.append("── Single correction point " + "─" * (W - 27))
    lines.append(f"   Ideal split position: wafer {N / 2:.1f}  "
                 f"→  max error = {ideal_single:.2f}° (1σ)")
    lines.append("")
    lines.append(f"   {'Rank':>4}  {'Cut':>4}  {'SplitAt':>7}  "
                 f"{'Before':>7}  {'After':>7}  {'MaxErr':>7}  "
                 f"{'MaxBlade':>8}  entry / exit")
    lines.append("   " + "-" * (W - 3))

    for rank, c in enumerate(result.get('single', []), 1):
        lines.append(
            f"   {rank:>4}  {c.cut_number:>4}  {c.split_pos:>6.1f}w"
            f"  {c.sec_before:>6.1f}w  {c.sec_after:>6.1f}w"
            f"  {c.max_err_deg:>6.2f}°"
            f"  {c.max_blade_deg:>7.2f}°"
            f"  {c.theta_entry_deg:.2f}° / {c.theta_exit_deg:.2f}°"
        )

    # ── Double correction ────────────────────────────────────────────────────
    ideal_double = sigma * math.sqrt(N / 3.0)
    lines.append("")
    lines.append("── Two correction points " + "─" * (W - 25))
    lines.append(f"   Ideal positions: wafers {N / 3:.1f} and {2 * N / 3:.1f}  "
                 f"→  max error = {ideal_double:.2f}° (1σ)")
    lines.append("")
    lines.append(f"   {'Rank':>4}  {'Cuts':>8}  "
                 f"{'Sec1':>6}  {'Sec2':>6}  {'Sec3':>6}  "
                 f"{'MaxErr':>7}  Blades (A / B)")
    lines.append("   " + "-" * (W - 3))

    for rank, c in enumerate(result.get('double', []), 1):
        lines.append(
            f"   {rank:>4}  {c.cut_a:>3},{c.cut_b:>4}"
            f"  {c.sec1:>5.1f}w  {c.sec2:>5.1f}w  {c.sec3:>5.1f}w"
            f"  {c.max_err_deg:>6.2f}°"
            f"  {c.max_blade_a:.2f}° / {c.max_blade_b:.2f}°"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("  MaxErr   = 1σ bound on accumulated spin error in the longest section.")
    lines.append("  MaxBlade = larger of entry/exit blade on the split wafer.")
    lines.append("             Higher values make the Z-axis mark easier to read.")
    lines.append("  Split is at the wafer midpoint (perpendicular cut).")
    lines.append("  The two halves become: (θ_entry, 0°, L/2) and (0°, θ_exit, Rot°+ε, L/2)")
    lines.append("  where ε is the measured correction rotation applied at build time.")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split-row generation and corrected cut list writer
# ---------------------------------------------------------------------------

def _fmt_frac(decimal_inches: float) -> str:
    """Format decimal inches as a fractional string (nearest 1/16")."""
    whole = int(decimal_inches)
    frac = decimal_inches - whole
    sixteenths = round(frac * 16)
    if sixteenths == 0:
        return f'{whole}"' if whole > 0 else '0"'
    if sixteenths == 16:
        return f'{whole + 1}"'
    num, den = sixteenths, 16
    while num % 2 == 0 and den % 2 == 0:
        num //= 2
        den //= 2
    return f'{whole} {num}/{den}"' if whole > 0 else f'{num}/{den}"'


def generate_split_rows(row: dict, regime_n: int) -> Tuple[dict, dict]:
    """Split one cut list row into two half-wafer rows for a correction point.

    Parameters
    ----------
    row : a cut list row dict (from parse_cut_list)
    regime_n : 1 or 2 — number of correction points in the overall build plan

    Returns
    -------
    (row_a, row_b) where:
      row_a — first half:  original entry blade, flat (0°) exit, no rotation, L/2
      row_b — second half: flat (0°) entry, original exit blade, original Rot°, L/2
              Rot° here is the NOMINAL value; builder adds measured ε at build time.

    Both rows carry a 'corr' field with the regime number.
    """
    blade = row.get('blade_deg', 0.0)
    theta_entry = row.get('theta_entry_deg', blade)
    theta_exit  = row.get('theta_exit_deg',  blade)
    rot         = row.get('rot_deg', 0.0)
    length      = row.get('length', 0.0)
    half_L      = length / 2.0

    base = {k: v for k, v in row.items()
            if k not in ('theta_entry_deg', 'theta_exit_deg', 'rot_deg',
                         'length', 'blade_deg', 'corr')}

    row_a = {
        **base,
        'length':         half_L,
        'blade_deg':      0.0,                 # exit face is perpendicular — this cut is flat
        'theta_entry_deg': theta_entry,
        'theta_exit_deg':  0.0,
        'rot_deg':         0.0,                # flat exit — no rotation at this cut
        'corr':            regime_n,
        '_split_half':     'A',
        '_orig_cut':       row.get('cut', '?'),
    }
    row_b = {
        **base,
        'length':         half_L,
        'blade_deg':      theta_exit,          # exit face is original exit angle
        'theta_entry_deg': 0.0,
        'theta_exit_deg':  theta_exit,
        'rot_deg':         rot,                # nominal; builder updates with measured ε
        'corr':            regime_n,
        '_split_half':     'B',
        '_orig_cut':       row.get('cut', '?'),
    }
    return row_a, row_b


def apply_corrections(rows: list,
                      split_wafer_0indices: List[int],
                      regime_n: int) -> list:
    """Return a new rows list with the specified wafers split into half-pairs.

    Parameters
    ----------
    rows : original cut list rows (0-based indexed list)
    split_wafer_0indices : 0-based indices of the wafers to split
    regime_n : 1 or 2 — written into the 'corr' field of split rows

    The output rows are renumbered sequentially from 1.  Non-split rows are
    unchanged except for the renumbered 'cut' field.  Split rows have their
    'cut' field set to the new sequential number and carry '_orig_cut',
    '_split_half', and 'corr' fields.
    """
    split_set = set(split_wafer_0indices)
    new_rows = []
    seq = 1
    for i, row in enumerate(rows):
        orig_num = row.get('cut', i + 1)   # preserve original model-tree number
        if i in split_set:
            ra, rb = generate_split_rows(row, regime_n)
            ra['cut'] = seq; ra['_orig_cut'] = orig_num; seq += 1
            rb['cut'] = seq; rb['_orig_cut'] = orig_num; seq += 1
            new_rows.append(ra)
            new_rows.append(rb)
        else:
            new_rows.append({**row, 'cut': seq, '_orig_cut': orig_num})
            seq += 1
    return new_rows


def _write_corrected_assembly_section(
        out_lines: list,
        segments_rows: list,
        sigma_blade: float,
        sigma_rot: float,
        radius_in: float) -> None:
    """Append an assembly/placement section with N resetting after each correction wafer."""
    radius_mm   = radius_in * 25.4
    sigma_b_rad = math.radians(sigma_blade)

    out_lines.append('\f')
    out_lines.append('=' * 90 + '\n')
    out_lines.append('ASSEMBLY / PLACEMENT LIST\n')
    out_lines.append('=' * 90 + '\n\n')
    out_lines.append('  Reference this list when gluing and assembling wafers.\n')
    out_lines.append('  Align marks face-to-face; Rot° gives the expected twist between wafers.\n')
    out_lines.append(f'  Build tolerances: Blade \u03c3 = \u00b1{sigma_blade:.3f}\u00b0'
                     f'   Rot \u03c3 = \u00b1{sigma_rot:.3f}\u00b0\n')
    out_lines.append('  Cumulative error grows as \u221aN; resets to zero at each correction wafer.\n\n')

    for seg_idx, seg_rows in enumerate(segments_rows):
        out_lines.append(f'\n--- Segment {seg_idx + 1} ---\n\n')
        out_lines.append(
            f"{'Wafer':<6} {'Length':<10} {'mm':<8} {'EntryBlade\u00b0':<12} {'ExitBlade\u00b0':<11} "
            f"{'Rot\u00b0':<9} {'\u03c3-Nrm\u00b0':<8} {'\u03c3-Spn\u00b0':<8} {'\u03c3-Lat mm':<9}\n"
        )
        out_lines.append('-' * 89 + '\n')

        n = 0   # error accumulator (resets at correction-B wafers)
        for row in seg_rows:
            n += 1
            blade   = row.get('blade_deg', 0.0)
            length  = row['length']
            t_entry = row.get('theta_entry_deg', blade)
            t_exit  = row.get('theta_exit_deg',  blade)
            rot     = row.get('rot_deg', 0.0)

            # Label: original model-tree number + 'A'/'B' suffix for split wafers
            orig_num   = row.get('_orig_cut', row.get('cut', n))
            split_half = row.get('_split_half', '')
            label      = f"{orig_num}{split_half}"

            length_str = _fmt_frac(length)
            mm_val     = length * 25.4

            sq_n          = math.sqrt(n)
            sigma_normal  = sigma_blade * sq_n
            sigma_spin    = sigma_rot   * sq_n
            sigma_lateral = radius_mm * math.sin(sigma_b_rad) * sq_n

            is_corr_b = split_half == 'B'
            note = '  \u2190 correction (error resets)' if is_corr_b else ''

            out_lines.append(
                f"{label:<6} {length_str:<10} {mm_val:<8.2f} {t_entry:<12.3f} "
                f"{t_exit:<11.3f} {rot:<9.3f} "
                f"{sigma_normal:<8.2f} {sigma_spin:<8.2f} {sigma_lateral:<9.2f}{note}\n"
            )

            if is_corr_b:
                n = 0   # error resets after correction; label numbering continues

        out_lines.append('\n')


def write_corrected_cut_list(source_path: str,
                             split_wafer_0indices: List[int],
                             regime_n: int,
                             output_path: Optional[str] = None) -> str:
    """Write a corrected cut list with the specified wafers split.

    The original file's header, segment metadata, and definitions are
    preserved.  The data table is replaced with the modified rows.  A
    'Corr' column is inserted before 'Done' on split rows; normal rows
    carry a blank in that column.

    Parameters
    ----------
    source_path : path to the original cut list .txt file
    split_wafer_0indices : 0-based wafer indices to split (from analyse_correction_points)
    regime_n : 1 or 2
    output_path : destination path; defaults to source_path with '_corrected' suffix

    Returns
    -------
    str : the output_path written
    """
    if output_path is None:
        base, ext = os.path.splitext(source_path)
        output_path = base + '_corrected' + ext

    with open(source_path, 'r') as fh:
        original_lines = fh.readlines()

    # Locate the data table: header line contains Cut/Length/Blade, data ends
    # at "Total cylinder length" or a blank section separator.
    header_pat = re.compile(r'^\s*Cut\s+(?:Length|SetLen)\s+.*Blade', re.IGNORECASE)
    total_pat  = re.compile(r'^\s*Total cylinder length', re.IGNORECASE)
    dash_pat   = re.compile(r'^-{10,}')

    # Find the column-header line
    header_line_idx = None
    for idx, line in enumerate(original_lines):
        if header_pat.match(line):
            header_line_idx = idx
            break

    if header_line_idx is None:
        raise ValueError(f"Could not find data table header in {source_path}")

    # Find the separator after the header (the ---... line)
    data_start_idx = header_line_idx + 1
    while data_start_idx < len(original_lines):
        if dash_pat.match(original_lines[data_start_idx]):
            data_start_idx += 1   # skip the dashes
            break
        data_start_idx += 1

    # Find end of data (Total cylinder length line or next ---...)
    data_end_idx = data_start_idx
    while data_end_idx < len(original_lines):
        line = original_lines[data_end_idx]
        if total_pat.match(line) or dash_pat.match(line):
            break
        data_end_idx += 1

    # Detect which optional columns are present in the original file's header.
    orig_header = original_lines[header_line_idx] if header_line_idx < len(original_lines) else ''
    has_set_len_orig = 'SetLen' in orig_header

    # Parse the original segment rows to get the full row data we need
    # (collinearity, azimuth, cylinder_angle are not in parsed rows, so
    # we extract them positionally from the original lines).
    orig_data_lines = original_lines[data_start_idx:data_end_idx]
    orig_row_map = {}   # cut_num → (collin, azimuth, cylinder_angle_str, cumul_str)
    for line in orig_data_lines:
        line = line.rstrip()
        if not line:
            continue
        tokens = line.split()
        if not tokens:
            continue
        try:
            cut_num = int(tokens[0])
        except ValueError:
            continue
        # Positions: 0=cut, 1=length, 2=mm, 3=blade, 4=entry, 5=exit, 6=rot,
        #            7=cylinder, 8=collin, 9=azimuth, 10=cumul, 11=done
        # (length may be 2 tokens for "1 3/8"", handled below)
        try:
            # Re-parse to find the right token positions
            idx = 1
            # consume length tokens
            while idx < len(tokens) and not tokens[idx].endswith('"'):
                idx += 1
            idx += 1   # past the closing " token
            has_mm  = False
            try:
                float(tokens[idx])
                has_mm = True
                idx += 1
            except (ValueError, IndexError):
                pass
            # Skip SetLen (fractional, 1-2 tokens) and SetMM (float) if present
            if has_set_len_orig:
                while idx < len(tokens) and not tokens[idx].endswith('"'):
                    idx += 1
                if idx < len(tokens):
                    idx += 1   # past closing '"' of SetLen
                try:
                    float(tokens[idx])
                    idx += 1   # SetMM float
                except (ValueError, IndexError):
                    pass
            idx += 1   # blade
            idx += 2   # entry, exit
            idx += 1   # rot
            cyl_str   = tokens[idx] if idx < len(tokens) else ''
            collin    = tokens[idx + 1] if idx + 1 < len(tokens) else ''
            azimuth   = tokens[idx + 2] if idx + 2 < len(tokens) else ''
            cumul_str = tokens[idx + 3] if idx + 3 < len(tokens) else ''
            orig_row_map[cut_num] = (cyl_str, collin, azimuth, cumul_str)
        except (IndexError, ValueError):
            orig_row_map[cut_num] = ('', '', '', '')

    # Build the modified rows
    from cut_list_reconstruction import parse_cut_list
    segments = parse_cut_list(source_path)
    if not segments:
        raise ValueError(f"No segments parsed from {source_path}")

    # Apply corrections to all segments (assume same split_indices apply to
    # each segment; for multi-segment files the caller can handle separately).
    new_segments_rows = []
    for seg in segments:
        new_segments_rows.append(apply_corrections(seg['rows'], split_wafer_0indices, regime_n))

    # Format a data row for the new table
    has_corr = bool(split_wafer_0indices)

    def _fmt_row(row, cumul_in, cum_rot, row_idx, cut_seq):
        length   = row['length']
        blade    = row.get('blade_deg', 0.0)
        t_entry  = row.get('theta_entry_deg', blade)
        corr_val = row.get('corr', '')
        orig_cut = row.get('_orig_cut', cut_seq)

        # Cylinder° — same parity logic as driver.py: odd rows get +180°
        if row_idx == 0:
            cyl_angle = 0.0
        elif row_idx % 2 == 1:
            cyl_angle = (cum_rot + 180.0) % 360.0
        else:
            cyl_angle = cum_rot % 360.0

        cumul_str = _fmt_frac(cumul_in)

        # Physical jig stop distance (same formula as driver.py)
        set_len     = length + cylinder_radius_in * math.tan(math.radians(t_entry))
        set_len_str = _fmt_frac(set_len)
        set_mm      = set_len * 25.4

        corr_col = f'{corr_val:<5}' if has_corr else ''
        split_note = ''
        if row.get('_split_half'):
            split_note = f"{orig_cut}{row['_split_half']}"   # e.g. "10A"

        label = f"{cut_seq:<4}"

        line = (f"{label} {set_len_str:<10} {set_mm:<8.2f} {blade:<8.3f} {cyl_angle:<10.0f} {'[ ]':<6} "
                f"{cumul_str:<12}{' ' + corr_col if has_corr else ''}")
        if split_note:
            line = line.rstrip() + f"  ({split_note})"
        return line

    # Extract initial_offset, cylinder radius, and sigma values from header.
    _offset_pat     = re.compile(r'Cylinder[°\u00b0]?\s+initial\s+offset[:\s]+([+-]?\d+\.?\d*)',
                                 re.IGNORECASE)
    _diam_pat       = re.compile(r'cylinder_diameter[:\s]+([0-9.]+)', re.IGNORECASE)
    _sigma_blade_pat = re.compile(r'Blade[°\u00b0]?\s+variance.*?[±+]([0-9.]+)', re.IGNORECASE)
    _sigma_rot_pat   = re.compile(r'Rot[°\u00b0]?\s+variance.*?[±+]([0-9.]+)', re.IGNORECASE)
    initial_offset = 0.0
    cylinder_radius_in = 0.0
    sigma_blade_deg = 0.5   # default matching cut list header
    sigma_rot_deg   = 2.0
    for line in original_lines[:header_line_idx]:
        m = _offset_pat.search(line)
        if m:
            try:
                initial_offset = float(m.group(1))
            except ValueError:
                pass
        m2 = _diam_pat.search(line)
        if m2:
            try:
                cylinder_radius_in = float(m2.group(1)) / 2.0
            except ValueError:
                pass
        m3 = _sigma_blade_pat.search(line)
        if m3:
            try:
                sigma_blade_deg = float(m3.group(1))
            except ValueError:
                pass
        m4 = _sigma_rot_pat.search(line)
        if m4:
            try:
                sigma_rot_deg = float(m4.group(1))
            except ValueError:
                pass

    # Assemble the new file
    out_lines = []

    # Lines before data header (preserve exactly)
    out_lines.extend(original_lines[:header_line_idx])

    # Note for builder about Cylinder° reference when using corrected list.
    neg_offset = (-initial_offset) % 360.0
    out_lines.append(
        f"  Corrected list note: Cylinder° values omit the initial offset.\n"
        f"  Before cut 1, set your angle indicator to {neg_offset:.1f}°\n"
        f"  (the negative of the initial offset above), then read Cylinder° directly.\n\n"
    )

    # New column header — page break then header; add Corr column if corrections present
    corr_col_hdr = f" {'Corr':<5}" if has_corr else ''
    out_lines.append('\f')   # page break before cut list data
    out_lines.append(
        f"{'Cut':<4} {'SetLen':<10} {'SetMM':<8} {'Blade°':<8} {'Cylinder°':<10} {'Done':<6} "
        f"{'Cumulative':<12}{corr_col_hdr}\n"
    )
    out_lines.append('-' * (64 + (6 if has_corr else 0)) + '\n')

    # Data rows for each segment
    for seg_rows in new_segments_rows:
        cumulative = 0.0
        cum_rot = 0.0
        for row_idx, row in enumerate(seg_rows):
            cumulative += row['length']
            cum_rot    += row.get('rot_deg', 0.0)
            out_lines.append(_fmt_row(row, cumulative, cum_rot, row_idx, row['cut']) + '\n')

    # Tail: lines from end of data table onward (Total length, definitions, etc.)
    # Strip any existing assembly section (begins at a form-feed line) so we can
    # regenerate it with correction-aware error reset.
    tail_lines = original_lines[data_end_idx:]
    formfeed_idx = next((i for i, l in enumerate(tail_lines) if '\f' in l), None)
    if formfeed_idx is not None:
        tail_lines = tail_lines[:formfeed_idx]

    out_lines.append('\n')
    out_lines.extend(tail_lines)

    # Regenerate assembly section with N resetting after each correction wafer.
    _write_corrected_assembly_section(
        out_lines, new_segments_rows, sigma_blade_deg, sigma_rot_deg, cylinder_radius_in
    )

    with open(output_path, 'w') as fh:
        fh.writelines(out_lines)

    return output_path
