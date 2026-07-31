#!/usr/bin/env python3
"""Validate the governed platform-facility matrix before a surface can ship."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


REQUIRED_COLUMNS = (
    "Facility",
    "Target",
    "Status",
    "API/ABI authority",
    "std or pinned-suite exposure",
    "Manual symbols",
    "Minimum OS/ABI",
    "Safety invariants",
    "Fallback when unavailable",
    "Blocked feature or subcommand",
)
REQUIRED_TARGETS = frozenset(
    {
        "`x86_64-unknown-linux-gnu`",
        "`aarch64-unknown-linux-gnu`",
        "`aarch64-apple-darwin`",
        "`x86_64-apple-darwin`",
        "`x86_64-pc-windows-msvc`",
    }
)
VALID_STATUSES = frozenset({"available", "blocked", "off"})
G0_REFERENCE = re.compile(r"^\s*PLATFORM_SURFACE:\s*`?(.+?)`?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RegistryRow:
    line: int
    fields: dict[str, str]


def log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} PLATFORM_SURFACES {message}", file=sys.stderr)


def split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row while preserving escaped pipes in cells."""

    if not line.startswith("|") or not line.rstrip().endswith("|"):
        raise ValueError("row must begin and end with a table pipe")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.strip()[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_registry(document: Path) -> list[RegistryRow]:
    try:
        lines = document.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {document}: {error}") from error

    header_line = next(
        (
            number
            for number, line in enumerate(lines, start=1)
            if line.startswith("|") and split_markdown_row(line) == list(REQUIRED_COLUMNS)
        ),
        None,
    )
    if header_line is None:
        raise ValueError("registry table header is missing or differs from the required columns")
    if header_line >= len(lines):
        raise ValueError("registry table is missing its separator row")
    if not separator_row(split_markdown_row(lines[header_line])):
        raise ValueError("registry table has an invalid separator row")

    rows: list[RegistryRow] = []
    for number, line in enumerate(lines[header_line + 1 :], start=header_line + 2):
        if not line.startswith("|"):
            break
        cells = split_markdown_row(line)
        if len(cells) != len(REQUIRED_COLUMNS):
            raise ValueError(
                f"line {number} has {len(cells)} columns; expected {len(REQUIRED_COLUMNS)}"
            )
        rows.append(RegistryRow(number, dict(zip(REQUIRED_COLUMNS, cells, strict=True))))
    if not rows:
        raise ValueError("registry table has no facility rows")
    return rows


def validate_rows(rows: list[RegistryRow]) -> list[str]:
    incomplete: list[str] = []
    pairs: set[tuple[str, str]] = set()
    facilities: set[str] = set()
    targets: set[str] = set()

    for row in rows:
        fields = row.fields
        for column in REQUIRED_COLUMNS:
            if not fields[column]:
                incomplete.append(f"line{row.line}:{column}")
        facility = fields["Facility"]
        target = fields["Target"]
        status = fields["Status"]
        facilities.add(facility)
        targets.add(target)
        pair = (facility, target)
        if pair in pairs:
            incomplete.append(f"line{row.line}:duplicate:{facility}:{target}")
        pairs.add(pair)
        if status not in VALID_STATUSES:
            incomplete.append(f"line{row.line}:Status={status}")
        if target not in REQUIRED_TARGETS:
            incomplete.append(f"line{row.line}:Target={target}")
        if fields["Manual symbols"] != "NONE":
            incomplete.append(f"line{row.line}:Manual symbols")
        if status == "available":
            exposure = fields["std or pinned-suite exposure"].lower()
            if "std" not in exposure and "suite" not in exposure:
                incomplete.append(f"line{row.line}:available authority")
            if "no approved" in exposure:
                incomplete.append(f"line{row.line}:available exposure")
        if status == "blocked" and not fields["Blocked feature or subcommand"]:
            incomplete.append(f"line{row.line}:Blocked feature or subcommand")

    if targets != REQUIRED_TARGETS:
        missing = sorted(REQUIRED_TARGETS - targets)
        extra = sorted(targets - REQUIRED_TARGETS)
        incomplete.append(f"target_matrix:missing={missing}:extra={extra}")
    for facility in sorted(facilities):
        covered = {target for listed_facility, target in pairs if listed_facility == facility}
        if covered != REQUIRED_TARGETS:
            incomplete.append(f"facility_matrix:{facility}")
    return incomplete


def validate_g0_references(adr_directory: Path, facilities: set[str]) -> list[str]:
    references: list[str] = []
    adrs = sorted(adr_directory.glob("ADR-G0-*.md")) if adr_directory.is_dir() else []
    if not adrs:
        log("g0_cross_reference=SKIPPED_NO_G0_ADRS")
        return references

    for adr in adrs:
        try:
            contents = adr.read_text(encoding="utf-8")
        except OSError as error:
            references.append(f"{adr}:read_error:{error}")
            continue
        for match in G0_REFERENCE.finditer(contents):
            facility = match.group(1).strip()
            if facility not in facilities:
                references.append(f"{adr.name}:PLATFORM_SURFACE:{facility}")
    return references


def validate(document: Path, adr_directory: Path) -> tuple[list[RegistryRow], list[str]]:
    rows = parse_registry(document)
    incomplete = validate_rows(rows)
    incomplete.extend(validate_g0_references(adr_directory, {row.fields["Facility"] for row in rows}))
    return rows, sorted(set(incomplete))


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document",
        type=Path,
        default=repository / "docs" / "PLATFORM_SURFACES.md",
        help="platform-surface Markdown registry",
    )
    parser.add_argument(
        "--adr-directory",
        type=Path,
        default=repository / "docs" / "adr",
        help="G0 ADR directory for activated cross-reference checks",
    )
    args = parser.parse_args()

    try:
        rows, incomplete = validate(args.document, args.adr_directory)
    except ValueError as error:
        rows = []
        incomplete = [str(error)]

    for row in rows:
        log(
            "facility="
            f"{row.fields['Facility']} target={row.fields['Target']} "
            f"status={row.fields['Status']} line={row.line}"
        )
    if incomplete:
        for item in incomplete:
            log(f"incomplete={item}")
        print(
            "PLATFORM_SURFACES RESULT=FAIL "
            f"rows={len(rows)} incomplete={','.join(incomplete)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PLATFORM_SURFACES RESULT=PASS rows={len(rows)} incomplete=none",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
