"""
Data Loader — Reads and parses all 5 data sources + changelog.

Handles:
- CSV loading with proper types
- CSM notes parsing (unstructured text → structured entries)
- Changelog parsing (markdown → structured deprecation/breaking-change list)
"""

import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from config import Config


@dataclass
class CSMNote:
    """A single parsed CSM call note."""
    raw_text: str
    date_str: str = ""
    account_name: str = ""
    account_id: int | None = None
    csm_name: str = ""


@dataclass
class ChangelogEntry:
    """A single parsed changelog item."""
    version: str
    date: str
    category: str          # "deprecation", "breaking_change", "security", "bug_fix", "new_feature"
    description: str
    affected_versions: list[str] = field(default_factory=list)
    deadline: str = ""     # e.g., "April 30, 2026"


def load_accounts(path: Path = Config.ACCOUNTS_CSV) -> pd.DataFrame:
    """Load accounts.csv with proper types."""
    df = pd.read_csv(path)
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"])
    df["arr"] = df["arr"].astype(int)
    df["account_id"] = df["account_id"].astype(int)
    return df


def load_usage_metrics(path: Path = Config.USAGE_CSV) -> pd.DataFrame:
    """Load usage_metrics.csv with proper types."""
    df = pd.read_csv(path)
    df["account_id"] = df["account_id"].astype(int)
    return df


def load_support_tickets(path: Path = Config.TICKETS_CSV) -> pd.DataFrame:
    """Load support_tickets.csv with proper types."""
    df = pd.read_csv(path)
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["account_id"] = df["account_id"].astype(int)
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    return df


def load_nps_responses(path: Path = Config.NPS_CSV) -> pd.DataFrame:
    """Load nps_responses.csv."""
    df = pd.read_csv(path)
    df["account_id"] = df["account_id"].astype(int)
    df["score"] = df["score"].astype(int)
    df["verbatim_comment"] = df["verbatim_comment"].fillna("")
    return df


def parse_csm_notes(path: Path = Config.CSM_NOTES_TXT) -> list[CSMNote]:
    """
    Parse the messy CSM notes text file into structured entries.

    The notes have inconsistent formatting:
    - "Mar 12 - Acme Corp."
    - "3/15 acct 1001 - BritePath Solutions"
    - "2026-03-20 | NovaTech Industries | James O."
    - "march 25 -- meridian health -- priya"
    """
    text = path.read_text(encoding="utf-8")

    # Split on the "---" separator lines
    raw_blocks = re.split(r"\n---+\n", text)
    notes = []

    for block in raw_blocks:
        block = block.strip()
        if not block or block.startswith("=== CSM"):
            continue

        note = CSMNote(raw_text=block)

        # Try to extract account ID like "acct 1001" or "(1004)" or "#1007" or "account 1016"
        id_match = re.search(r"(?:acct|account|#)\s*(\d{4})", block, re.IGNORECASE)
        if id_match:
            note.account_id = int(id_match.group(1))

        # Try to extract account name from first line
        first_line = block.split("\n")[0]

        # Pattern: various date formats followed by separator and account name
        # "Mar 12 - Acme Corp."
        # "3/15 acct 1001 - BritePath Solutions (sic) call"
        # "2026-03-20 | NovaTech Industries | James O."
        # "march 25 -- meridian health -- priya"
        name_patterns = [
            r"\d{4}-\d{2}-\d{2}\s*\|\s*([^|]+?)(?:\s*\||\s*$)",     # ISO date | Name |
            r"(?:Mar|Apr|Jan|Feb|march|april)\s+\d+\s*[-–]+\s*(.+?)(?:\s*[-–(]|$)",  # "Mar 12 - Name"
            r"\d+/\d+\s+(?:acct\s+\d+\s*[-–]+\s*)?(.+?)(?:\s+(?:call|sic|\()|$)",    # "3/15 acct 1001 - Name"
            r"^\d{2}/\d{2}\s+(.+?)(?:\s*$)",                         # "04/03 Atlas Financial"
        ]

        for pattern in name_patterns:
            m = re.search(pattern, first_line, re.IGNORECASE)
            if m:
                name = m.group(1).strip().rstrip(".")
                # Clean up: remove "(sic)", "call", etc.
                name = re.sub(r"\s*\(sic\).*", "", name)
                name = re.sub(r"\s+call\b.*", "", name, flags=re.IGNORECASE)
                if len(name) > 3:  # Skip too-short matches
                    note.account_name = name
                    break

        # If we still don't have a name, try a simpler approach for the first line
        if not note.account_name:
            # Remove date-like prefixes and grab what's left
            cleaned = re.sub(r"^[\d/\-\s]+(?:acct\s+\d+\s*[-–]+\s*)?", "", first_line, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*[-–]+\s*.*$", "", cleaned)  # Remove trailing " -- priya" etc.
            cleaned = cleaned.strip().rstrip(".")
            if len(cleaned) > 3:
                note.account_name = cleaned

        notes.append(note)

    return notes


def parse_changelog(path: Path = Config.CHANGELOG_MD) -> list[ChangelogEntry]:
    """
    Parse changelog.md into structured entries, focusing on:
    - Deprecations (SDK sunset dates, API changes)
    - Breaking changes
    - Security patches
    - Bug fixes
    """
    text = path.read_text(encoding="utf-8")
    entries = []

    # Split by version headers (### vX.X.X)
    version_blocks = re.split(r"### (v[\d.]+\s*[-—–]+?\s*.+?)(?=\n)", text)

    current_version = ""
    current_date = ""

    for i, block in enumerate(version_blocks):
        # Check if this is a version header
        version_match = re.match(r"(v[\d.]+)\s*[-—–]+?\s*(.+)", block.strip())
        if version_match:
            current_version = version_match.group(1)
            current_date = version_match.group(2).strip()
            continue

        if not current_version:
            continue

        # Extract deprecation entries
        for m in re.finditer(r"[⚠️🔴]\s*(.+?)(?:\n|$)", block):
            desc = m.group(1).strip()
            entry = ChangelogEntry(
                version=current_version,
                date=current_date,
                category="deprecation" if "deprecat" in desc.lower() or "sunset" in desc.lower() else "breaking_change",
                description=desc,
            )
            # Extract affected SDK versions
            sdk_matches = re.findall(r"(?:SDK|v)(\d+\.\w+(?:\.\w+)?)", desc)
            entry.affected_versions = [f"v{v}" for v in sdk_matches]

            # Extract deadlines
            deadline_match = re.search(
                r"(?:before|by|after|until)\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,?\s*\d{4})",
                desc, re.IGNORECASE
            )
            if deadline_match:
                entry.deadline = deadline_match.group(1)

            entries.append(entry)

        # Extract security patches
        for m in re.finditer(r"(?:Patched|Fixed|CVE).*?(?:CVE-[\d-]+).*?(?:\n|$)", block):
            entries.append(ChangelogEntry(
                version=current_version,
                date=current_date,
                category="security",
                description=m.group(0).strip(),
            ))

        # Extract breaking changes
        for m in re.finditer(r"🔴\s*(.+?)(?:\n|$)", block):
            desc = m.group(1).strip()
            if "breaking" in desc.lower() or "changes the response" in desc.lower():
                entry = ChangelogEntry(
                    version=current_version,
                    date=current_date,
                    category="breaking_change",
                    description=desc,
                )
                sdk_matches = re.findall(r"v(\d+\.\w+(?:\.\w+)?)", desc)
                entry.affected_versions = [f"v{v}" for v in sdk_matches]
                entries.append(entry)

    return entries


def load_all_data() -> dict:
    """Load all data sources and return as a dictionary."""
    return {
        "accounts": load_accounts(),
        "usage": load_usage_metrics(),
        "tickets": load_support_tickets(),
        "nps": load_nps_responses(),
        "csm_notes": parse_csm_notes(),
        "changelog": parse_changelog(),
    }
