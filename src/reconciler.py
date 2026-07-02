"""
Account Reconciler — Fuzzy-matches messy account names to canonical account IDs.

Handles known data inconsistencies:
- "BritePath Solutions" → "BrightPath Solutions" (account 1001)
- "Pinacle Media" → "Pinnacle Media Group" (account 1004)
- "Thunderbolt Moters" → "Thunderbolt Motors" (account 1056)
- "vanguard retail" → "Vanguard Retail" (account 1005)
- "crescent labs" → "Crescent Labs" (account 1008)
- etc.
"""

import pandas as pd
from rapidfuzz import fuzz, process
from src.data_loader import CSMNote


def build_account_lookup(accounts_df: pd.DataFrame) -> dict[int, str]:
    """Build account_id → account_name mapping."""
    return dict(zip(accounts_df["account_id"], accounts_df["account_name"]))


def build_name_to_id(accounts_df: pd.DataFrame) -> dict[str, int]:
    """Build lowercase account_name → account_id mapping."""
    return {name.lower(): aid for aid, name in
            zip(accounts_df["account_id"], accounts_df["account_name"])}


def fuzzy_match_name(
    query_name: str,
    account_names: list[str],
    threshold: int = 65,
) -> tuple[str | None, int]:
    """
    Fuzzy-match a query name against the list of canonical account names.

    Returns (best_match_name, score) or (None, 0) if no good match.
    """
    if not query_name or len(query_name) < 3:
        return None, 0

    result = process.extractOne(
        query_name.lower(),
        [n.lower() for n in account_names],
        scorer=fuzz.token_sort_ratio,
    )

    if result and result[1] >= threshold:
        # Map back to original case
        idx = result[2]
        return account_names[idx], result[1]

    return None, 0


def reconcile_csm_notes(
    csm_notes: list[CSMNote],
    accounts_df: pd.DataFrame,
) -> list[CSMNote]:
    """
    Reconcile CSM notes by matching account names/IDs to canonical accounts.

    Strategy:
    1. If account_id is already present in the note, validate it
    2. If only account_name, fuzzy-match against accounts.csv
    3. Log any unmatched notes
    """
    lookup = build_account_lookup(accounts_df)
    name_to_id = build_name_to_id(accounts_df)
    canonical_names = accounts_df["account_name"].tolist()

    reconciled = []

    for note in csm_notes:
        # Case 1: Account ID is explicitly mentioned
        if note.account_id and note.account_id in lookup:
            # Validate and enrich with canonical name
            note.account_name = lookup[note.account_id]
            reconciled.append(note)
            continue

        # Case 2: Only have a name — try exact match first
        if note.account_name:
            exact = name_to_id.get(note.account_name.lower())
            if exact:
                note.account_id = exact
                note.account_name = lookup[exact]
                reconciled.append(note)
                continue

            # Case 3: Fuzzy match
            match_name, score = fuzzy_match_name(note.account_name, canonical_names)
            if match_name:
                note.account_id = name_to_id[match_name.lower()]
                original_name = note.account_name
                note.account_name = match_name
                reconciled.append(note)
                continue

        # Case 4: Could not reconcile — still include but flag it
        reconciled.append(note)

    return reconciled


def get_csm_notes_by_account(
    csm_notes: list[CSMNote],
) -> dict[int, list[CSMNote]]:
    """Group reconciled CSM notes by account_id."""
    grouped: dict[int, list[CSMNote]] = {}
    for note in csm_notes:
        if note.account_id:
            grouped.setdefault(note.account_id, []).append(note)
    return grouped
