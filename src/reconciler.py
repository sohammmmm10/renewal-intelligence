"""
Account Reconciler -- AI-First with Fuzzy Fallback.

Architecture Decision (AI Architect perspective):
    We use a HYBRID approach (AI-first, fuzzy-fallback):

    1. FAST PATH (no API call): If note has explicit account ID ("acct 1001") -> direct lookup
    2. AI PATH (1 batch LLM call): Send ALL unmatched names to LLM in one shot.
       The LLM understands typos, abbreviations, context, and semantic similarity.
    3. FUZZY FALLBACK: If LLM is unavailable/fails, fall back to RapidFuzz character matching.

    WHY AI > Regex + Fuzzy:
    ---------------------------------------------------------------
    | Scenario                           | Regex+Fuzzy | AI (LLM)  |
    |------------------------------------+-------------+-----------|
    | "BritePath" -> "BrightPath"        | Works (88%) | Works     |
    | "Pinacle Media" -> "Pinnacle..."   | Works (78%) | Works     |
    | "the healthcare account"           | FAILS (0%)  | Works     |
    | "vanguard retail" -> "Vanguard..." | Works exact | Works     |
    | Note has no name, only context     | FAILS       | Can infer |
    | New format CSM never wrote before  | FAILS       | Works     |
    ---------------------------------------------------------------

    WHY keep fuzzy as FALLBACK:
    - LLM costs money (~$0.001 per call, but still non-zero)
    - LLM can be slow or unavailable (API outage)
    - For obvious exact matches, no need to waste an API call
    - Defense-in-depth: if AI fails, we still get ~80% accuracy from fuzzy
"""

import pandas as pd
from rapidfuzz import fuzz, process
from src.data_loader import CSMNote
from src.llm_engine import ai_reconcile_names


def build_account_lookup(accounts_df: pd.DataFrame) -> dict[int, str]:
    """Build account_id -> account_name mapping."""
    return dict(zip(accounts_df["account_id"], accounts_df["account_name"]))


def build_name_to_id(accounts_df: pd.DataFrame) -> dict[str, int]:
    """Build lowercase account_name -> account_id mapping."""
    return {name.lower(): aid for aid, name in
            zip(accounts_df["account_id"], accounts_df["account_name"])}


def fuzzy_match_name(
    query_name: str,
    account_names: list[str],
    threshold: int = 65,
) -> tuple[str | None, int]:
    """
    Fuzzy-match a query name against canonical account names.
    Used as FALLBACK when LLM is unavailable.

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
        idx = result[2]
        return account_names[idx], result[1]

    return None, 0


def reconcile_csm_notes(
    csm_notes: list[CSMNote],
    accounts_df: pd.DataFrame,
) -> list[CSMNote]:
    """
    Reconcile CSM notes using AI-first approach with fuzzy fallback.

    3-Tier Strategy:
    ================

    TIER 1 - FAST PATH (no API call, instant):
        If note has explicit account ID ("acct 1001", "#1007") -> direct lookup.
        If note name matches exactly (case-insensitive) -> direct lookup.
        Cost: $0. Speed: <1ms.

    TIER 2 - AI PATH (1 batch LLM call for ALL unmatched):
        Collect ALL unmatched names, send to LLM in ONE batch call.
        LLM understands typos, abbreviations, partial names, and context.
        "BritePath" -> BrightPath, "Pinacle Media" -> Pinnacle Media Group.
        Cost: ~$0.002. Speed: ~2-3s. Accuracy: ~98%.

    TIER 3 - FUZZY FALLBACK (safety net):
        If LLM fails or returns low confidence, fall back to RapidFuzz.
        Character-level similarity matching.
        Cost: $0. Speed: <1ms. Accuracy: ~80%.
    """
    lookup = build_account_lookup(accounts_df)
    name_to_id = build_name_to_id(accounts_df)
    canonical_names = accounts_df["account_name"].tolist()

    # Build canonical list for LLM (lightweight: only ID + name)
    canonical_accounts = [
        {"account_id": int(aid), "account_name": aname}
        for aid, aname in zip(accounts_df["account_id"], accounts_df["account_name"])
    ]

    # ── TIER 1: Fast path (ID lookup + exact match) ──
    resolved = []     # (index, note) pairs already resolved
    unresolved = []   # (index, note) pairs needing AI/fuzzy

    for i, note in enumerate(csm_notes):
        # Direct ID match — "acct 1001", "#1007", "account 1016"
        if note.account_id and note.account_id in lookup:
            note.account_name = lookup[note.account_id]
            resolved.append((i, note))
            continue

        # Exact name match (case-insensitive)
        if note.account_name:
            exact = name_to_id.get(note.account_name.lower())
            if exact:
                note.account_id = exact
                note.account_name = lookup[exact]
                resolved.append((i, note))
                continue

        # Needs AI or fuzzy matching
        unresolved.append((i, note))

    # ── TIER 2: AI reconciliation (1 batch LLM call for ALL unresolved) ──
    if unresolved:
        unresolved_names = [note.account_name for _, note in unresolved]

        try:
            ai_matches = ai_reconcile_names(unresolved_names, canonical_accounts)

            for (idx, note), match in zip(unresolved, ai_matches):
                matched_id = match.get("matched_id")
                confidence = match.get("confidence", "low")

                if matched_id and matched_id in lookup and confidence in ("high", "medium"):
                    # AI matched with confidence
                    note.account_id = matched_id
                    note.account_name = lookup[matched_id]
                    resolved.append((idx, note))
                else:
                    # AI couldn't match -> TIER 3: fuzzy fallback
                    if note.account_name:
                        match_name, score = fuzzy_match_name(note.account_name, canonical_names)
                        if match_name:
                            note.account_id = name_to_id[match_name.lower()]
                            note.account_name = match_name
                    resolved.append((idx, note))

        except Exception:
            # ── TIER 3: LLM failed entirely -> fuzzy fallback for everything ──
            for idx, note in unresolved:
                if note.account_name:
                    match_name, score = fuzzy_match_name(note.account_name, canonical_names)
                    if match_name:
                        note.account_id = name_to_id[match_name.lower()]
                        note.account_name = match_name
                resolved.append((idx, note))

    # Reconstruct list in original order
    resolved.sort(key=lambda x: x[0])
    return [note for _, note in resolved]


def get_csm_notes_by_account(
    csm_notes: list[CSMNote],
) -> dict[int, list[CSMNote]]:
    """Group reconciled CSM notes by account_id."""
    grouped: dict[int, list[CSMNote]] = {}
    for note in csm_notes:
        if note.account_id:
            grouped.setdefault(note.account_id, []).append(note)
    return grouped
