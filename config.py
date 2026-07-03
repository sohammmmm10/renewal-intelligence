"""
Configuration module — loads settings from .env file.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")


class Config:
    """Central configuration for the Renewal Intelligence Engine."""

    # --- Paths ---
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"

    # --- Data Files ---
    ACCOUNTS_CSV = DATA_DIR / "accounts.csv"
    USAGE_CSV = DATA_DIR / "usage_metrics.csv"
    TICKETS_CSV = DATA_DIR / "support_tickets.csv"
    CSM_NOTES_TXT = DATA_DIR / "csm_notes.txt"
    NPS_CSV = DATA_DIR / "nps_responses.csv"
    CHANGELOG_MD = DATA_DIR / "changelog.md"

    # --- OpenAI ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Risk Scoring Weights (must sum to ~1.0) ---
    WEIGHT_USAGE_DECLINE = float(os.getenv("WEIGHT_USAGE_DECLINE", "0.25"))
    WEIGHT_SUPPORT_HEALTH = float(os.getenv("WEIGHT_SUPPORT_HEALTH", "0.15"))
    WEIGHT_NPS = float(os.getenv("WEIGHT_NPS", "0.10"))
    WEIGHT_CSM_SENTIMENT = float(os.getenv("WEIGHT_CSM_SENTIMENT", "0.25"))
    WEIGHT_SDK_RISK = float(os.getenv("WEIGHT_SDK_RISK", "0.15"))
    WEIGHT_ENGAGEMENT = float(os.getenv("WEIGHT_ENGAGEMENT", "0.10"))

    # --- Renewal Window ---
    RENEWAL_WINDOW_DAYS = int(os.getenv("RENEWAL_WINDOW_DAYS", "90"))

    # --- Dataset Snapshot Date ---
    # Fixed date for reproducible results. The dataset was created ~April 2026
    # (CSM notes are dated March-April 2026). Using April 10, 2026 as the
    # "today" reference so the 90-day renewal window correctly captures the
    # 27 accounts the assignment intended.
    # Override with DATASET_DATE env var (format: YYYY-MM-DD).
    DATASET_DATE = pd.Timestamp(os.getenv("DATASET_DATE", "2026-04-10"))

    # --- Risk Tiers ---
    # Calibrated thresholds: these represent churn probability.
    # 0.60+ = "more likely to churn than not, accounting for signal noise"
    # 0.35+ = "enough signals to warrant proactive attention"
    HIGH_RISK_THRESHOLD = 0.60
    MEDIUM_RISK_THRESHOLD = 0.35

    @classmethod
    def ensure_dirs(cls):
        """Create output directory if it doesn't exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate that required config values are set."""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY.startswith("sk-your"):
            raise ValueError(
                "OPENAI_API_KEY not set. Copy env_example.txt to .env and add your key."
            )
        for path in [cls.ACCOUNTS_CSV, cls.USAGE_CSV, cls.TICKETS_CSV,
                     cls.CSM_NOTES_TXT, cls.NPS_CSV, cls.CHANGELOG_MD]:
            if not path.exists():
                raise FileNotFoundError(f"Data file missing: {path}")
