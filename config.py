"""
config.py — Central configuration for the Streamlit Ticket Storytelling App.

Contains all constants, mappings, and configuration loaders used across modules.
Designed to be the single source of truth for business rules and app behavior.
"""

import json
import os
from pathlib import Path

# =============================================================================
# File Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
STYLE_CONFIG_PATH = BASE_DIR / "style_config.json"

# =============================================================================
# Valid Service Categories (tickets outside these are filtered out)
# =============================================================================
VALID_CATEGORIES = ["HDW", "NET", "KAI", "KAV", "GIGA", "VOD", "KAD"]

# =============================================================================
# Category → Product Mapping
# =============================================================================
CATEGORY_TO_PRODUCT = {
    "KAI": "Broadband",
    "NET": "Broadband",
    "KAV": "Voice",
    "KAD": "TV",
    "GIGA": "GIGA",
    "VOD": "VOD",
    "HDW": "Unmapped/Hardware",
}

# =============================================================================
# Date Columns — fields that should be parsed as datetime
# =============================================================================
DATE_COLUMNS = [
    "ACCEPTANCE_TIME",
    "COMPLETION_TIME",
    "CUSTOMER_COMPLETION_TIME",
    "PROCESSING_END_TIME_MAXIMUM",
    "PROCESSING_END_TIME_MINIMUM",
    "ACCEPTANCE_TIME_MINIMUM",
    "ASSIGNMENT_TIME_MINIMUM",
    "IMIL_TIME_MINIMUM",
    "CUSTOMER_TIME_MINIMUM",
    "START_TIME_MINIMUM",
    "ASSIGNMENT_TIME",
]

# =============================================================================
# Narrative-Relevant Fields — only these are sent to the LLM prompt
# Excludes pure routing/plumbing metadata (IDs, planning groups, etc.)
# =============================================================================
NARRATIVE_FIELDS = [
    "ORDER_NUMBER",
    "ACCEPTANCE_TIME",
    "COMPLETION_TIME",
    "CUSTOMER_COUNT"
    "PROCESSING_STATUS",
    "ORDER_TYPE",
    "ORDER_DESCRIPTION_1",
    "ORDER_DESCRIPTION_2",
    "ORDER_DESCRIPTION_3_MAXIMUM",
    "ADDITIONAL_ORDER_DESCRIPTION_MAXIMUM",
    "NOTE_MAXIMUM",
    "COMPLETION_RESULT_KB",
    "COMPLETION_NOTE_MAXIMUM",
    "CAUSE",
    "SERVICE_PROVIDER",
]

# =============================================================================
# Summary Phase Definitions
# =============================================================================
SUMMARY_PHASES = [
    {
        "name": "Initial Issue",
        "description": "Earliest tickets: nature of problem, customer feedback, immediate action taken.",
    },
    {
        "name": "Follow-ups",
        "description": "Subsequent related activity, customer interactions, support team responses.",
    },
    {
        "name": "Developments",
        "description": "New issues, resolution progress, changes in customer experience.",
    },
    {
        "name": "Later Incidents",
        "description": "Recurrence or new problems, handling, ongoing feedback.",
    },
    {
        "name": "Recent Events",
        "description": "Most recent status, resolutions, final customer feedback.",
    },
]

# =============================================================================
# Gemini LLM Configuration
# =============================================================================
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_TEMPERATURE = 0.4
GEMINI_MAX_OUTPUT_TOKENS = 4096

# =============================================================================
# Style Configuration Loader
# =============================================================================

DEFAULT_STYLE = {
    "page_title": "Ticket Data Storytelling",
    "page_icon": "🎫",
    "layout": "wide",
    "primary_color": "#6C63FF",
    "background_color": "#0E1117",
    "secondary_background_color": "#1A1F2E",
    "text_color": "#FAFAFA",
    "font": "Inter",
    "chart_color_palette": [
        "#6C63FF",
        "#FF6584",
        "#43E97B",
        "#F8D800",
        "#38F9D7",
        "#FA709A",
        "#FEE140",
        "#A18CD1",
    ],
    "chart_template": "plotly_dark",
    "card_border_radius": "12px",
    "card_background": "rgba(255, 255, 255, 0.05)",
    "card_border_color": "rgba(108, 99, 255, 0.3)",
    "accent_gradient": "linear-gradient(135deg, #6C63FF 0%, #FF6584 100%)",
    "phase_colors": {
        "Initial Issue": "#6C63FF",
        "Follow-ups": "#38F9D7",
        "Developments": "#43E97B",
        "Later Incidents": "#F8D800",
        "Recent Events": "#FF6584",
    },
}


def load_style_config() -> dict:
    """
    Load style configuration from style_config.json.
    Falls back to DEFAULT_STYLE if the file doesn't exist or is malformed.

    Returns:
        dict: Style configuration dictionary.
    """
    if STYLE_CONFIG_PATH.exists():
        try:
            with open(STYLE_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            # Merge with defaults: user config overrides defaults
            merged = {**DEFAULT_STYLE, **user_config}
            # Merge nested dicts (phase_colors)
            if "phase_colors" in user_config:
                merged["phase_colors"] = {
                    **DEFAULT_STYLE.get("phase_colors", {}),
                    **user_config["phase_colors"],
                }
            return merged
        except (json.JSONDecodeError, IOError):
            return DEFAULT_STYLE.copy()
    return DEFAULT_STYLE.copy()
