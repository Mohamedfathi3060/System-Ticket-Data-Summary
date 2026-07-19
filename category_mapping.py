"""
category_mapping.py — Maps SERVICE_CATEGORY codes to product names.

Applies the business-defined mapping from config.py to add a PRODUCT column.
"""

import pandas as pd
from typing import List

from config import CATEGORY_TO_PRODUCT


def apply_product_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a PRODUCT column to the DataFrame by mapping SERVICE_CATEGORY.

    Categories not found in the mapping are labeled "Unknown".

    Args:
        df: Cleaned DataFrame with SERVICE_CATEGORY column.

    Returns:
        pd.DataFrame: DataFrame with PRODUCT column added.
    """
    df = df.copy()
    df["PRODUCT"] = df["SERVICE_CATEGORY"].map(CATEGORY_TO_PRODUCT).fillna("Unknown")
    return df


def get_product_categories(df: pd.DataFrame) -> List[str]:
    """
    Get a sorted list of unique product categories present in the data.

    Args:
        df: DataFrame with PRODUCT column.

    Returns:
        List[str]: Sorted list of unique product names.
    """
    if "PRODUCT" not in df.columns:
        return []
    return sorted(df["PRODUCT"].dropna().unique().tolist())


def get_customer_products(df: pd.DataFrame, customer_number: str) -> List[str]:
    """
    Get the list of product categories a specific customer has tickets for.

    Args:
        df: DataFrame with PRODUCT and CUSTOMER_NUMBER columns.
        customer_number: Customer number to filter by.

    Returns:
        List[str]: Sorted list of product categories for this customer.
    """
    customer_df = df[df["CUSTOMER_NUMBER"] == str(customer_number)]
    if "PRODUCT" not in customer_df.columns:
        return []
    return sorted(customer_df["PRODUCT"].dropna().unique().tolist())


def get_category_emoji(product: str) -> str:
    """
    Return a display emoji for each product category for UI enhancement.

    Args:
        product: Product category name.

    Returns:
        str: Emoji string for the category.
    """
    emoji_map = {
        "Broadband": "🌐",
        "Voice": "📞",
        "TV": "📺",
        "GIGA": "⚡",
        "VOD": "🎬",
        "Unmapped/Hardware": "🔧",
        "Unknown": "❓",
    }
    return emoji_map.get(product, "📋")
