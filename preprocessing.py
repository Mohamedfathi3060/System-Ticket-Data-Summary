"""
preprocessing.py — Data parsing, cleaning, and filtering for ticket data.

Handles the full pipeline from raw text file upload to a clean, analysis-ready DataFrame.
Designed to gracefully handle malformed files, missing columns, and NaN values.
"""

import pandas as pd
import io
from typing import Optional

from config import VALID_CATEGORIES, DATE_COLUMNS


def parse_ticket_file(uploaded_file) -> pd.DataFrame:
    """
    Parse an uploaded .txt file (CSV format) into a pandas DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object.

    Returns:
        pd.DataFrame: Raw parsed DataFrame.

    Raises:
        ValueError: If the file cannot be parsed as CSV.
    """
    try:
        # Read the file content and decode
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        # Parse CSV from string content
        df = pd.read_csv(io.StringIO(content), dtype=str, sep=None, engine='python', on_bad_lines='skip')

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        raise ValueError(f"Failed to parse the uploaded file: {str(e)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw ticket DataFrame.

    Steps:
        1. Replace 'N/A' strings with actual NaN values.
        2. Strip whitespace from all string columns.
        3. Filter to only valid SERVICE_CATEGORY values.
        4. Parse date columns to datetime format.
        5. Convert CUSTOMER_NUMBER to string (for consistent selection).

    Args:
        df: Raw DataFrame from parse_ticket_file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only valid categories.

    Raises:
        ValueError: If required columns are missing.
    """
    # Validate required columns exist
    required_columns = ["SERVICE_CATEGORY", "CUSTOMER_NUMBER", "ACCEPTANCE_TIME"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Please check that the uploaded file has the correct format."
        )

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Replace 'N/A' and empty strings with NaN
    df = df.replace(["N/A", "n/a", "N/a", "", " "], pd.NA)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", pd.NA)

    # Filter to valid service categories only
    df = df[df["SERVICE_CATEGORY"].isin(VALID_CATEGORIES)].copy()

    if df.empty:
        raise ValueError(
            f"No tickets found with valid categories ({', '.join(VALID_CATEGORIES)}). "
            f"Please check the data file."
        )

    # Parse date columns — errors='coerce' converts unparseable dates to NaT
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # Ensure CUSTOMER_NUMBER is string for consistent UI selection
    df["CUSTOMER_NUMBER"] = df["CUSTOMER_NUMBER"].astype(str).str.strip()

    # Sort by acceptance time for chronological processing
    df = df.sort_values("ACCEPTANCE_TIME", na_position="last").reset_index(drop=True)

    return df


def get_customer_list(df: pd.DataFrame) -> list:
    """
    Get a sorted list of unique customer numbers from the cleaned data.

    Args:
        df: Cleaned DataFrame.

    Returns:
        list: Sorted list of unique customer number strings.
    """
    return sorted(df["CUSTOMER_NUMBER"].dropna().unique().tolist())


def get_customer_tickets(
    df: pd.DataFrame, customer_number: str
) -> pd.DataFrame:
    """
    Get all tickets for a specific customer, sorted chronologically.

    Args:
        df: Cleaned DataFrame.
        customer_number: The customer number to filter by.

    Returns:
        pd.DataFrame: Tickets for the customer, sorted by ACCEPTANCE_TIME.
    """
    customer_df = df[df["CUSTOMER_NUMBER"] == str(customer_number)].copy()
    customer_df = customer_df.sort_values("ACCEPTANCE_TIME", na_position="last")
    return customer_df.reset_index(drop=True)


def compute_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a RESOLUTION_MINUTES column representing the time between
    ACCEPTANCE_TIME and COMPLETION_TIME in minutes.

    Args:
        df: DataFrame with parsed datetime columns.

    Returns:
        pd.DataFrame: DataFrame with RESOLUTION_MINUTES column added.
    """
    df = df.copy()
    if "ACCEPTANCE_TIME" in df.columns and "COMPLETION_TIME" in df.columns:
        delta = df["COMPLETION_TIME"] - df["ACCEPTANCE_TIME"]
        df["RESOLUTION_MINUTES"] = delta.dt.total_seconds() / 60.0
    return df


def convert_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.

    Args:
        df: DataFrame to convert.

    Returns:
        bytes: CSV content encoded as UTF-8 bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def convert_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to Excel bytes for download.

    Args:
        df: DataFrame to convert.

    Returns:
        bytes: Excel file content as bytes.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    return output.getvalue()
