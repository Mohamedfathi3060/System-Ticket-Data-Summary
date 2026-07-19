"""
visualizations.py — Plotly-based data visualizations for ticket analytics.

All charts dynamically render labels from the data (handles multilingual
content natively since Plotly renders Unicode). Chart styling is driven
by the external style_config.json for easy customization.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

from config import load_style_config


def _get_style() -> dict:
    """Load style config once per visualization set."""
    return load_style_config()


def plot_ticket_volume_over_time(
    df: pd.DataFrame, by_product: bool = True
) -> go.Figure:
    """
    Create a ticket volume over time chart.

    Shows daily ticket counts, optionally broken down by product category.
    Handles multilingual category names dynamically.

    Args:
        df: Cleaned DataFrame with ACCEPTANCE_TIME and PRODUCT columns.
        by_product: If True, color-code by product category.

    Returns:
        go.Figure: Plotly figure object.
    """
    style = _get_style()

    if "ACCEPTANCE_TIME" not in df.columns:
        return _empty_figure("No date data available")

    plot_df = df.copy()
    plot_df["Date"] = plot_df["ACCEPTANCE_TIME"].dt.date

    if by_product and "PRODUCT" in plot_df.columns:
        # Group by date and product
        volume = (
            plot_df.groupby(["Date", "PRODUCT"])
            .size()
            .reset_index(name="Ticket Count")
        )
        fig = px.bar(
            volume,
            x="Date",
            y="Ticket Count",
            color="PRODUCT",
            title="📈 Ticket Volume Over Time by Product",
            color_discrete_sequence=style["chart_color_palette"],
            template=style["chart_template"],
            barmode="stack",
        )
    else:
        # Overall volume
        volume = plot_df.groupby("Date").size().reset_index(name="Ticket Count")
        fig = px.bar(
            volume,
            x="Date",
            y="Ticket Count",
            title="📈 Overall Ticket Volume Over Time",
            color_discrete_sequence=style["chart_color_palette"],
            template=style["chart_template"],
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tickets",
        legend_title="Product",
        hovermode="x unified",
        font=dict(family=style["font"]),
    )

    return fig


def plot_avg_resolution_time(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing average resolution time by product category.

    Resolution time = COMPLETION_TIME - ACCEPTANCE_TIME in minutes.

    Args:
        df: DataFrame with RESOLUTION_MINUTES and PRODUCT columns.

    Returns:
        go.Figure: Plotly figure object.
    """
    style = _get_style()

    if "RESOLUTION_MINUTES" not in df.columns or "PRODUCT" not in df.columns:
        return _empty_figure("Resolution time data not available")

    avg_res = (
        df.groupby("PRODUCT")["RESOLUTION_MINUTES"]
        .mean()
        .reset_index()
        .rename(columns={"RESOLUTION_MINUTES": "Avg Resolution (min)"})
        .sort_values("Avg Resolution (min)", ascending=True)
    )

    fig = px.bar(
        avg_res,
        x="Avg Resolution (min)",
        y="PRODUCT",
        orientation="h",
        title="⏱️ Average Resolution Time by Product",
        color="PRODUCT",
        color_discrete_sequence=style["chart_color_palette"],
        template=style["chart_template"],
        text="Avg Resolution (min)",
    )

    fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    fig.update_layout(
        xaxis_title="Average Resolution Time (minutes)",
        yaxis_title="Product Category",
        showlegend=False,
        font=dict(family=style["font"]),
    )

    return fig


def plot_common_issues(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart of the most frequent issue types.

    Uses ORDER_DESCRIPTION_1 as the primary issue classifier, with
    CAUSE as a secondary axis. Handles multilingual labels dynamically —
    labels are displayed as they appear in the data.

    Args:
        df: DataFrame with ORDER_DESCRIPTION_1 and/or CAUSE columns.
        top_n: Number of top issues to display.

    Returns:
        go.Figure: Plotly figure object.
    """
    style = _get_style()

    # Combine issue description and cause for richer insights
    issues = []
    if "ORDER_DESCRIPTION_1" in df.columns:
        desc_counts = (
            df["ORDER_DESCRIPTION_1"]
            .dropna()
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        desc_counts.columns = ["Issue", "Count"]
        desc_counts["Source"] = "Description"
        issues.append(desc_counts)

    if "CAUSE" in df.columns:
        cause_counts = (
            df["CAUSE"].dropna().value_counts().head(top_n).reset_index()
        )
        cause_counts.columns = ["Issue", "Count"]
        cause_counts["Source"] = "Cause"
        issues.append(cause_counts)

    if not issues:
        return _empty_figure("No issue data available")

    combined = pd.concat(issues, ignore_index=True)

    fig = px.bar(
        combined,
        x="Count",
        y="Issue",
        color="Source",
        orientation="h",
        title="🔍 Most Common Issue Types",
        color_discrete_sequence=style["chart_color_palette"],
        template=style["chart_template"],
        barmode="group",
    )

    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        legend_title="Source",
        font=dict(family=style["font"]),
        height=max(400, len(combined) * 30),
    )

    return fig


def plot_recurring_tickets(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing ticket frequency per customer by product category.

    Helps identify customers with recurring/repeat issues.

    Args:
        df: DataFrame with CUSTOMER_NUMBER and PRODUCT columns.

    Returns:
        go.Figure: Plotly figure object.
    """
    style = _get_style()

    if "CUSTOMER_NUMBER" not in df.columns or "PRODUCT" not in df.columns:
        return _empty_figure("Customer/product data not available")

    repeat_df = (
        df.groupby(["CUSTOMER_NUMBER", "PRODUCT"])
        .size()
        .reset_index(name="Ticket Count")
    )

    fig = px.bar(
        repeat_df,
        x="CUSTOMER_NUMBER",
        y="Ticket Count",
        color="PRODUCT",
        title="🔄 Ticket Frequency per Customer by Product",
        color_discrete_sequence=style["chart_color_palette"],
        template=style["chart_template"],
        barmode="group",
        text="Ticket Count",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Customer Number",
        yaxis_title="Number of Tickets",
        legend_title="Product",
        font=dict(family=style["font"]),
    )

    return fig


def plot_ticket_status_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie/donut chart showing the distribution of processing statuses.

    Args:
        df: DataFrame with PROCESSING_STATUS column.

    Returns:
        go.Figure: Plotly figure object.
    """
    style = _get_style()

    if "PROCESSING_STATUS" not in df.columns:
        return _empty_figure("No status data available")

    status_counts = (
        df["PROCESSING_STATUS"].dropna().value_counts().reset_index()
    )
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        values="Count",
        names="Status",
        title="📊 Ticket Status Distribution",
        color_discrete_sequence=style["chart_color_palette"],
        template=style["chart_template"],
        hole=0.4,
    )

    fig.update_layout(font=dict(family=style["font"]))

    return fig


def _empty_figure(message: str) -> go.Figure:
    """
    Create an empty figure with a centered message.

    Used when data is insufficient for a visualization.

    Args:
        message: Message to display.

    Returns:
        go.Figure: Empty figure with annotation.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig
