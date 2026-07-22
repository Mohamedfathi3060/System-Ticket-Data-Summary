"""
app.py — Streamlit entrypoint for the Ticket Data Storytelling App.

This is the main application file that orchestrates:
1. File upload and data preprocessing
2. Customer selection and storytelling summary generation
3. Trends & Insights dashboard with interactive visualizations

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import logging

# =============================================================================
# Logging Configuration — output to both console and app.log
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from config import load_style_config, SUMMARY_PHASES, VALID_CATEGORIES, CATEGORY_TO_PRODUCT
from preprocessing import (
    parse_ticket_file,
    clean_data,
    get_customer_list,
    get_customer_tickets,
    compute_resolution_time,
    convert_to_csv,
    convert_to_excel,
)
from category_mapping import (
    apply_product_mapping,
    get_customer_products,
    get_category_emoji,
)
from llm_summary import generate_summary, generate_insights
from visualizations import (
    plot_ticket_volume_over_time,
    plot_avg_resolution_time,
    plot_common_issues,
    plot_recurring_tickets,
    plot_ticket_status_distribution,
)

# =============================================================================
# Page Configuration
# =============================================================================
style = load_style_config()

st.set_page_config(
    page_title=style["page_title"],
    page_icon=style["page_icon"],
    layout=style["layout"],
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS — Driven by style_config.json
# =============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family={style["font"]}:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {{
        font-family: '{style["font"]}', sans-serif;
    }}

    /* Header gradient */
    .main-header {{
        background: {style["accent_gradient"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    /* Subtitle */
    .sub-header {{
        color: {style["text_color"]}99;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}

    /* Metric cards */
    .metric-card {{
        background: {style["card_background"]};
        border: 1px solid {style["card_border_color"]};
        border-radius: {style["card_border_radius"]};
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        background: {style["accent_gradient"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        color: {style["text_color"]}88;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}

    /* Phase timeline card */
    .phase-card {{
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        background: {style["card_background"]};
        border-radius: 0 {style["card_border_radius"]} {style["card_border_radius"]} 0;
        transition: transform 0.2s ease;
    }}
    .phase-card:hover {{
        transform: translateX(4px);
    }}
    .phase-title {{
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }}
    .phase-timeframe {{
        color: {style["text_color"]}88;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }}
    .phase-tickets {{
        color: {style["primary_color"]};
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-family: monospace;
    }}
    .phase-narrative {{
        line-height: 1.6;
        color: {style["text_color"]}dd;
    }}

    /* Info banner */
    .info-banner {{
        background: linear-gradient(135deg, rgba(108,99,255,0.1) 0%, rgba(255,101,132,0.1) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: {style["card_border_radius"]};
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }}

    /* Divider */
    .styled-divider {{
        height: 2px;
        background: {style["accent_gradient"]};
        border: none;
        margin: 2rem 0;
        border-radius: 1px;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Session State Initialization
# =============================================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "summaries_cache" not in st.session_state:
    st.session_state.summaries_cache = {}


# =============================================================================
# Helper Functions
# =============================================================================


def render_metric_card(value: str, label: str) -> str:
    """Generate HTML for a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_phase_card(
    phase_name: str, timeframe: str, ticket_numbers: list, narrative: str, color: str
) -> str:
    """Generate HTML for a storytelling phase card."""
    tickets_str = ", ".join(ticket_numbers) if ticket_numbers else "—"
    return f"""
    <div class="phase-card" style="border-left-color: {color};">
        <div class="phase-title" style="color: {color};">📍 {phase_name}</div>
        <div class="phase-timeframe">🗓️ {timeframe}</div>
        <div class="phase-tickets">🎫 {tickets_str}</div>
        <div class="phase-narrative">{narrative}</div>
    </div>
    """


# =============================================================================
# Header
# =============================================================================
st.markdown('<div class="main-header">Ticket Data Storytelling</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered storytelling summaries from ticket data</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar — File Upload & Data Info
# =============================================================================
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload ticket data (.txt)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        help="Upload the raw ticket data text file(s) (CSV format).",
    )

    if uploaded_files:
        try:
            with st.spinner("Parsing file(s)..."):
                dfs = []
                for file in uploaded_files:
                    try:
                        dfs.append(parse_ticket_file(file))
                    except Exception as e:
                        st.error(f"❌ Failed to parse {file.name}: {str(e)}")
                
                if not dfs:
                    raise ValueError("No valid file could be parsed.")
                
                df_raw = pd.concat(dfs, ignore_index=True)
                st.session_state.df_raw = df_raw

            with st.spinner("Cleaning data..."):
                df_clean = clean_data(df_raw)
                df_clean = apply_product_mapping(df_clean)
                df_clean = compute_resolution_time(df_clean)
                st.session_state.df_clean = df_clean

            st.success(f"✅ Loaded {len(df_clean)} tickets (filtered from {len(df_raw)} rows)")

            # Data info
            st.markdown("---")
            st.markdown("### 📊 Data Overview")

            customers = get_customer_list(df_clean)
            categories = df_clean["PRODUCT"].nunique()
            st.markdown(f"- **Customers:** {len(customers)}")
            st.markdown(f"- **Product Categories:** {categories}")
            st.markdown(f"- **Date Range:** {df_clean['ACCEPTANCE_TIME'].min().strftime('%Y-%m-%d')} to {df_clean['ACCEPTANCE_TIME'].max().strftime('%Y-%m-%d')}")

            # Category filter info
            filtered_out = len(df_raw) - len(df_clean)
            if filtered_out > 0:
                st.info(f"ℹ️ {filtered_out} row(s) filtered out (categories not in {VALID_CATEGORIES})")

            # Download buttons
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 CSV",
                    data=convert_to_csv(df_clean),
                    file_name="tickets_cleaned.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    "📊 Excel",
                    data=convert_to_excel(df_clean),
                    file_name="tickets_cleaned.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.session_state.df_clean = None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.session_state.df_clean = None

# =============================================================================
# Main Content
# =============================================================================
if st.session_state.df_clean is not None:
    df = st.session_state.df_clean

    # Top-level metrics
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card(str(len(df)), "Total Tickets"), unsafe_allow_html=True)
    with m2:
        st.markdown(
            render_metric_card(str(df["CUSTOMER_NUMBER"].nunique()), "Customers"),
            unsafe_allow_html=True,
        )
    with m3:
        avg_res = df["RESOLUTION_MINUTES"].mean()
        st.markdown(
            render_metric_card(f"{avg_res:.0f} min", "Avg Resolution"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            render_metric_card(str(df["PRODUCT"].nunique()), "Product Categories"),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # Tabs
    tab_story, tab_insights = st.tabs(["📖 Customer Story", "📈 Trends & Insights"])

    # =================================================================
    # Tab 1: Customer Storytelling Summary
    # =================================================================
    with tab_story:
        st.markdown("### 📖 Customer Storytelling Summary")
        st.markdown(
            '<div class="info-banner">'
            "Select a customer to generate an AI-powered storytelling summary "
            "of their ticket history, organized by product category."
            "</div>",
            unsafe_allow_html=True,
        )

        customers = get_customer_list(df)
        selected_customer = st.selectbox(
            "Select Customer",
            options=customers,
            format_func=lambda x: f"Customer {x}",
        )

        if selected_customer:
            customer_tickets = get_customer_tickets(df, selected_customer)
            products = get_customer_products(df, selected_customer)

            st.markdown(
                f"**Customer {selected_customer}** has **{len(customer_tickets)} tickets** "
                f"across **{len(products)} product categories**: "
                + ", ".join([f"{get_category_emoji(p)} {p}" for p in products])
            )

            # Generate summaries button
            is_generating = st.session_state.get("generating_summaries", False)
            
            if st.button("🤖 Generate AI Summaries", type="primary", width="stretch", disabled=is_generating):
                st.session_state.generating_summaries = True
                st.rerun()
                
            if st.session_state.get("generating_summaries", False):
                try:
                    cache_key = selected_customer

                    for product in products:
                        product_tickets = customer_tickets[
                            customer_tickets["PRODUCT"] == product
                        ]
                        emoji = get_category_emoji(product)

                        with st.expander(
                            f"{emoji} {product} ({len(product_tickets)} tickets)",
                            expanded=True,
                        ):
                            summary_cache_key = f"{selected_customer}_{product}"

                            if summary_cache_key in st.session_state.summaries_cache:
                                summary = st.session_state.summaries_cache[summary_cache_key]
                            else:
                                with st.spinner(
                                    f"Generating {product} summary with Gemini AI..."
                                ):
                                    try:
                                        summary = generate_summary(
                                            selected_customer, product, product_tickets
                                        )
                                        st.session_state.summaries_cache[summary_cache_key] = summary
                                    except ValueError as e:
                                        st.error(f"⚙️ Configuration Error: {str(e)}")
                                        summary = None
                                    except RuntimeError as e:
                                        st.error(f"🤖 LLM Error: {str(e)}")
                                        summary = None
                                    except Exception as e:
                                        st.error(f"❌ Unexpected error: {str(e)}")
                                        summary = None

                            # Render the summary phases
                            if summary and "summary" in summary:
                                phase_colors = style.get("phase_colors", {})
                                for phase_data in summary["summary"]:
                                    phase_name = phase_data.get("phase", "Unknown")
                                    color = phase_colors.get(phase_name, style["primary_color"])
                                    st.markdown(
                                        render_phase_card(
                                            phase_name=phase_name,
                                            timeframe=phase_data.get("timeframe", "N/A"),
                                            ticket_numbers=phase_data.get("ticket_numbers", []),
                                            narrative=phase_data.get("narrative", "No data available."),
                                            color=color,
                                        ),
                                        unsafe_allow_html=True,
                                    )
                            elif summary is not None:
                                st.warning("⚠️ Summary format was unexpected. Raw response:")
                                st.json(summary)
                finally:
                    st.session_state.generating_summaries = False

            # Show raw data expander
            with st.expander("📋 View Raw Ticket Data", expanded=False):
                display_cols = [
                    "ORDER_NUMBER", "ACCEPTANCE_TIME", "COMPLETION_TIME",
                    "SERVICE_CATEGORY", "PRODUCT", "PROCESSING_STATUS",
                    "ORDER_DESCRIPTION_1", "ORDER_DESCRIPTION_2",
                    "NOTE_MAXIMUM", "COMPLETION_RESULT_KB",
                ]
                available_cols = [c for c in display_cols if c in customer_tickets.columns]
                st.dataframe(
                    customer_tickets[available_cols],
                    width="stretch",
                    hide_index=True,
                )

    # =================================================================
    # Tab 2: Trends & Insights Dashboard
    # =================================================================
    with tab_insights:
        st.markdown("### 📈 Trends & Insights Dashboard")
        st.markdown(
            '<div class="info-banner">'
            "Interactive visualizations showing patterns and trends in the ticket data. "
            "Use these insights to improve customer service and operational efficiency."
            "</div>",
            unsafe_allow_html=True,
        )

        # Row 1: Volume + Resolution
        col1, col2 = st.columns(2)

        with col1:
            fig_volume = plot_ticket_volume_over_time(df)
            st.plotly_chart(fig_volume, width="stretch")

        with col2:
            fig_resolution = plot_avg_resolution_time(df)
            st.plotly_chart(fig_resolution, width="stretch")

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

        # Row 2: Common Issues + Recurring Tickets
        col3, col4 = st.columns(2)

        with col3:
            fig_issues = plot_common_issues(df)
            st.plotly_chart(fig_issues, width="stretch")

        with col4:
            fig_recurring = plot_recurring_tickets(df)
            st.plotly_chart(fig_recurring, width="stretch")

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

        # Row 3: Status Distribution
        col5, col6 = st.columns(2)

        with col5:
            fig_status = plot_ticket_status_distribution(df)
            st.plotly_chart(fig_status, width="stretch")

        # AI Business Insights
        with col6:
            st.markdown("#### 🧠 AI-Generated Business Insights")
            if st.button("Generate Insights", type="secondary", width="stretch"):
                with st.spinner("Analyzing trends with Gemini AI..."):
                    try:
                        # Build stats for the insights prompt
                        stats = {
                            "total_tickets": len(df),
                            "categories": df["PRODUCT"].value_counts().to_dict(),
                            "avg_resolution_minutes": df["RESOLUTION_MINUTES"].mean()
                            if "RESOLUTION_MINUTES" in df.columns
                            else 0,
                            "top_issues": df["ORDER_DESCRIPTION_1"]
                            .value_counts()
                            .head(5)
                            .to_dict(),
                            "customer_ticket_counts": df["CUSTOMER_NUMBER"]
                            .value_counts()
                            .to_dict(),
                        }
                        insights_text = generate_insights(stats)
                        st.markdown(insights_text)
                    except Exception as e:
                        st.error(f"❌ Failed to generate insights: {str(e)}")

        # Data table
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        with st.expander("📋 Full Cleaned Dataset", expanded=False):
            st.dataframe(df, width="stretch", hide_index=True)

else:
    # No data loaded — show welcome screen
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📁</div>
            <h2 style="color: #FAFAFA;">Upload your ticket data to get started</h2>
            <p style="color: #FAFAFA88; max-width: 500px; margin: 1rem auto;">
                Use the sidebar to upload a <code>.txt</code> file containing your ticket
                data. The app will clean, analyze, and generate AI-powered storytelling
                summaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
