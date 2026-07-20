"""
llm_summary.py — AI-powered storytelling summary generation using Gemini API.

Handles the full pipeline from ticket data to structured narrative summaries:
1. Build minimal ticket context objects (only narrative-relevant fields).
2. Split tickets into 5 chronological phases.
3. Construct a one-shot prompt with explicit multilingual handling.
4. Call Gemini API and parse the structured JSON response.
"""

import json
import os
import re
from typing import Dict, List, Optional

import streamlit as st
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

from config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    NARRATIVE_FIELDS,
    SUMMARY_PHASES,
    MAX_TICKETS_PER_SUMMARY,
    MAX_FIELD_LENGTH,
)

# Load environment variables from .env file
load_dotenv()


def _configure_gemini() -> None:
    """
    Configure the Gemini API with the API key from environment variables.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your .env file or set it in your environment."
        )
    genai.configure(api_key=api_key)


def build_ticket_context(tickets_df: pd.DataFrame) -> List[Dict]:
    """
    Extract only narrative-relevant fields from each ticket row.

    Excludes pure routing/plumbing metadata (ORDER_ID, ORDER_UNIT_ID, etc.) to keep
    the LLM prompt lean and focused.

    Args:
        tickets_df: DataFrame containing ticket data for one customer+product.

    Returns:
        List[Dict]: List of ticket context dictionaries with only narrative fields.
    """
    contexts = []

    # Sort to ensure we take ONLY the most recent N tickets to heavily protect against token waste/abuse
    # if a user uploads a wildly huge fake volume of tickets for one customer.
    recent_tickets = tickets_df.tail(MAX_TICKETS_PER_SUMMARY)

    for _, row in recent_tickets.iterrows():
        context = {}
        for field in NARRATIVE_FIELDS:
            if field in row.index:
                value = row[field]
                # Convert timestamps to readable strings
                if isinstance(value, pd.Timestamp):
                    context[field] = value.strftime("%Y-%m-%d %H:%M")
                elif pd.notna(value) and str(value).strip():
                    text_val = str(value).strip()
                    # Hard truncate any absurdly long text fields to save tokens
                    if len(text_val) > MAX_FIELD_LENGTH:
                        text_val = text_val[:MAX_FIELD_LENGTH] + "... [truncated]"
                    context[field] = text_val
                # Skip NaN/empty values to keep prompt minimal
        contexts.append(context)

    return contexts



def _build_system_prompt() -> str:
    """
    Build the system prompt for the Gemini LLM.

    Returns:
        str: System prompt string that instructs the LLM on its role and output format.
    """
    return """You are an expert support analyst who creates clear, professional storytelling summaries from customer ticket histories.

CRITICAL INSTRUCTIONS:
1. The ticket data may contain text in MULTIPLE LANGUAGES (English, German, or others). You MUST understand ALL languages present and produce your summary ENTIRELY IN ENGLISH regardless of the input language.
2. Translate any non-English ticket descriptions, notes, and statuses into English in your narrative.
3. You will be provided with a chronologically sorted list of tickets. YOU MUST group these tickets into exactly 5 storytelling phases: "Initial Issue", "Follow-ups", "Developments", "Later Incidents", and "Recent Events".
4. Produce a structured JSON response with exactly 5 sections (one for each phase in order).
5. Each section must have: "phase" (the phase name), "timeframe" (date range string), "ticket_numbers" (list of order numbers), and "narrative" (a coherent paragraph).
6. The narrative should tell a STORY — connect events, describe the customer's experience, explain what actions were taken and their outcomes.
7. If you genuinely believe a phase has no tickets that fit its description, still include it with an empty ticket list and a brief note like "No activity recorded during this period."
8. Be concise but informative. Each narrative should be 2-4 sentences.
9. Return ONLY valid JSON, no markdown formatting or extra text."""


def _build_user_prompt(
    customer_number: str,
    product: str,
    tickets: List[Dict],
) -> str:
    """
    Build the user prompt with ticket data and one-shot example.

    The prompt includes:
    - Customer and product context
    - A one-shot example showing the expected JSON format
    - The actual ticket data organized as a chronological list

    Args:
        customer_number: The customer identifier.
        product: The product category name.
        tickets: List of ticket context dictionaries.

    Returns:
        str: Complete user prompt string.
    """
    # One-shot example for the LLM to follow, featuring a non-empty, solid narrative for all phases
    one_shot_example = json.dumps(
        {
            "customer_number": "EXAMPLE-001",
            "product": "Broadband",
            "summary": [
                {
                    "phase": "Initial Issue",
                    "timeframe": "2024-11-05 to 2024-11-06",
                    "ticket_numbers": ["001-0671177/24"],
                    "narrative": "The customer first reported broadband connectivity issues on November 5th, experiencing slow WLAN speeds. A support ticket was raised and initial tests confirmed a synchronization drop at the local exchange."
                },
                {
                    "phase": "Follow-ups",
                    "timeframe": "2024-11-07 to 2024-11-08",
                    "ticket_numbers": ["001-0671178/24"],
                    "narrative": "A follow-up interaction occurred when the customer reported a complete absence of signal. The support team attempted a remote reset and re-provisioned the router, but the issue persisted."
                },
                {
                    "phase": "Developments",
                    "timeframe": "2024-11-10",
                    "ticket_numbers": ["001-0672001/24"],
                    "narrative": "The situation progressed as a field technician was dispatched to the customer's premises. The technician identified a faulty DSL line card in the street cabinet as the root cause."
                },
                {
                    "phase": "Later Incidents",
                    "timeframe": "2024-11-12",
                    "ticket_numbers": ["001-0672503/24", "001-0672510/24"],
                    "narrative": "Subsequent tickets showed the customer experienced intermittent drops after the repair. The network operations team was looped in to monitor the DSLAM stability, ensuring line parameters remained within bounds."
                },
                {
                    "phase": "Recent Events",
                    "timeframe": "2024-11-15",
                    "ticket_numbers": ["001-0673011/24"],
                    "narrative": "Most recently, the customer confirmed stable service. A final monitoring check showed 100% uptime over a 48-hour window, resulting in successful closure of the incident."
                }
            ],
        },
        indent=2,
    )

    # Build the actual ticket data section as a flat list
    ticket_data_str = ""
    if not tickets:
        ticket_data_str = "No tickets found.\n"
    else:
        for i, t in enumerate(tickets, 1):
            ticket_data_str += f"\nTicket {i}:\n"
            for key, value in t.items():
                ticket_data_str += f"  {key}: {value}\n"

    prompt = f"""Generate a storytelling summary for the following customer's ticket history.

**Customer Number:** {customer_number}
**Product Category:** {product}

---

**EXAMPLE OUTPUT FORMAT (follow this structure exactly):**
```json
{{
{one_shot_example[2:-2]}
}}
```

---

**ACTUAL TICKET DATA TO SUMMARIZE (Sorted Chronologically):**
{ticket_data_str}

---

Now generate the JSON summary for customer {customer_number}'s {product} tickets. Remember:
- Translate any non-English text to English in your narrative.
- You must distribute the provided tickets logically across the 5 phases based on their progression and chronology.
- Return ONLY valid JSON matching the example structure above.
- Use the actual customer number and product in the response.
- The "summary" array must contain exactly 5 objects."""

    return prompt


def generate_summary(
    customer_number: str,
    product: str,
    tickets_df: pd.DataFrame,
) -> Optional[Dict]:
    """
    Generate a storytelling summary for a customer's tickets in a product category.

    Pipeline:
        1. Build minimal ticket contexts (narrative fields only).
        2. Construct one-shot prompt (LLM manages phase grouping).
        3. Call Gemini API.
        4. Parse and return the JSON response.

    Args:
        customer_number: Customer identifier.
        product: Product category name.
        tickets_df: DataFrame of tickets for this customer+product, sorted by time.

    Returns:
        Optional[Dict]: Parsed summary dictionary with 5 phases, or None on failure.

    Raises:
        ValueError: If Gemini API key is not configured.
    """
    _configure_gemini()

    # Step 1: Build ticket context objects
    ticket_contexts = build_ticket_context(tickets_df)

    # Step 2: Build prompt
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(customer_number, product, ticket_contexts)

    # Step 3: Call Gemini API
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )

        response = model.generate_content(user_prompt)

        # Step 5: Parse the JSON response
        response_text = response.text.strip()

        # Try direct JSON parse
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            json_match = re.search(
                r"```(?:json)?\s*([\s\S]*?)\s*```", response_text
            )
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                raise json.JSONDecodeError(
                    "No valid JSON found in response", response_text, 0
                )

        return result

    except Exception as e:
        raise RuntimeError(f"LLM summary generation failed: {str(e)}")


def generate_insights(stats: Dict) -> str:
    """
    Generate business insights from aggregated ticket statistics using Gemini.

    Args:
        stats: Dictionary containing aggregated statistics:
            - total_tickets: int
            - categories: dict of category → count
            - avg_resolution_minutes: float
            - top_issues: list of (issue, count) tuples
            - customer_ticket_counts: dict of customer → count

    Returns:
        str: 2-3 actionable business insights as formatted text.
    """
    _configure_gemini()

    prompt = f"""Based on the following ticket statistics, generate 2-3 concise, actionable business insights. Focus on patterns that could improve customer service and operational efficiency.

**Statistics:**
- Total tickets analyzed: {stats.get('total_tickets', 'N/A')}
- Tickets by product category: {json.dumps(stats.get('categories', {}), indent=2)}
- Average resolution time: {stats.get('avg_resolution_minutes', 'N/A'):.1f} minutes
- Top reported issues: {json.dumps(stats.get('top_issues', []), indent=2)}
- Tickets per customer: {json.dumps(stats.get('customer_ticket_counts', {}), indent=2)}

NOTE: The data may contain multilingual entries (English, German). Provide all insights in English.

Format each insight as:
📊 **Insight Title**: Brief explanation with specific numbers from the data.

Return ONLY the formatted insights text, no JSON wrapper."""

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1024,
            ),
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"⚠️ Could not generate insights: {str(e)}"
