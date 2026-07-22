"""
llm_summary.py — AI-powered storytelling summary generation using Gemini API.

Handles the full pipeline from ticket data to structured narrative summaries:
1. Build minimal ticket context objects (only narrative-relevant fields).
2. Split tickets into 5 chronological phases.
3. Construct a one-shot prompt with explicit multilingual handling.
4. Call Gemini API and parse the structured JSON response.

Uses the modern `google-genai` SDK (replaces deprecated `google-generativeai`).
"""

import json
import os
import re
import time
import logging
from typing import Dict, List, Optional

import streamlit as st
from google import genai
from google.genai import types
from langsmith import traceable
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

# Automatically sync Langchain keys from Streamlit Secrets to os.environ 
try:
    for _key in ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGSMITH_ENDPOINT"]:
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except FileNotFoundError:
    pass
    

# Set up logging for background tracking
logger = logging.getLogger(__name__)

# Module-level Gemini client (initialized once)
_gemini_client: Optional[genai.Client] = None

# Timeout for Gemini API calls (in seconds)
GEMINI_TIMEOUT_SECONDS = 120


def _get_gemini_client() -> genai.Client:
    """
    Get or create the Gemini API client (singleton pattern).

    Uses the API key from environment variables or Streamlit secrets.
    Configures HTTP timeouts to prevent indefinite hangs.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.

    Returns:
        genai.Client: Configured Gemini client instance.
    """
    global _gemini_client

    if _gemini_client is not None:
        logger.debug("Reusing existing Gemini client instance.")
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment or Streamlit secrets.")
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your .env file or set it in your environment."
        )

    logger.info("Initializing Gemini client (model=%s, timeout=%ds).", GEMINI_MODEL, GEMINI_TIMEOUT_SECONDS)

    _gemini_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT_SECONDS * 1000,  # milliseconds
        ),
    )

    logger.info("Gemini client initialized successfully.")
    return _gemini_client


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
    logger.debug("Building ticket context: %d tickets (capped from %d).", len(recent_tickets), len(tickets_df))

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

    logger.debug("Built %d ticket contexts.", len(contexts))
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
5. Each section must have: "phase" (the phase name), "timeframe" (date range string), "ticket_numbers" (a list of EXACT `ORDER_NUMBER` identifiers from the provided dataset. DO NOT invent arbitrary sequential tags like "Ticket 1" or "T-01"), and "narrative" (a coherent paragraph).
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
    prompt += "\n```"

    return prompt


@traceable(name="Gemini API Request", run_type="llm")
def _call_gemini_api(system_prompt: str, user_prompt: str) -> str:
    """
    Isolated wrapper around the Gemini API to explicitly log the exact 
    system instructions and user configurations into LangSmith.

    Includes comprehensive timing and error logging.
    """
    client = _get_gemini_client()

    logger.info(
        "Sending Gemini API request: model=%s, temperature=%s, max_output_tokens=%d, prompt_length=%d chars",
        GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS, len(user_prompt),
    )

    start_time = time.monotonic()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error("Gemini API call FAILED after %.2fs: %s: %s", elapsed, type(e).__name__, e)
        raise

    elapsed = time.monotonic() - start_time
    
    try:
        response_text = response.text.strip()
        logger.info(
            "Gemini API call SUCCEEDED in %.2fs — response_length=%d chars",
            elapsed, len(response_text),
        )
        return response_text
    except ValueError:
        logger.error(
            "Gemini returned empty/blocked response after %.2fs. Candidates: %s",
            elapsed, response.candidates,
        )
        raise RuntimeError("The AI response was blocked or returned an empty payload.")


@traceable(name="Generate Ticket Summary", run_type="chain")
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
        3. Validate the API configuration.
        4. Call Gemini LLM (wrapped in traceable span for input/output visibility).
        5. Parse response safely falling back to Regex.

    Args:
        customer_number: The customer identifier.
        product: The product category name.
        tickets_df: DataFrame containing the customer's tickets for the product.

    Returns:
        Optional[Dict]: A dictionary matching the required JSON format, or None on failure.

    Raises:
        ValueError: If Gemini API key is not configured.
    """
    overall_start = time.monotonic()
    logger.info("=== Starting summary generation: customer=%s, product=%s, tickets=%d ===",
                customer_number, product, len(tickets_df))

    try:
        # Step 1: Build robust context list
        step_start = time.monotonic()
        contexts = build_ticket_context(tickets_df)
        logger.info("Step 1 — Context building: %.2fs (%d contexts)", time.monotonic() - step_start, len(contexts))
        if not contexts:
            logger.warning("No ticket contexts built — returning None.")
            return None

        # Step 2: Build prompts
        step_start = time.monotonic()
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(customer_number, product, contexts)
        logger.info("Step 2 — Prompt building: %.2fs (system=%d chars, user=%d chars)",
                     time.monotonic() - step_start, len(system_prompt), len(user_prompt))

        # Step 3 & 4: Call LLM explicitly via traceable runner
        step_start = time.monotonic()
        response_text = _call_gemini_api(system_prompt, user_prompt)
        api_elapsed = time.monotonic() - step_start
        logger.info("Step 3 — Gemini API call: %.2fs", api_elapsed)
        
        # Step 5: Parse the JSON response
        step_start = time.monotonic()
        try:
            result = json.loads(response_text)
            logger.info("Step 4 — JSON parsing: success (direct parse)")
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            json_match = re.search(
                r"```(?:json)?\s*([\s\S]*?)\s*```", response_text
            )
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    logger.info("Step 4 — JSON parsing: success (extracted from markdown block)")
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON inside markdown block. Raw output:\n%s", response_text)
                    raise RuntimeError("The AI failed to generate a valid data structure.")
            else:
                logger.error("No valid JSON found in response. Raw output:\n%s", response_text)
                raise RuntimeError("The AI failed to generate a valid data structure.")
        logger.info("Step 4 — JSON parsing: %.2fs", time.monotonic() - step_start)

        total_elapsed = time.monotonic() - overall_start
        logger.info("=== Summary generation COMPLETE: customer=%s, product=%s, total=%.2fs ===",
                     customer_number, product, total_elapsed)
        return result

    except Exception as e:
        total_elapsed = time.monotonic() - overall_start
        logger.error("=== Summary generation FAILED after %.2fs: %s: %s ===",
                      total_elapsed, type(e).__name__, e)
        raise RuntimeError(f"LLM summary generation failed: {str(e)}")


@traceable(name="Generate Business Insights")
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
    client = _get_gemini_client()

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

    logger.info("Generating business insights: total_tickets=%s", stats.get('total_tickets', 'N/A'))
    start_time = time.monotonic()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=1024,
            ),
        )
        elapsed = time.monotonic() - start_time
        result = response.text.strip()
        logger.info("Business insights generated in %.2fs (%d chars).", elapsed, len(result))
        return result

    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error("Business insights FAILED after %.2fs: %s: %s", elapsed, type(e).__name__, e)
        return f"⚠️ Could not generate insights: {str(e)}"
