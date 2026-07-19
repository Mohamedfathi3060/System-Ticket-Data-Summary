# 🎫 Ticket Data Storytelling — Streamlit App

AI-powered storytelling summaries from ticket data, built with Streamlit and Google Gemini.

## 📁 Project Structure

```
├── app.py                  # Streamlit entrypoint — UI, routing, layout
├── config.py               # Constants, mappings, style config loader
├── preprocessing.py        # File parsing, data cleaning, filtering, export
├── category_mapping.py     # SERVICE_CATEGORY → PRODUCT mapping logic
├── llm_summary.py          # Gemini API integration, prompt engineering
├── visualizations.py       # Plotly chart builders (5 chart types)
├── style_config.json       # External theme config (colors, fonts, palette)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── README.md               # This file
└── Ticket Data.txt         # Sample data (42 rows)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Gemini API key
# Get one at: https://aistudio.google.com/apikey
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Use the App

1. **Upload** → Use the sidebar to upload `Ticket Data.txt` (or any similarly formatted `.txt` file)
2. **Explore** → View top-level metrics (total tickets, customers, avg resolution time)
3. **Story Tab** → Select a customer from the dropdown → Click "Generate AI Summaries"
4. **Insights Tab** → View interactive charts → Click "Generate Insights" for AI analysis
5. **Export** → Download cleaned data as CSV or Excel from the sidebar

---

## 🔧 Module Documentation

### `config.py` — Configuration Hub
Central source of truth for all business rules and constants:
- **VALID_CATEGORIES**: Categories kept during filtering (`HDW, NET, KAI, KAV, GIGA, VOD, KAD`)
- **CATEGORY_TO_PRODUCT**: Maps service categories to product names
- **NARRATIVE_FIELDS**: Fields sent to the LLM (excludes routing metadata)
- **Gemini settings**: Model name, temperature, token limits
- **`load_style_config()`**: Merges `style_config.json` with built-in defaults

### `preprocessing.py` — Data Pipeline
Handles the full journey from raw file to clean DataFrame:
1. **`parse_ticket_file()`** → Reads CSV-formatted text files
2. **`clean_data()`** → Replaces N/A, filters categories, parses dates
3. **`compute_resolution_time()`** → Adds `RESOLUTION_MINUTES` column
4. **`convert_to_csv/excel()`** → Export utilities

### `category_mapping.py` — Product Classification
Maps `SERVICE_CATEGORY` codes to business-facing product names:
- KAI, NET → Broadband
- KAV → Voice
- KAD → TV
- GIGA → GIGA
- VOD → VOD
- HDW → Unmapped/Hardware (see Design Decisions)

### `llm_summary.py` — AI Storytelling Engine
Orchestrates the Gemini API for narrative generation:
1. **`build_ticket_context()`** → Extracts only narrative-relevant fields per ticket
2. **`split_into_phases()`** → Splits tickets into 5 chronological phases
3. **`build_prompt()`** → Constructs one-shot prompt with JSON example
4. **`generate_summary()`** → Calls Gemini, parses JSON response
5. **`generate_insights()`** → Generates business insights from aggregated stats

### `visualizations.py` — Interactive Charts
5 Plotly chart builders, all styled via `style_config.json`:
- Ticket volume over time (stacked by product)
- Average resolution time by product
- Most common issue types (description + cause)
- Ticket frequency per customer by product
- Ticket status distribution (donut chart)

---

## 🎨 Customization

Edit `style_config.json` to change the app's visual style without modifying code:

```json
{
    "page_title": "🎫 Ticket Data Storytelling",
    "primary_color": "#6C63FF",
    "font": "Inter",
    "chart_template": "plotly_dark",
    "chart_color_palette": ["#6C63FF", "#FF6584", "#43E97B", ...]
}
```

---

## 📝 Design Decisions

### 1. HDW Category Handling
HDW is included in the valid filter list but has **no formal product mapping** in the spec. We keep it and map it to `"Unmapped/Hardware"`, displaying a warning in the UI. This avoids silently dropping data.

### 2. Phase Splitting Strategy
- **≤5 tickets**: One ticket per phase (simple assignment)
- **>5 tickets**: Even quintile splitting (divide into 5 roughly equal groups)

The quintile approach was chosen over time-gap clustering because the dataset is small (42 rows, ~7-14 per customer). With so few tickets, gap-based clustering can produce highly uneven splits.

### 3. Fields Sent to LLM
Only 14 of 38 columns are sent to the LLM prompt (defined in `NARRATIVE_FIELDS`). Excluded fields fall into two categories:
- **Routing metadata**: `ORDER_ID`, `ORDER_UNIT_ID`, `PLANNING_GROUP_KB`, `SUBUNIT_NAME`
- **Scheduling fields**: `ASSIGNMENT_TIME_MINIMUM`, `IMIL_TIME_MINIMUM`, `ASSIGNED_BY_NAME`, etc.

These add noise without narrative value. The included fields provide the story arc: what happened, when, what was done, and the outcome.

### 4. Multilingual Handling
The data contains both English and German text. Rather than preprocessing translations, we instruct the LLM to handle multilingual input and produce English output. For charts, Plotly renders Unicode natively, so labels appear exactly as they are in the data.

### 5. One-Shot Prompting
The prompt includes a complete JSON example showing the expected output format. This significantly improves response format consistency vs. zero-shot, especially for the structured JSON output requirement.

---

## ⚙️ Assumptions

1. The input file is CSV-formatted (comma-delimited) with a header row matching the expected column names.
2. Date fields use `MM/DD/YYYY HH:MM` format (handled flexibly via `format="mixed"`).
3. `N/A` strings in the data represent missing values.
4. A single customer can have tickets across multiple product categories.
5. The Gemini API key has sufficient quota for the number of summary requests.
