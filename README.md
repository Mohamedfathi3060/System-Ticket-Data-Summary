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

## 🚀 User Guide

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

### 4. How to Use the App
1. **Upload Data** → Use the sidebar to upload one or **multiple** `Ticket Data.txt` files (CSV formatted). The app safely merges multiple files and ignores invalid text files seamlessly.
2. **Explore Metrics** → View top-level metrics (total tickets, customers, avg resolution time).
3. **Data Grid** → Expand the raw data table below the charts to preview the loaded and cleaned tickets.
4. **Story Tab** → Select a customer from the dropdown. 
    - The Customer's tickets will be automatically grouped by product.
    - Click **"🤖 Generate AI Summaries"** to prompt Google Gemini to turn the linear chronological tickets into a compelling 5-phase storytelling narrative.
    - *Note: While the app is actively generating summaries, the button will be organically disabled to prevent accidental spam-clicks.*
5. **Insights Tab** → View interactive Plotly charts showing volume trends, resolutions, and typical recurring issues. Click **"Generate Insights"** to have the AI write a business evaluation based on these statistics.
6. **Export** → Download the merged, cleaned data as a `.csv` or `.xlsx` file using the native export buttons in the sidebar.

---

## 🏗️ Implementation Steps & Pipeline Documentation

This section details the internal journey data takes inside the application:

### Step 1: Data Preprocessing (`preprocessing.py`)
1. **File Ingestion**: Uploaded text files are decoded. We use `pd.read_csv(io.StringIO(content), engine='python', on_bad_lines='skip', low_memory=False)` to smoothly consume huge payloads and ignore broken lines without crashing Streamlit.
2. **Missing Variables Extraction**: `clean_data()` filters spaces, replaces custom 'N/A' strings with actual Pandas `pd.NA`, and guarantees column shapes (Customer numbers to strings).
3. **Date Generation**: Crucial temporal columns like `ACCEPTANCE_TIME` and `COMPLETION_TIME` are strictly cast to `datetime`. The `RESOLUTION_MINUTES` is derived and injected natively for metric boards.

### Step 2: Category Mapping (`category_mapping.py` / `config.py`)
1. **Business Rules**: `CATEGORY_TO_PRODUCT` drives mapping from backend system codes to user-friendly product categories (e.g., `HDW` or `KAI` -> `Broadband`, `KAD` -> `TV`).
2. **Association**: The function `apply_product_mapping()` reads the sanitized frame, executing `.map()` to dynamically build the `PRODUCT` column.
3. **Iconography**: `get_category_emoji()` layers custom frontend UI styling automatically based on the resolved `PRODUCT`.

### Step 3: Summary Generation (`llm_summary.py`)
1. **Data Thinning**: `build_ticket_context()` strips out complex routing/plumbing metadata, retaining only fields like dates, descriptions, and statuses (defined tightly in `NARRATIVE_FIELDS`).
2. **Prompt Compilation**: The system constructs a **User Prompt** consisting of a purely chronological flat list of tickets alongside a deeply rich One-Shot JSON example.
3. **LLM Routing Pipeline**: A rigid **System Prompt** firmly instructs Google Gemini to autonomously orchestrate assigning chronologically sorted tickets across exactly **5 narrative phases** (Initial Issue, Follow-ups, Developments, Later Incidents, Recent Events).
4. **Resolution**: The LLM parses the payload, translates any multilingual entries intrinsically to English, and outputs a strict parseable JSON string back to Streamlit for real-time visual UI rendering.

---

---

## 🔧 Module Documentation (Code Annotations)

The codebase is heavily annotated with Python docstrings detailing functionality, logic strategies, and parameter scopes. The architecture is broadly segmented below:

### `config.py` — Configuration Hub
Central source of truth for all business rules and constants:
- **VALID_CATEGORIES**: Categories kept during filtering.
- **CATEGORY_TO_PRODUCT**: Maps service categories to product names (HDW -> Broadband).
- **NARRATIVE_FIELDS**: Fields sent to the LLM (excludes routing metadata).
- **Gemini Settings & Token Safeties**: 
  - API limits: Model name, temperature, and token window logic.
  - `MAX_TICKETS_PER_SUMMARY`: Absolute limit (default 20) restricting maximum ticket ingestion to avoid API crashes or token-spend abuse by large dumps.
  - `MAX_FIELD_LENGTH`: Hard string length limit (default 500) per field trimming excessive user-descriptions automatically.
- **`load_style_config()`**: Merges `style_config.json` with built-in defaults.

### `preprocessing.py` — Data Pipeline
Handles the full journey from raw file to clean DataFrame, leveraging defensive programming to guarantee app stability:
- `parse_ticket_file()`: Reads CSV-formatted text files smoothly. 
- `clean_data()`: Replaces N/A, filters categories, wraps datetimes.
- `compute_resolution_time()`: Derives the performance deltas.

### `category_mapping.py` — Product Classification
Isolates classification logic:
- `apply_product_mapping()`: Binds business configurations dynamically via `.map()`.
- `get_category_emoji()`: Serves dynamic UI emojis per component.

### `llm_summary.py` — AI Storytelling Engine
Orchestrates the Gemini API for narrative generation, focusing extensively on prompt compilation:
- `build_ticket_context()`: Secures minimal ticket footprints protecting AI limits.
- `_build_system_prompt()`: Defines the role, tone, translation obligations, and strict format mapping mandates.
- `generate_summary()`: Combines contexts, handles direct AI network connectivity, and protects Streamlit with rigid parser fallbacks.

### `visualizations.py` — Interactive Charts
A suite of Plotly chart builder functions natively hooked onto the core data tables containing dynamic tooltips via the internal theme definitions (`style_config.json`).


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

### 1. Phase Grouping Logic
Unlike conventional chunking, Phase Grouping natively shifts entirely to the AI:
- The system simply passes a purely flat chronological array directly to Google Gemini. 
- The LLM orchestrates resolving tickets into 5 specific phases intelligently. 

### 2. Fields Sent to LLM
Only critical narrative columns (defined in `NARRATIVE_FIELDS`) are transported to Gemini. Excluded fields fall into two categories:
- **Routing metadata**: `ORDER_ID`, `ORDER_UNIT_ID`, `PLANNING_GROUP_KB`
- **Scheduling fields**: `ASSIGNMENT_TIME_MINIMUM`, `IMIL_TIME_MINIMUM`

This strictly controls token bandwidth and eliminates logic misinterpretations by Gemini API parameters.

### 3. Multilingual Handling
The data contains both English and German text. Rather than preprocessing translations, we instruct the LLM to handle multilingual input and produce English output. For charts, Plotly renders Unicode natively, so labels appear exactly as they are in the data.

### 4. One-Shot Prompting
The prompt includes a complete JSON example showing the expected output format. This significantly improves response format consistency vs. zero-shot, especially for the structured JSON output requirement.

---

## ⚙️ Assumptions

1. The input file is CSV-formatted (comma-delimited) with a header row matching the expected column names.
2. Date fields use `MM/DD/YYYY HH:MM` format (handled flexibly via `format="mixed"`).
3. `N/A` strings in the data represent missing values.
4. A single customer can have tickets across multiple product categories.
5. The Gemini API key has sufficient quota for the number of summary requests.
