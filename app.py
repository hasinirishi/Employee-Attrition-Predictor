import os
import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Enterprise Attrition Risk Workspace",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
IGNORED_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


@st.cache_resource
def load_prediction_assets():
    model = joblib.load("attrition_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features


@st.cache_data
def load_and_preprocess_data():
    df_raw = pd.read_csv(DATA_URL)
    df_display = df_raw.copy()

    df_model = df_raw.drop(columns=IGNORED_COLUMNS)
    df_model["Attrition"] = df_model["Attrition"].map({"Yes": 1, "No": 0})

    encoders = {}
    categorical_cols = df_model.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

    return df_raw, df_display, df_model, encoders


def risk_bucket(probability):
    if probability <= 0.30:
        return "Low Risk"
    if probability <= 0.70:
        return "Medium Risk"
    return "High Risk"


def risk_color(probability):
    if probability <= 0.30:
        return "#16a34a"
    if probability <= 0.70:
        return "#d97706"
    return "#dc2626"


# ---------------------------------------------------------------------------
# Theme helpers – each returns a CSS string for one concern
# ---------------------------------------------------------------------------

def _get_theme_colors(theme: str) -> dict:
    """Return a dictionary of colour tokens for the given theme."""
    if theme == "Dark":
        return {
            "bg": "#0B1220",
            "sidebar_bg": "#0B1220",
            "card": "#1E293B",
            "text": "#FFFFFF",
            "secondary": "#CBD5E1",
            "border": "#334155",
            "hover": "#334155",
            "input_bg": "#1E293B",
            "input_text": "#FFFFFF",
            "accent": "#3B82F6",
        }
    # Light
    return {
        "bg": "#FFFFFF",
        "sidebar_bg": "#F9FAFB",
        "card": "#FFFFFF",
        "text": "#111827",
        "secondary": "#64748B",
        "border": "#E5E7EB",
        "hover": "#F3F4F6",
        "input_bg": "#FFFFFF",
        "input_text": "#111827",
        "accent": "#2563EB",
    }


def _generate_css_variables(c: dict) -> str:
    """Root-level custom properties – the ONLY place colours are defined."""
    return f"""
:root {{
    --bg: {c["bg"]};
    --sidebar-bg: {c["sidebar_bg"]};
    --card: {c["card"]};
    --text: {c["text"]};
    --secondary: {c["secondary"]};
    --border: {c["border"]};
    --hover: {c["hover"]};
    --input-bg: {c["input_bg"]};
    --input-text: {c["input_text"]};
    --accent: {c["accent"]};
}}
"""


def _generate_app_layout_css() -> str:
    """Remove top gap, header, footer; set main background."""
    return """
/* ── Strip Streamlit header / footer / spacer ── */
/* Ensure header and sidebar toggle button are always visible, on top, and clickable */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 999999 !important;
}

header[data-testid="stHeader"] button {
    color: var(--text) !important;
    fill: var(--text) !important;
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    transition: all 0.15s ease !important;
    pointer-events: auto !important;
}

header[data-testid="stHeader"] button svg {
    color: inherit !important;
    fill: inherit !important;
}

header[data-testid="stHeader"] button:hover {
    background-color: var(--hover) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Hide footer only */
footer {
    visibility: hidden;
}

/* Reduce top padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* ── Background & base text ── */
.stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* ── Targeted text color inheritance ── */
.stMarkdown, .stText, .stCaption, .stWrite {
    color: var(--text);
}
.stMarkdown p, .stText p {
    color: var(--text);
}
"""


def _generate_sidebar_css() -> str:
    """Sidebar background, headings, labels, scrollbar, captions."""
    return """
/* ── Sidebar background ── */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
}

/* ── Sidebar padding ── */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* ── Remove top margin of the first element in sidebar ── */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child * {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ── Headings and widget text inside sidebar ── */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span,
section[data-testid="stSidebar"] div[data-testid="stCheckboxToggle"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckboxToggle"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckboxToggle"] span,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] p,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] label,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] label,
section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] label,
section[data-testid="stSidebar"] div[data-testid="stSlider"] label,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label {
    color: var(--text) !important;
}

/* ── Sidebar captions ── */
section[data-testid="stSidebar"] .stCaption p {
    color: var(--secondary) !important;
}

/* ── Sidebar subheaders and dividers ── */
section[data-testid="stSidebar"] .stMarkdown hr {
    border-color: var(--border) !important;
}

/* ── Sidebar info/warning/success alerts ── */
section[data-testid="stSidebar"] .stAlert {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] .stAlert p {
    color: var(--text) !important;
}

/* ── Sidebar scrollbar ── */
section[data-testid="stSidebar"] {
    scrollbar-width: thin;
    scrollbar-color: var(--hover) var(--border);
}
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 6px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: var(--border);
    border-radius: 20px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: var(--hover);
    border-radius: 20px;
    border: 2px solid var(--border);
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

/* ── Sidebar toggle/checkbox ── */
section[data-testid="stSidebar"] input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] .stButton button {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--accent) !important;
    background-color: var(--hover) !important;
    color: var(--accent) !important;
}
"""


def _generate_input_css() -> str:
    """All interactive widgets – select, text, number, multiselect, radio, slider."""
    return """
/* ── Dropdown Selectbox & Multiselect main input field container ── */
div[data-baseweb="select"] > div {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    min-height: 48px !important;
    display: flex !important;
    align-items: center !important;
    padding: 6px 12px !important;
    box-sizing: border-box !important;
    transition: border-color 0.15s ease !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: var(--accent) !important;
}

/* Ensure inner select component elements center vertically and use correct colors */
div[data-baseweb="select"] div[role="button"] {
    display: flex !important;
    align-items: center !important;
    color: var(--input-text) !important;
}

div[data-baseweb="select"] svg {
    color: var(--input-text) !important;
    fill: var(--input-text) !important;
    margin-top: 0 !important;
    align-self: center !important;
}

/* Search input field inside selectbox */
div[data-baseweb="select"] input {
    color: var(--input-text) !important;
    background-color: transparent !important;
    margin: 0 !important;
}

/* Multiselect tags */
div[data-baseweb="tag"] {
    background-color: var(--hover) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--input-text) !important;
    height: 28px !important;
    margin: 2px !important;
    display: inline-flex !important;
    align-items: center !important;
}
div[data-baseweb="tag"] span {
    color: var(--input-text) !important;
}

/* ── Text / number / password inputs ── */
input[type="text"],
input[type="number"],
input[type="password"],
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    height: 48px !important;
    box-sizing: border-box !important;
    line-height: normal !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

input[type="text"]:focus,
input[type="number"]:focus,
input[type="password"]:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--hover) !important;
}

input::placeholder {
    color: var(--secondary) !important;
    opacity: 0.8;
}

/* ── Radio buttons ── */
div[role="radiogroup"] {
    background-color: transparent !important;
    gap: 0.25rem;
}
div[role="radio"] {
    background-color: transparent !important;
    padding: 0.25rem 0;
}
input[type="radio"] {
    accent-color: var(--accent) !important;
}
div[role="radio"] p {
    color: var(--text) !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] {
    background-color: transparent !important;
}
div[data-testid="stSlider"] input[type="range"] {
    accent-color: var(--accent) !important;
}
div[data-testid="stSlider"] p {
    color: var(--text) !important;
}

/* ── Global widget labels (inputs, sliders, selectboxes, etc.) ── */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
    color: var(--text) !important;
}
"""


def _generate_file_uploader_css() -> str:
    """Modern dashed-border uploader with theme-aware text."""
    return """
/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background-color: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background-color: var(--hover) !important;
}
div[data-testid="stFileUploader"] button {
    background-color: var(--accent) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.4rem 1rem !important;
    transition: background 0.15s ease !important;
}
div[data-testid="stFileUploader"] button:hover {
    opacity: 0.9 !important;
}
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
    color: var(--secondary) !important;
}
div[data-testid="stFileUploader"] div[data-testid="stMarkdown"] p {
    color: var(--secondary) !important;
}
"""


def _generate_dropdown_css() -> str:
    """Popover / menu dropdowns with modern styling."""
    return """
/* ── Dropdown popover menu items ── */
div[data-baseweb="popover"],
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] span {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
}
div[data-baseweb="popover"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
    overflow: hidden !important;
}
div[data-baseweb="menu"] {
    padding: 0.25rem 0 !important;
}
div[data-baseweb="menu"] li,
div[data-baseweb="popover"] li {
    padding: 0.6rem 1rem !important;
    font-size: 0.875rem !important;
    transition: background-color 0.1s ease !important;
}
div[data-baseweb="menu"] li:hover,
div[data-baseweb="popover"] li:hover {
    background-color: var(--hover) !important;
    color: var(--accent) !important;
}
div[data-baseweb="menu"] li[aria-selected="true"],
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: var(--hover) !important;
    color: var(--accent) !important;
}
"""


def _generate_hero_card_css() -> str:
    """Hero header banner with modern SaaS styling."""
    return """
/* ── Hero header ── */
.main-header {
    background: var(--card);
    padding: 1.75rem 2rem;
    border-radius: 14px;
    margin: 0 0 1.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    border: 1px solid var(--border);
}
.main-header h1 {
    color: var(--text) !important;
    margin: 0 0 0.35rem 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
}
.main-header p {
    color: var(--secondary) !important;
    margin: 0 !important;
    font-size: 0.9rem;
    line-height: 1.5;
}
"""


def _generate_metric_card_css() -> str:
    """Main-content metric cards, risk badge, employee title."""
    return """
/* ── Metric cards (main area) ── */
.metric-card {
    background-color: var(--card);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: all 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.metric-card .label {
    font-size: 0.75rem;
    color: var(--secondary);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.25rem;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.metric-card .note {
    font-size: 0.75rem;
    color: var(--secondary);
    line-height: 1.3;
}

/* ── Sidebar metric cards (employee profile cards) ── */
.sidebar-card {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.55rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
.sidebar-card:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
.sidebar-card .card-label {
    color: var(--secondary) !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.4 !important;
}
.sidebar-card .card-value {
    color: var(--text) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 9999px;
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.4px;
    line-height: 1.5;
}

/* ── Employee title ── */
.employee-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: var(--text);
    letter-spacing: -0.01em;
}
"""


def _generate_main_area_cards_css() -> str:
    """Main area cards for profile display, etc."""
    return """
/* ── Employee detail card (in main area) ── */
.detail-card {
    background-color: var(--card);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border: 1px solid var(--border);
    margin-bottom: 0.75rem;
    transition: all 0.15s ease;
}
.detail-card:hover {
    border-color: var(--accent);
}
.detail-card .detail-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--secondary);
    margin-bottom: 0.15rem;
}
.detail-card .detail-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
}
"""


def _generate_generic_css() -> str:
    """Buttons, dataframes, tabs, expanders, headings, alerts, spacing, charts."""
    return """
/* ── Buttons ── */
.stButton > button {
    background-color: var(--accent);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.15s ease;
    padding: 0.5rem 1.25rem;
    line-height: 1.5;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.9;
}
.stButton > button:active {
    transform: scale(0.97);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0;
    margin-bottom: 1.25rem;
}
.stTabs [role="tablist"] button {
    color: var(--secondary);
    border-radius: 8px 8px 0 0;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    font-size: 0.875rem;
    transition: all 0.15s ease;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent;
}
.stTabs [role="tablist"] button:hover {
    color: var(--text);
    background-color: var(--hover);
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
    background: transparent;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    background-color: var(--card) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
[data-testid="stExpander"] summary {
    background-color: var(--card) !important;
    color: var(--text) !important;
    padding: 0.75rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: none !important;
    list-style: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: var(--hover) !important;
}
[data-testid="stExpander"] summary svg {
    fill: var(--text) !important;
}
[data-testid="stExpander"] > details > div {
    background-color: var(--card) !important;
    color: var(--text) !important;
    padding: 1.25rem 1.5rem !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: var(--text) !important;
}

/* ── Section headers ── */
h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    letter-spacing: -0.01em;
}

/* ── subheader styling ── */
.stSubheader {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* ── Alert boxes ── */
.stAlert {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    border: 1px solid var(--border);
    background-color: var(--card) !important;
    color: var(--text) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.15s ease !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer !important;
}
.stDownloadButton > button:hover {
    background-color: var(--hover) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Global scrollbar ── */
.main {
    scrollbar-width: thin;
    scrollbar-color: var(--hover) var(--border);
}
.main::-webkit-scrollbar { width: 6px; }
.main::-webkit-scrollbar-track {
    background: var(--border);
    border-radius: 20px;
}
.main::-webkit-scrollbar-thumb {
    background: var(--hover);
    border-radius: 20px;
    border: 2px solid var(--border);
}
.main::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

/* ── Matplotlib & general charts/images centering ── */
.element-container:has(div[data-testid="stImage"]),
.element-container:has(div[data-testid="stPyplot"]),
.element-container:has(div[data-testid="stPlotlyChart"]),
.element-container:has(div[data-testid="stVegaLiteChart"]),
div[data-testid="stImage"],
div[data-testid="stPyplot"],
div[data-testid="stPlotlyChart"],
div[data-testid="stVegaLiteChart"],
.stPlotlyContainer,
.stImage,
.stPyplot {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
    width: 100% !important;
}

div[data-testid="stImage"] img,
div[data-testid="stImage"] svg,
div[data-testid="stImage"] > div,
div[data-testid="stPyplot"] img,
div[data-testid="stPyplot"] svg,
div[data-testid="stPyplot"] > div,
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stVegaLiteChart"] > div,
.stImage img,
.stImage svg,
.stPyplot img,
.stPyplot svg {
    margin-left: auto !important;
    margin-right: auto !important;
    display: inline-block !important;
    text-align: center !important;
}

div[data-testid="stImage"] img,
div[data-testid="stImage"] svg,
div[data-testid="stPyplot"] img,
div[data-testid="stPyplot"] svg {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 8px;
}

/* ── Dataframes & Tables styling (contrast fix) ── */
div[data-testid="stDataFrame"],
div[data-testid="stTable"],
.stDataFrame,
.stTable,
table {
    color: var(--text) !important;
    background-color: var(--card) !important;
    --style-th-color: var(--text) !important;
    --style-th-bg: var(--hover) !important;
    --style-td-color: var(--text) !important;
    --style-td-bg: var(--card) !important;
    --style-border-color: var(--border) !important;
}

/* Style header elements */
div[data-testid="stDataFrame"] th,
div[data-testid="stTable"] th,
table th,
thead tr th,
.stDataFrame th,
.stTable th {
    background-color: var(--hover) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
}

/* Force child elements inside table headers to inherit the correct text color */
div[data-testid="stDataFrame"] th *,
div[data-testid="stTable"] th *,
table th * {
    color: var(--text) !important;
}

/* Style data cell elements */
div[data-testid="stDataFrame"] td,
div[data-testid="stTable"] td,
table td,
tbody tr td,
.stDataFrame td,
.stTable td {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
}
"""


def inject_dashboard_styles(theme: str) -> None:
    """Inject all CSS for the dashboard using CSS custom properties.

    Theme switching works by re-declaring ``:root { --var: value }`` based on
    the active *theme* ("Dark" or "Light").  Every other rule references
    ``var(--x)``, so a single variable update propagates everywhere.
    """
    colors = _get_theme_colors(theme)

    css_parts = [
        _generate_css_variables(colors),
        _generate_app_layout_css(),
        _generate_sidebar_css(),
        _generate_input_css(),
        _generate_file_uploader_css(),
        _generate_dropdown_css(),
        _generate_hero_card_css(),
        _generate_metric_card_css(),
        _generate_main_area_cards_css(),
        _generate_generic_css(),
    ]

    st.markdown(
        f"<style>{''.join(css_parts)}</style>",
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_card(label, value):
    """Employee profile card with theme-aware colours (sidebar use)."""
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(category, probability):
    st.markdown(
        f"""
        <span class="risk-badge" style="background:{risk_color(probability)}">
            {category}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_top_factor_chart(top_factors, theme="Dark"):
    chart_data = top_factors.copy().sort_values("SHAP_Value")
    colors = ["#dc2626" if value > 0 else "#16a34a" for value in chart_data["SHAP_Value"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.barh(chart_data["Feature"], chart_data["SHAP_Value"], color=colors, height=0.6, edgecolor="none")
    
    text_color = "#FFFFFF" if theme == "Dark" else "#111827"
    sec_color = "#CBD5E1" if theme == "Dark" else "#64748B"
    grid_color = "#334155" if theme == "Dark" else "#E5E7EB"
    
    ax.axvline(0, color=sec_color, linewidth=0.8)
    ax.set_xlabel("SHAP impact on leave risk", fontsize=8, color=sec_color)
    ax.set_ylabel("")
    ax.set_title("Personalized Risk Drivers", fontsize=10, fontweight=600, color=text_color, pad=10)
    ax.grid(axis="x", alpha=0.15, color=grid_color)
    ax.tick_params(colors=sec_color, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_risk_comparison_chart(employee_probability, dataset_average, theme="Dark"):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    
    profiles = ["Selected Employee", "Dataset Average"]
    risks = [employee_probability, dataset_average]
    colors = ["#2563EB" if theme == "Light" else "#3B82F6", "#64748B" if theme == "Light" else "#475569"]
    
    ax.bar(profiles, risks, color=colors, width=0.4, edgecolor="none")
    
    text_color = "#FFFFFF" if theme == "Dark" else "#111827"
    sec_color = "#CBD5E1" if theme == "Dark" else "#64748B"
    grid_color = "#334155" if theme == "Dark" else "#E5E7EB"
    
    ax.set_ylabel("Leave Risk", fontsize=8, color=sec_color)
    ax.set_title("Risk Comparison", fontsize=10, fontweight=600, color=text_color, pad=10)
    ax.grid(axis="y", alpha=0.15, color=grid_color)
    ax.tick_params(colors=sec_color, labelsize=7.5)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_profile_radar(selected_raw, theme="Dark"):
    radar_fields = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "JobInvolvement",
        "WorkLifeBalance",
        "PerformanceRating",
    ]
    available_fields = [field for field in radar_fields if field in selected_raw.index]
    if len(available_fields) < 3:
        st.info("Not enough satisfaction/performance fields to draw a profile radar.")
        return

    values = []
    for field in available_fields:
        value = pd.to_numeric(selected_raw[field], errors="coerce")
        values.append(0 if pd.isna(value) else float(value))

    max_values = [4 if field != "PerformanceRating" else 4 for field in available_fields]
    normalized = [value / max_value for value, max_value in zip(values, max_values)]
    angles = np.linspace(0, 2 * np.pi, len(available_fields), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    text_color = "#FFFFFF" if theme == "Dark" else "#111827"
    sec_color = "#CBD5E1" if theme == "Dark" else "#64748B"
    grid_color = "#334155" if theme == "Dark" else "#E5E7EB"

    ax.spines['polar'].set_color(sec_color)
    ax.spines['polar'].set_linewidth(0.8)

    ax.plot(angles, normalized, color="#3B82F6" if theme == "Dark" else "#2563EB", linewidth=2)
    ax.fill(angles, normalized, color="#3B82F6" if theme == "Dark" else "#2563EB", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_fields, fontsize=7.5, color=sec_color)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["1", "2", "3", "4"], fontsize=6.5, color=sec_color)
    ax.set_ylim(0, 1)
    ax.set_title("Personal Profile Shape", pad=14, fontsize=10, fontweight=600, color=text_color)
    ax.grid(alpha=0.2, color=grid_color)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def preprocess_uploaded_dataset(uploaded_df, feature_names, encoders, training_model_df, user_overrides=None):
    from preprocessor import UniversalPreprocessor
    preprocessor = UniversalPreprocessor(target_features=feature_names)
    
    # Check if matches IBM format exactly
    if preprocessor.detect_schema(uploaded_df):
        df_display = uploaded_df.copy()
        df_model = uploaded_df.copy()

        for col in IGNORED_COLUMNS:
            if col in df_model.columns:
                df_model = df_model.drop(columns=[col])

        actual_attrition = None
        if "Attrition" in df_model.columns:
            actual_attrition = df_model["Attrition"].copy()
            df_model = df_model.drop(columns=["Attrition"])

        missing_features = [col for col in feature_names if col not in df_model.columns]
        if missing_features:
            raise ValueError(
                "Uploaded dataset is missing required model columns: "
                + ", ".join(missing_features)
            )

        df_model = df_model[feature_names].copy()

        for col in feature_names:
            if col in encoders:
                if pd.api.types.is_numeric_dtype(df_model[col]):
                    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
                else:
                    values = df_model[col].astype(str)
                    allowed = set(encoders[col].classes_)
                    unknown = sorted(set(values.dropna().unique()) - allowed)

                    numeric_values = pd.to_numeric(values, errors="coerce")
                    numeric_encoded = (
                        numeric_values.notna().all()
                        and numeric_values.between(0, len(encoders[col].classes_) - 1).all()
                    )

                    if numeric_encoded:
                        df_model[col] = numeric_values.astype(int)
                    elif unknown:
                        raise ValueError(
                            f"Column '{col}' contains unknown categories: {', '.join(unknown[:8])}"
                        )
                    else:
                        df_model[col] = encoders[col].transform(values)
            else:
                df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

        missing_values = df_model.columns[df_model.isna().any()].tolist()
        if missing_values:
            medians = training_model_df[feature_names].median(numeric_only=True)
            df_model = df_model.fillna(medians)

        st.session_state["preprocessor_report"] = {
            "mapped": {c: c for c in feature_names if c in df_display.columns},
            "derived": {},
            "imputed": {col: "training median" for col in missing_values},
            "ignored": [col for col in IGNORED_COLUMNS if col in df_display.columns],
            "unknown": []
        }
        st.session_state["is_ibm_format"] = True
        st.session_state["profile_summary"] = preprocessor.get_profile_summary(uploaded_df)
        return df_display, df_model, actual_attrition, missing_values

    else:
        st.session_state["is_ibm_format"] = False
        
        # Build mapping using overrides if present
        preprocessor.map_columns(uploaded_df, user_overrides=user_overrides)
        
        # Derive features
        derived_df = preprocessor.derive_features(uploaded_df)
        
        # Rename raw names to standard names
        mapped_df = uploaded_df.rename(columns=preprocessor.mappings)
        
        # Keep only standard features
        keep_cols = [c for c in mapped_df.columns if c in feature_names]
        mapped_df = mapped_df[keep_cols]
        
        # Merge derived columns
        for col in derived_df.columns:
            if col in feature_names:
                mapped_df[col] = derived_df[col]
                
        # Impute missing values
        medians = training_model_df[feature_names].median(numeric_only=True).to_dict()
        imputed_df = preprocessor.handle_missing(mapped_df, training_medians=medians)
        
        # Encode categorical variables using target encoders
        encoded_df = preprocessor.encode_categories(imputed_df, encoders)
        
        # Align features to model exactly
        aligned_df = preprocessor.align_to_model(encoded_df, feature_names)
        
        # Extract actual attrition
        actual_attrition = None
        attrition_cols = [raw for raw, target in preprocessor.mappings.items() if target == "Attrition"]
        if not attrition_cols:
            for col in uploaded_df.columns:
                if col.lower() == "attrition":
                    actual_attrition = uploaded_df[col].copy()
                    break
        else:
            actual_attrition = uploaded_df[attrition_cols[0]].copy()
            
        st.session_state["preprocessor_report"] = preprocessor.generate_report()
        st.session_state["profile_summary"] = preprocessor.get_profile_summary(uploaded_df)
        
        # Prep df_display for the Streamlit UI cards & filter components
        df_display_mapped = uploaded_df.copy()
        
        emp_num_mapped = None
        for raw, target in preprocessor.mappings.items():
            if target == "EmployeeNumber":
                emp_num_mapped = raw
                break
        
        if emp_num_mapped and emp_num_mapped in df_display_mapped.columns:
            df_display_mapped["EmployeeNumber"] = df_display_mapped[emp_num_mapped]
        else:
            id_cols = [c for c in df_display_mapped.columns if c.lower() in ["id", "empid", "employeename", "employeenumber", "employeenum", "number"]]
            if id_cols:
                df_display_mapped["EmployeeNumber"] = df_display_mapped[id_cols[0]]
            else:
                df_display_mapped["EmployeeNumber"] = [f"EMP-{i+1001}" for i in range(len(df_display_mapped))]

        display_mappings = {}
        for raw, target in preprocessor.mappings.items():
            if target in ["Age", "Gender", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany", "OverTime"]:
                display_mappings[raw] = target
        df_display_mapped = df_display_mapped.rename(columns=display_mappings)
        
        from preprocessor import INTELLIGENT_DEFAULTS
        for key, def_val in INTELLIGENT_DEFAULTS.items():
            if key not in df_display_mapped.columns:
                if key in derived_df.columns:
                    df_display_mapped[key] = derived_df[key]
                else:
                    df_display_mapped[key] = def_val
                    
        df_display_mapped = df_display_mapped.fillna("-")
        
        filled_list = list(preprocessor.imputations.keys())
        return df_display_mapped, aligned_df, actual_attrition, filled_list


def build_prediction_frame(active_features, feature_names):
    input_df = pd.DataFrame([active_features])
    return input_df[feature_names]


def get_shap_values_for_class_one(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[0, :, 1]
    else:
        shap_values_class1 = shap_values[0]

    expected_value_class1 = (
        explainer.expected_value[1]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )
    return shap_values_class1, expected_value_class1


# ---------------------------------------------------------------------------
# App body – no backend logic changed
# ---------------------------------------------------------------------------

# Load assets
try:
    model, feature_names = load_prediction_assets()
    df_raw, df_display, df_model, encoders = load_and_preprocess_data()
except Exception as e:
    st.error(
        "Could not initialize assets: "
        f"{str(e)}. Ensure 'attrition_model.pkl' and 'model_features.pkl' sit in this folder."
    )
    st.stop()


# Initialize theme and styling
st.sidebar.header("🎨 Workspace Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

theme = "Dark" if dark_mode else "Light"
inject_dashboard_styles(theme)

# Hero Header - Improved layout
st.markdown(
    """
    <div class="main-header">
        <h1>🔮 Enterprise Attrition Risk Workspace</h1>
        <p>Upload employee data, predict attrition, explore SHAP explanations, and generate retention insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.header("📋 Choose Input Method")

input_mode = st.sidebar.radio(
    "Select analysis mode:",
    [
        "Search Existing Employee",
        "Interactive Simulator (Manual)",
        "Upload Dataset",
    ],
)

active_features = {}
is_historical = False
historical_status = ""


if input_mode == "Search Existing Employee":
    st.sidebar.subheader("Search Parameters")

    historical_model_features_df = df_model.drop(columns=["Attrition"])
    historical_probabilities = model.predict_proba(historical_model_features_df[feature_names])[:, 1]
    historical_results_df = df_display.copy()
    historical_results_df["PredictedAttritionRisk"] = historical_probabilities
    historical_results_df["PredictedOutcome"] = np.where(
        historical_probabilities > 0.50,
        "Likely to Leave",
        "Likely to Stay",
    )
    historical_results_df["RiskCategory"] = [
        risk_bucket(probability) for probability in historical_probabilities
    ]

    if "historical_risk_filter" not in st.session_state:
        st.session_state.historical_risk_filter = "All"

    if st.sidebar.button("Show All", use_container_width=True):
        st.session_state.historical_risk_filter = "All"
    if st.sidebar.button("High Risk Only", use_container_width=True):
        st.session_state.historical_risk_filter = "High Risk"

    historical_search_text = st.sidebar.text_input(
        "Search employee, department, role, or profile text",
        value="",
    )
    historical_department = st.sidebar.selectbox(
        "Department",
        ["All"] + sorted(df_display["Department"].dropna().astype(str).unique()),
    )
    historical_role = st.sidebar.selectbox(
        "Job Role",
        ["All"] + sorted(df_display["JobRole"].dropna().astype(str).unique()),
    )
    historical_risk = st.sidebar.selectbox(
        "Risk Category",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        index=["All", "High Risk", "Medium Risk", "Low Risk"].index(
            st.session_state.historical_risk_filter
        ),
    )
    st.session_state.historical_risk_filter = historical_risk

    filtered_historical_df = historical_results_df.copy()
    if historical_risk != "All":
        filtered_historical_df = filtered_historical_df[
            filtered_historical_df["RiskCategory"] == historical_risk
        ]
    if historical_department != "All":
        filtered_historical_df = filtered_historical_df[
            filtered_historical_df["Department"].astype(str) == historical_department
        ]
    if historical_role != "All":
        filtered_historical_df = filtered_historical_df[
            filtered_historical_df["JobRole"].astype(str) == historical_role
        ]
    if historical_search_text.strip():
        searchable = filtered_historical_df.astype(str).agg(" ".join, axis=1).str.lower()
        filtered_historical_df = filtered_historical_df[
            searchable.str.contains(historical_search_text.strip().lower(), regex=False)
        ]

    if filtered_historical_df.empty:
        st.sidebar.warning("No matching employees. Showing all records.")
        filtered_historical_df = historical_results_df.copy()

    employee_options = [
        (
            f"{idx} | Employee {row['EmployeeNumber']} | {row['Department']} | "
            f"{row['RiskCategory']} | {row['PredictedAttritionRisk']:.2%}"
        )
        for idx, row in filtered_historical_df.iterrows()
    ]
    selected_employee = st.sidebar.selectbox(
        "Select Employee:",
        employee_options,
        index=0,
    )
    row_idx = int(selected_employee.split(" | ", 1)[0])
    selected_id = df_raw.iloc[row_idx]["EmployeeNumber"]

    is_historical = True
    raw_attrition = df_raw.iloc[row_idx]["Attrition"]
    historical_status = "Left Company" if raw_attrition == "Yes" else "Currently Active (Stayed)"

    model_row = df_model.drop(columns=["Attrition"]).iloc[[row_idx]]
    active_features = model_row.to_dict(orient="records")[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 Employee ID {selected_id}")
    display_row = df_display.iloc[row_idx]

    # Profile info as sidebar cards with proper theme-aware colours inside sidebar context
    with st.sidebar:
        sidebar_card("Age / Gender", f"{display_row['Age']} / {display_row['Gender']}")
        sidebar_card("Department", f"{display_row['Department']}")
        sidebar_card("Role", f"{display_row['JobRole']}")
        sidebar_card("Monthly Income", f"${display_row['MonthlyIncome']:,}")
        sidebar_card("Overtime Status", f"{display_row['OverTime']}")
        sidebar_card("Years at Company", f"{display_row['YearsAtCompany']} years")

elif input_mode == "Interactive Simulator (Manual)":
    st.sidebar.subheader("Simulator Settings")
    st.sidebar.info(
        "Adjust parameters in the main workspace expanders to create a hypothetical profile."
    )

else:
    st.sidebar.subheader("Dataset Upload")
    uploaded_file = st.sidebar.file_uploader("Upload employee CSV", type=["csv"])
    st.sidebar.caption(
        "Upload a CSV in the original IBM employee attrition format. The app drops "
        "EmployeeCount, EmployeeNumber, Over18, and StandardHours before prediction."
    )


tab1, tab2 = st.tabs(["Diagnostic Risk Hub", "Global Model Insights"])


with tab1:
    if input_mode == "Upload Dataset":
        st.subheader("📤 Upload Dataset Attrition Insights", anchor=False)

        if uploaded_file is None:
            st.info("📁 Upload a CSV from the left sidebar to score employees in bulk.")
            st.markdown("#### 🔄 Upload Processing")
            st.write(
                "The uploaded dataset is automatically prepared with the same preprocessing "
                "used during training: unused IBM columns are removed, categorical fields are "
                "encoded, columns are reordered to match `model_features.pkl`, and the model "
                "returns attrition probabilities."
            )

            sample_template = df_raw.head(25).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download IBM-format sample CSV",
                data=sample_template,
                file_name="ibm_attrition_upload_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("#### ✅ Required Prediction Columns")
            st.dataframe(pd.DataFrame({"Required Column": feature_names}), use_container_width=True)
        else:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                from preprocessor import UniversalPreprocessor
                temp_preprocessor = UniversalPreprocessor(target_features=feature_names)
                is_ibm = temp_preprocessor.detect_schema(uploaded_df)
                
                # State management for mapping overrides
                if "mapping_overrides" not in st.session_state:
                    st.session_state["mapping_overrides"] = {}
                if "run_prediction_clicked" not in st.session_state:
                    st.session_state["run_prediction_clicked"] = False
                
                # Reset state if a new file is uploaded
                if "last_uploaded_file_name" not in st.session_state or st.session_state["last_uploaded_file_name"] != uploaded_file.name:
                    st.session_state["last_uploaded_file_name"] = uploaded_file.name
                    st.session_state["mapping_overrides"] = {}
                    st.session_state["run_prediction_clicked"] = False
                
                # If custom schema and not yet clicked run, show mapping and profiling UI
                if not is_ibm and not st.session_state.get("run_prediction_clicked", False):
                    st.markdown("### 📊 Universal Schema Mapping & Profiling")
                    st.info("⚠️ Custom employee dataset layout detected. Please verify column alignments below.")
                    
                    # Generate initial mapping to get suggestions
                    temp_preprocessor.map_columns(uploaded_df, user_overrides=st.session_state["mapping_overrides"])
                    profile = temp_preprocessor.get_profile_summary(uploaded_df)
                    
                    # Layout: 2 Columns (Profiling vs Mapping)
                    p_col1, p_col2 = st.columns([1, 1.2], gap="large")
                    
                    with p_col1:
                        st.markdown("#### 📈 Data Profiling Summary")
                        score = profile["score"]
                        score_color = "#16a34a" if score >= 80 else "#d97706" if score >= 50 else "#dc2626"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: var(--card); border: 1px solid var(--border); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                                <div style="font-size: 14px; color: var(--secondary); margin-bottom: 5px;">DATA QUALITY SCORE</div>
                                <div style="font-size: 36px; font-weight: 700; color: {score_color}; margin-bottom: 10px;">{score} / 100</div>
                                <div style="background-color: var(--border); height: 8px; border-radius: 4px; overflow: hidden;">
                                    <div style="background-color: {score_color}; width: {score}%; height: 100%;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        prof_cols = st.columns(2)
                        with prof_cols[0]:
                            metric_card("Total Rows", f"{profile['rows']:,}", "Employees in dataset")
                        with prof_cols[1]:
                            metric_card("Total Columns", f"{profile['columns']:,}", "Uploaded fields")
                            
                        prof_cols2 = st.columns(2)
                        with prof_cols2[0]:
                            metric_card("Missing Values", f"{profile['missing_pct']:.2f}%", "Cell percentage")
                        with prof_cols2[1]:
                            mapped_vs_total = f"{profile['mapped_columns_count']} / {len(feature_names)}"
                            metric_card("Mapped Features", mapped_vs_total, "Model alignment")
                            
                        st.markdown("**Quality suggestions:**")
                        for sug in profile["suggestions"]:
                            st.markdown(f"- {sug}")
                            
                    with p_col2:
                        st.markdown("#### ⚙️ Custom Schema Column Alignment")
                        st.write("Confirm or adjust how your columns map to the required model features.")
                        
                        # Build options list
                        standard_options = ["Ignore", "Attrition", "EmployeeNumber"] + sorted(feature_names)
                        
                        overrides = {}
                        with st.expander("Configure Schema Mappings", expanded=True):
                            for col in uploaded_df.columns:
                                # Get auto-mapped value or override
                                current_mapping = st.session_state["mapping_overrides"].get(col)
                                if current_mapping is None:
                                    if col in temp_preprocessor.mappings:
                                        current_mapping = temp_preprocessor.mappings[col]
                                    elif col in temp_preprocessor.ignored_columns:
                                        current_mapping = "Ignore"
                                    else:
                                        current_mapping = "Ignore"
                                
                                # Heuristic suggestion for label caption (Phase 6)
                                sugg_target, sugg_desc = temp_preprocessor.suggest_llm_mapping(col)
                                if sugg_target:
                                    label_suffix = f" (💡 Suggestion: {sugg_target} - {sugg_desc})"
                                else:
                                    label_suffix = ""
                                
                                select_idx = standard_options.index(current_mapping) if current_mapping in standard_options else 0
                                selected_val = st.selectbox(
                                    f"Map column: **`{col}`**{label_suffix}",
                                    options=standard_options,
                                    index=select_idx,
                                    key=f"map_sel_{col}"
                                )
                                overrides[col] = selected_val
                        
                        if st.button("Apply Mappings & Run Predictions", type="primary", use_container_width=True):
                            st.session_state["mapping_overrides"] = overrides
                            st.session_state["run_prediction_clicked"] = True
                            st.rerun()
                            
                    st.stop()
                
                # Exec preprocessing and model inference
                uploaded_display, uploaded_model_df, actual_attrition, filled_cols = (
                    preprocess_uploaded_dataset(
                        uploaded_df, 
                        feature_names, 
                        encoders, 
                        df_model, 
                        user_overrides=st.session_state.get("mapping_overrides", {})
                    )
                )
                probabilities = model.predict_proba(uploaded_model_df)[:, 1]
                predictions = (probabilities > 0.50).astype(int)

                results_df = uploaded_display.copy()
                results_df["PredictedAttritionRisk"] = probabilities
                results_df["PredictedOutcome"] = np.where(predictions == 1, "Likely to Leave", "Likely to Stay")
                results_df["RiskCategory"] = [risk_bucket(p) for p in probabilities]

                total_employees = len(results_df)
                avg_risk = float(np.mean(probabilities))
                high_risk_count = int((probabilities > 0.70).sum())

                if filled_cols:
                    st.warning(
                        "⚠️ Some blank numeric values were filled with training medians or defaults: "
                        + ", ".join(filled_cols)
                    )

                st.markdown("#### 📊 Upload Summary")
                summary_cols = st.columns(4, gap="medium")
                with summary_cols[0]:
                    metric_card("Employees Scored", f"{total_employees:,}", "Rows processed")
                with summary_cols[1]:
                    metric_card("Average Leave Risk", f"{avg_risk:.2%}", "Across uploaded file")
                with summary_cols[2]:
                    metric_card("High Risk Employees", f"{high_risk_count:,}", "Above 70% risk")
                with summary_cols[3]:
                    metric_card("Projected Attrition", f"{int(predictions.sum()):,}", "Above 50% probability")
                    
                # Display universal preprocessor transformation report (Phase 9)
                st.markdown("#### ⚙️ Universal Preprocessing Explanation")
                with st.expander("View pipeline transformations & feature engineering report", expanded=True):
                    report = st.session_state.get("preprocessor_report", {})
                    
                    # Mapped Columns
                    if report.get("mapped"):
                        st.markdown("**Mapped columns:**")
                        mapped_text = ", ".join([f"`{raw}` → `{tgt}`" for raw, tgt in report["mapped"].items()])
                        st.write(mapped_text)
                        
                    # Derived Columns
                    if report.get("derived"):
                        st.markdown("**Derived features:**")
                        for tgt, info in report["derived"].items():
                            st.write(f"- `{tgt}` derived from `{info['source']}` using: `{info['method']}`")
                            
                    # Imputed Columns
                    if report.get("imputed"):
                        st.markdown("**Missing features handled:**")
                        for tgt, method in report["imputed"].items():
                            st.info(f"⚠️ `{tgt}` missing in upload. Imputed using: **{method}**.")
                            
                    if not is_ibm:
                        if st.button("Modify Schema Mappings", use_container_width=True):
                            st.session_state["run_prediction_clicked"] = False
                            st.rerun()

                if actual_attrition is not None:
                    actual_numeric = actual_attrition.map({"Yes": 1, "No": 0})
                    if actual_numeric.isna().all():
                        actual_numeric = pd.to_numeric(actual_attrition, errors="coerce")

                    if actual_numeric.notna().any():
                        comparable = actual_numeric.notna()
                        accuracy = (predictions[comparable] == actual_numeric[comparable]).mean()
                        actual_rate = actual_numeric[comparable].mean()

                        st.markdown("#### 📈 Model Validation")
                        hist_cols = st.columns(2, gap="medium")
                        with hist_cols[0]:
                            metric_card("Actual Attrition Rate", f"{actual_rate:.2%}", "From uploaded labels")
                        with hist_cols[1]:
                            metric_card("Model Match Rate", f"{accuracy:.2%}", "Compared with actual labels")

                st.markdown("---")
                st.subheader("🔍 Find and Filter Employees", anchor=False)

                if "upload_risk_filter" not in st.session_state:
                    st.session_state.upload_risk_filter = "All"

                filter_buttons = st.columns(5, gap="small")
                if filter_buttons[0].button("All Employees", use_container_width=True):
                    st.session_state.upload_risk_filter = "All"
                if filter_buttons[1].button("High Risk", use_container_width=True):
                    st.session_state.upload_risk_filter = "High Risk"
                if filter_buttons[2].button("Medium Risk", use_container_width=True):
                    st.session_state.upload_risk_filter = "Medium Risk"
                if filter_buttons[3].button("Low Risk", use_container_width=True):
                    st.session_state.upload_risk_filter = "Low Risk"
                if filter_buttons[4].button("Reset Filters", use_container_width=True):
                    st.session_state.upload_risk_filter = "All"

                filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1, 1])
                with filter_col1:
                    search_text = st.text_input(
                        "Search EmployeeNumber, department, role, or any profile text",
                        value="",
                    )
                with filter_col2:
                    dept_options = ["All"]
                    if "Department" in results_df.columns:
                        dept_options += sorted(results_df["Department"].dropna().astype(str).unique())
                    selected_department = st.selectbox("Department", dept_options)
                with filter_col3:
                    role_options = ["All"]
                    if "JobRole" in results_df.columns:
                        role_options += sorted(results_df["JobRole"].dropna().astype(str).unique())
                    selected_role = st.selectbox("Job Role", role_options)

                filtered_df = results_df.copy()
                if st.session_state.upload_risk_filter != "All":
                    filtered_df = filtered_df[
                        filtered_df["RiskCategory"] == st.session_state.upload_risk_filter
                    ]
                if selected_department != "All":
                    filtered_df = filtered_df[
                        filtered_df["Department"].astype(str) == selected_department
                    ]
                if selected_role != "All":
                    filtered_df = filtered_df[
                        filtered_df["JobRole"].astype(str) == selected_role
                    ]
                if search_text.strip():
                    searchable = filtered_df.astype(str).agg(" ".join, axis=1).str.lower()
                    filtered_df = filtered_df[
                        searchable.str.contains(search_text.strip().lower(), regex=False)
                    ]

                st.caption(
                    f"Showing {len(filtered_df):,} of {len(results_df):,} uploaded employees."
                )

                preferred_columns = [
                    "EmployeeNumber",
                    "Age",
                    "Department",
                    "JobRole",
                    "MonthlyIncome",
                    "OverTime",
                    "YearsAtCompany",
                    "PredictedAttritionRisk",
                    "RiskCategory",
                    "PredictedOutcome",
                ]
                visible_columns = [col for col in preferred_columns if col in results_df.columns]

                if filtered_df.empty:
                    st.warning("No employees match the current filters.")
                    filtered_df = results_df.copy()

                table_col, employee_col = st.columns([1.05, 1.35])

                with table_col:
                    st.markdown("#### Filtered Employees")
                    preview_df = filtered_df.sort_values(
                        "PredictedAttritionRisk", ascending=False
                    )
                    st.dataframe(
                        preview_df[visible_columns].style.format(
                            {"PredictedAttritionRisk": "{:.2%}"}
                        ),
                        use_container_width=True,
                        height=430,
                    )

                    st.download_button(
                        "Download scored dataset",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="attrition_scored_dataset.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with employee_col:
                    st.markdown("#### Employee Risk Profile")
                    if "EmployeeNumber" in results_df.columns:
                        employee_options = [
                            f"{idx} | Employee {row['EmployeeNumber']} | {row['RiskCategory']} | {row['PredictedAttritionRisk']:.2%}"
                            for idx, row in filtered_df.iterrows()
                        ]
                    else:
                        employee_options = [
                            f"{idx} | Uploaded row {idx + 1} | {row['RiskCategory']} | {row['PredictedAttritionRisk']:.2%}"
                            for idx, row in filtered_df.iterrows()
                        ]

                    selected_employee = st.selectbox(
                        "Choose a filtered employee:",
                        employee_options,
                        index=0,
                    )
                    selected_upload_idx = int(selected_employee.split(" | ", 1)[0])

                    selected_raw = results_df.loc[selected_upload_idx]
                    selected_input_df = uploaded_model_df.loc[[selected_upload_idx], feature_names]
                    selected_probability_leave = float(
                        model.predict_proba(selected_input_df)[0][1]
                    )
                    selected_probability_stay = 1 - selected_probability_leave
                    selected_category = risk_bucket(selected_probability_leave)

                    employee_label = (
                        f"Employee {selected_raw['EmployeeNumber']}"
                        if "EmployeeNumber" in results_df.columns
                        else f"Uploaded row {selected_upload_idx + 1}"
                    )
                    st.markdown(f'<div class="employee-title">{employee_label}</div>', unsafe_allow_html=True)
                    risk_badge(selected_category, selected_probability_leave)

                    emp_summary_cols = st.columns(3)
                    with emp_summary_cols[0]:
                        metric_card("Leave Risk", f"{selected_probability_leave:.2%}", "Model probability")
                    with emp_summary_cols[1]:
                        metric_card("Stay Probability", f"{selected_probability_stay:.2%}", "Complementary score")
                    with emp_summary_cols[2]:
                        percentile = (
                            (results_df["PredictedAttritionRisk"] <= selected_probability_leave).mean()
                            * 100
                        )
                        metric_card("Risk Percentile", f"{percentile:.0f}%", "Vs uploaded dataset")

                    if actual_attrition is not None:
                        actual_value = actual_attrition.loc[selected_upload_idx]
                        if str(actual_value).lower() in ["yes", "1", "1.0"]:
                            st.error(f"Actual Attrition: {actual_value}")
                        else:
                            st.success(f"Actual Attrition: {actual_value}")

                    with st.spinner("Calculating employee-level explainability values..."):
                        selected_shap_values, selected_expected_value = (
                            get_shap_values_for_class_one(model, selected_input_df)
                        )

                    selected_shap_df = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "SHAP_Value": selected_shap_values,
                        }
                    )
                    selected_top_factors = selected_shap_df.reindex(
                        selected_shap_df["SHAP_Value"]
                        .abs()
                        .sort_values(ascending=False)
                        .index
                    ).head(5)

                    st.markdown("#### Personalized Drivers")
                    for _, row in selected_top_factors.iterrows():
                        direction = (
                            "increases flight risk"
                            if row["SHAP_Value"] > 0
                            else "decreases flight risk"
                        )
                        feature = row["Feature"]
                        feature_value = (
                            selected_raw[feature]
                            if feature in selected_raw.index
                            else selected_input_df.iloc[0][feature]
                        )
                        st.write(
                            f"**{feature}** = `{feature_value}` {direction} "
                            f"(impact: {row['SHAP_Value']:+.4f})"
                        )

                    report_lines = [
                        f"Employee: {employee_label}",
                        f"Risk category: {selected_category}",
                        f"Probability of leaving: {selected_probability_leave:.2%}",
                        f"Probability of staying: {selected_probability_stay:.2%}",
                        "",
                        "Top personalized drivers:",
                    ]
                    for _, row in selected_top_factors.iterrows():
                        report_lines.append(
                            f"- {row['Feature']}: {row['SHAP_Value']:+.4f}"
                        )
                    st.download_button(
                        "Download selected employee report",
                        data="\n".join(report_lines).encode("utf-8"),
                        file_name=f"{employee_label.lower().replace(' ', '_')}_risk_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("#### 📊 Detailed Analysis")

                graph_col1, graph_col2, graph_col3 = st.columns(3, gap="medium")
                with graph_col1:
                    st.subheader("Risk Comparison", anchor=False)
                    render_risk_comparison_chart(selected_probability_leave, avg_risk, theme)
                with graph_col2:
                    st.subheader("Driver Impact", anchor=False)
                    render_top_factor_chart(selected_top_factors, theme)
                with graph_col3:
                    st.subheader("Profile Shape", anchor=False)
                    render_profile_radar(selected_raw, theme)

                st.markdown("---")
                with st.expander("🔍 View Detailed SHAP Explanation"):
                    shap_container = st.container()
                    with shap_container:
                        shap.force_plot(
                            selected_expected_value,
                            selected_shap_values,
                            selected_input_df.iloc[0],
                            matplotlib=True,
                            show=False,
                        )
                        st.pyplot(plt.gcf(), use_container_width=True)
                        plt.close()

                st.markdown("---")
                st.subheader("🚨 Highest Risk Employees", anchor=False)

                # Filter to show only employees with Medium or High Risk (> 30% probability)
                high_risk_df = results_df[results_df["PredictedAttritionRisk"] > 0.30]
                if not high_risk_df.empty:
                    top_risk = high_risk_df.sort_values("PredictedAttritionRisk", ascending=False).head(15)
                    st.dataframe(
                        top_risk[visible_columns].style.format(
                            {"PredictedAttritionRisk": "{:.2%}"}
                        ),
                        use_container_width=True,
                        height=430,
                    )
                else:
                    st.info("🎉 No high or medium risk employees detected in the uploaded dataset.")

            except Exception as e:
                st.error(f"Could not process uploaded dataset: {e}")

    else:
        col_input, col_output = st.columns([1, 1.2], gap="medium")

        with col_input:
            if input_mode == "Interactive Simulator (Manual)":
                st.subheader("🎛️ Customize Simulated Profile", anchor=False)

                with st.expander("👤 Personal Profile", expanded=True):
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        age = st.slider("Age", 18, 65, 30)
                        gender = st.selectbox("Gender", ["Female", "Male"])
                        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                    with sub_col2:
                        distance_from_home = st.slider("Distance From Home (Miles)", 1, 30, 8)
                        education = st.slider("Education Level (1-5)", 1, 5, 3)
                        education_field = st.selectbox(
                            "Education Field",
                            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"],
                        )

                with st.expander("💼 Work Role & Schedule", expanded=True):
                    sub_col3, sub_col4 = st.columns(2)
                    with sub_col3:
                        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
                        business_travel = st.selectbox(
                            "Business Travel",
                            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                        )
                        department = st.selectbox(
                            "Department",
                            ["Sales", "Research & Development", "Human Resources"],
                        )
                    with sub_col4:
                        job_role = st.selectbox(
                            "Job Role",
                            [
                                "Sales Executive",
                                "Research Scientist",
                                "Laboratory Technician",
                                "Manufacturing Director",
                                "Healthcare Representative",
                                "Manager",
                                "Sales Representative",
                                "Research Director",
                                "Human Resources",
                            ],
                        )
                        job_level = st.slider("Job Level (1-5)", 1, 5, 2)
                        training_times = st.slider("Training Times Last Year", 0, 6, 2)

                with st.expander("💰 Financials & Tenure", expanded=False):
                    sub_col5, sub_col6 = st.columns(2)
                    with sub_col5:
                        monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, 100)
                        salary_hike = st.slider("Salary Hike (%)", 11, 25, 14)
                        stock_level = st.slider("Stock Option Level (0-3)", 0, 3, 0)
                        daily_rate = st.number_input("Daily Rate ($)", 100, 1500, 800)
                    with sub_col6:
                        years_at_company = st.slider("Years at Company", 0, 40, 3)
                        years_in_role = st.slider("Years in Current Role", 0, 40, 2)
                        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
                        years_with_manager = st.slider("Years with Current Manager", 0, 20, 2)
                        total_working_years = st.slider("Total Working Years", 0, 40, 6)
                        num_companies_worked = st.slider("Companies Worked Before", 0, 10, 2)
                        hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 65)
                        monthly_rate = st.number_input("Monthly Rate ($)", 2000, 30000, 14000)

                with st.expander("😊 Satisfaction & Performance", expanded=False):
                    sub_col7, sub_col8 = st.columns(2)
                    with sub_col7:
                        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
                        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
                        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
                    with sub_col8:
                        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
                        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
                        performance_rating = st.slider("Performance Rating (3-4)", 3, 4, 3)

                overtime_mapped = 1 if overtime == "Yes" else 0
                gender_mapped = encoders["Gender"].transform([gender])[0]
                travel_mapped = encoders["BusinessTravel"].transform([business_travel])[0]
                dept_mapped = encoders["Department"].transform([department])[0]
                marital_mapped = encoders["MaritalStatus"].transform([marital_status])[0]
                field_mapped = encoders["EducationField"].transform([education_field])[0]
                role_mapped = encoders["JobRole"].transform([job_role])[0]

                active_features = {
                    "Age": age,
                    "MonthlyIncome": monthly_income,
                    "OverTime": overtime_mapped,
                    "JobSatisfaction": job_satisfaction,
                    "YearsAtCompany": years_at_company,
                    "TotalWorkingYears": total_working_years,
                    "DistanceFromHome": distance_from_home,
                    "NumCompaniesWorked": num_companies_worked,
                    "JobLevel": job_level,
                    "EnvironmentSatisfaction": environment_satisfaction,
                    "WorkLifeBalance": work_life_balance,
                    "YearsSinceLastPromotion": years_since_promotion,
                    "YearsInCurrentRole": years_in_role,
                    "YearsWithCurrManager": years_with_manager,
                    "MaritalStatus": marital_mapped,
                    "BusinessTravel": travel_mapped,
                    "Department": dept_mapped,
                    "Education": education,
                    "EducationField": field_mapped,
                    "Gender": gender_mapped,
                    "JobInvolvement": job_involvement,
                    "JobRole": role_mapped,
                    "PerformanceRating": performance_rating,
                    "RelationshipSatisfaction": relationship_satisfaction,
                    "StockOptionLevel": stock_level,
                    "TrainingTimesLastYear": training_times,
                    "DailyRate": daily_rate,
                    "HourlyRate": hourly_rate,
                    "MonthlyRate": monthly_rate,
                    "PercentSalaryHike": salary_hike,
                }
            else:
                st.subheader("👥 Employee Database Explorer", anchor=False)

                db_summary_cols = st.columns(3, gap="medium")
                with db_summary_cols[0]:
                    metric_card(
                        "Records Matched",
                        f"{len(filtered_historical_df):,}",
                        "After sidebar filters",
                    )
                with db_summary_cols[1]:
                    metric_card(
                        "Dataset Avg Risk",
                        f"{historical_results_df['PredictedAttritionRisk'].mean():.2%}",
                        "All historical records",
                    )
                with db_summary_cols[2]:
                    high_risk_share = (
                        historical_results_df["RiskCategory"].eq("High Risk").mean()
                    )
                    metric_card(
                        "High Risk Share",
                        f"{high_risk_share:.2%}",
                        "Above 70% risk",
                    )

                st.markdown("#### 📋 Filtered Employee List")
                historical_visible_columns = [
                    "EmployeeNumber",
                    "Age",
                    "Department",
                    "JobRole",
                    "MonthlyIncome",
                    "OverTime",
                    "YearsAtCompany",
                    "Attrition",
                    "PredictedAttritionRisk",
                    "RiskCategory",
                ]
                historical_visible_columns = [
                    col for col in historical_visible_columns if col in historical_results_df.columns
                ]
                st.dataframe(
                    filtered_historical_df.sort_values(
                        "PredictedAttritionRisk",
                        ascending=False,
                    )[historical_visible_columns]
                    .head(30)
                    .style.format({"PredictedAttritionRisk": "{:.2%}"}),
                    height=390,
                    use_container_width=True,
                )

                st.markdown("#### Selected Employee Profile")
                display_row = df_display.iloc[row_idx]
                profile_columns = [
                    "Age",
                    "Gender",
                    "Department",
                    "JobRole",
                    "MonthlyIncome",
                    "OverTime",
                    "YearsAtCompany",
                    "TotalWorkingYears",
                    "JobSatisfaction",
                    "EnvironmentSatisfaction",
                    "WorkLifeBalance",
                    "PerformanceRating",
                ]
                profile_columns = [col for col in profile_columns if col in display_row.index]
                profile_df = display_row[profile_columns].to_frame("Value")
                st.dataframe(profile_df, height=380, use_container_width=True)

        with col_output:
            st.subheader("📊 Risk Assessment", anchor=False)

            input_df = build_prediction_frame(active_features, feature_names)
            probability_leave = float(model.predict_proba(input_df)[0][1])
            probability_stay = 1 - probability_leave
            reference_probabilities = model.predict_proba(
                df_model.drop(columns=["Attrition"])[feature_names]
            )[:, 1]
            reference_avg_risk = float(reference_probabilities.mean())

            category = risk_bucket(probability_leave)
            risk_badge(category, probability_leave)

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                metric_card("Leave Risk", f"{probability_leave:.2%}", "Model probability")
            with kpi_col2:
                metric_card("Stay Probability", f"{probability_stay:.2%}", "Complementary score")
            with kpi_col3:
                percentile = (reference_probabilities <= probability_leave).mean() * 100
                metric_card("Risk Percentile", f"{percentile:.0f}%", "Vs historical database")

            if is_historical:
                st.markdown("---")
                st.subheader("Historical Verification")
                if "Left" in historical_status:
                    st.error(f"**Actual Historical Outcome:** {historical_status} (Yes)")
                    st.caption("The employee resigned from the company in the original records.")
                else:
                    st.success(f"**Actual Historical Outcome:** {historical_status} (No)")
                    st.caption("The employee remained active with the company in the original records.")

            st.markdown("---")

            with st.spinner("Calculating Shapley explainability values..."):
                shap_values_class1, expected_value_class1 = get_shap_values_for_class_one(
                    model, input_df
                )

            st.subheader("🎯 Personalized Drivers", anchor=False)
            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP_Value": shap_values_class1,
                }
            )
            top_factors = shap_df.reindex(
                shap_df["SHAP_Value"].abs().sort_values(ascending=False).index
            ).head(5)
            st.markdown("---")

            st.subheader("💡 AI Recommendations", anchor=False)

            recommendations = []

            if "OverTime" in active_features:
                if active_features["OverTime"] == 1:
                    recommendations.append(
                        "Reduce overtime burden to improve employee satisfaction."
                    )

            if "WorkLifeBalance" in active_features:
                if active_features["WorkLifeBalance"] <= 2:
                    recommendations.append(
                        "Improve work-life balance through flexible policies."
                    )

            if "StockOptionLevel" in active_features:
                if active_features["StockOptionLevel"] == 0:
                    recommendations.append(
                        "Consider stock options or financial incentives."
                    )

            if "YearsSinceLastPromotion" in active_features:
                if active_features["YearsSinceLastPromotion"] >= 3:
                    recommendations.append(
                        "Review promotion and career growth opportunities."
                    )

            if recommendations:
                for rec in recommendations:
                    st.info(f"💡 {rec}")
            else:
                st.success(
                    "No major retention concerns detected for this employee."
                )

            for _, row in top_factors.iterrows():
                direction = "increases flight risk" if row["SHAP_Value"] > 0 else "decreases flight risk"
                bullet = "🔴" if row["SHAP_Value"] > 0 else "🟢"
                st.write(f"{bullet} **{row['Feature']}**: {direction}")

            # Enhanced graph section with better spacing
            st.markdown("#### 📊 Detailed Analysis")

            graph_col1, graph_col2 = st.columns(2, gap="medium")
            with graph_col1:
                st.subheader("Risk Comparison", anchor=False)
                render_risk_comparison_chart(probability_leave, reference_avg_risk, theme)
            with graph_col2:
                st.subheader("Driver Impact", anchor=False)
                render_top_factor_chart(top_factors, theme)

            st.markdown("---")

            st.subheader("Profile Shape", anchor=False)
            radar_col = st.columns(1)[0]
            with radar_col:
                if input_mode == "Search Existing Employee":
                    render_profile_radar(df_display.iloc[row_idx], theme)
                else:
                    render_profile_radar(pd.Series(active_features), theme)

            report_label = (
                f"employee_{selected_id}"
                if input_mode == "Search Existing Employee"
                else "simulated_employee"
            )
            report_lines = [
                f"Profile: {report_label}",
                f"Risk category: {category}",
                f"Probability of leaving: {probability_leave:.2%}",
                f"Probability of staying: {probability_stay:.2%}",
                "",
                "Top personalized drivers:",
            ]
            for _, row in top_factors.iterrows():
                report_lines.append(f"- {row['Feature']}: {row['SHAP_Value']:+.4f}")
            st.download_button(
                "⬇️ Download Employee Report",
                data="\n".join(report_lines).encode("utf-8"),
                file_name=f"{report_label}_risk_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("---")

            with st.expander("🔍 View Detailed SHAP Explanation", expanded=False):
                st.markdown("#### SHAP Force Plot Explanation")
                st.write("This visualization shows how each feature contributes to the model's prediction. "
                         "Red features push the prediction toward attrition, blue features push against it.")

                shap_plot_col = st.columns(1)[0]
                with shap_plot_col:
                    shap.force_plot(
                        expected_value_class1,
                        shap_values_class1,
                        input_df.iloc[0],
                        matplotlib=True,
                        show=False,
                    )
                    st.pyplot(plt.gcf(), use_container_width=True)
                    plt.close()


with tab2:
    st.markdown("#### 🔬 Global Predictive Indicators")
    st.write("These metrics represent patterns computed across the employee database during training.")

    st.markdown("---")

    col_img1, col_img2 = st.columns(2, gap="large")
    with col_img1:
        st.subheader("📊 Top 10 Feature Importance", anchor=False)
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)
        else:
            st.info("📁 Missing 'feature_importance.png'. Run the training script to generate it.")

    with col_img2:
        st.subheader("🔍 Global Explanation Matrix (SHAP)", anchor=False)
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", use_container_width=True)
        else:
            st.info(
        "📁 Missing 'shap_summary.png'. Run the training script to generate it."
    )