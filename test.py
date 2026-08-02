from __future__ import annotations

import html
import json
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

st.set_page_config(
    page_title="Job Search AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

DATE_OPTIONS = {
    "Today": "today",
    "Last 3 days": "3days",
    "Last week": "week",
    "Last month": "month",
    "Any time": "all",
}


def get_secret(name: str, default: str = "") -> str:
    """Read a value from Streamlit Secrets first, then from environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #070A12;
            --panel: rgba(17, 23, 40, 0.76);
            --panel-strong: #11172A;
            --stroke: rgba(255, 255, 255, 0.10);
            --text: #F7F8FC;
            --muted: #AAB2C8;
            --primary: #8B5CF6;
            --primary-2: #22D3EE;
            --success: #34D399;
        }

        html, body, [class*="css"], .stApp {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            color: var(--text);
            background:
                radial-gradient(circle at 12% 2%, rgba(139, 92, 246, 0.22), transparent 28%),
                radial-gradient(circle at 90% 8%, rgba(34, 211, 238, 0.14), transparent 25%),
                linear-gradient(145deg, #070A12 0%, #0A0F1D 48%, #060810 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(12, 16, 29, 0.98), rgba(8, 11, 21, 0.98));
            border-right: 1px solid var(--stroke);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.1rem 2.2rem;
            margin-bottom: 1.25rem;
            border: 1px solid var(--stroke);
            border-radius: 28px;
            background:
                linear-gradient(120deg, rgba(139, 92, 246, 0.18), rgba(34, 211, 238, 0.08)),
                rgba(13, 18, 33, 0.78);
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(16px);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -85px;
            top: -110px;
            border-radius: 50%;
            background: rgba(139, 92, 246, 0.23);
            filter: blur(8px);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.72rem;
            border: 1px solid rgba(139, 92, 246, 0.4);
            border-radius: 999px;
            background: rgba(139, 92, 246, 0.11);
            color: #D9CCFF;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            margin: 1rem 0 0.6rem;
            max-width: 760px;
            font-size: clamp(2.1rem, 5vw, 4.25rem);
            line-height: 1.02;
            letter-spacing: -0.055em;
            font-weight: 800;
        }

        .gradient-text {
            background: linear-gradient(90deg, #C4B5FD, #67E8F9 70%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 700px;
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        div[data-testid="stForm"] {
            padding: 1rem 1rem 0.35rem;
            border: 1px solid var(--stroke);
            border-radius: 22px;
            background: rgba(14, 19, 35, 0.72);
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
        }

        [data-testid="stTextInput"] div[data-baseweb="input"],
        [data-testid="stNumberInput"] div[data-baseweb="input"],
        .stSelectbox div[data-baseweb="select"] > div {
            min-height: 48px;
            border: 1px solid rgba(255, 255, 255, 0.11) !important;
            border-radius: 13px !important;
            background: #F8FAFC !important;
        }

        /* Streamlit/BaseWeb may change the wrapper selector between releases.
           Target every relevant text-input layer so typed values stay readable. */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        input[type="text"],
        input[type="number"] {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #7C3AED !important;
            background-color: transparent !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stNumberInput"] input::placeholder,
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="base-input"] input::placeholder,
        input[type="text"]::placeholder,
        input[type="number"]::placeholder {
            color: #6B7280 !important;
            -webkit-text-fill-color: #6B7280 !important;
            opacity: 1 !important;
        }

        [data-testid="stTextInput"] input:-webkit-autofill,
        [data-testid="stTextInput"] input:-webkit-autofill:hover,
        [data-testid="stTextInput"] input:-webkit-autofill:focus {
            -webkit-text-fill-color: #111827 !important;
            -webkit-box-shadow: 0 0 0 1000px #F8FAFC inset !important;
            transition: background-color 9999s ease-out 0s;
        }

        [data-testid="stTextInput"] input::selection,
        [data-testid="stNumberInput"] input::selection {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            background: #6D28D9 !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
        [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
            border-color: rgba(139, 92, 246, 0.8) !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.14) !important;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 48px;
            border: 0;
            border-radius: 13px;
            color: white;
            font-weight: 750;
            background: linear-gradient(100deg, #7C3AED, #2563EB 58%, #0891B2);
            box-shadow: 0 12px 28px rgba(79, 70, 229, 0.26);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 34px rgba(79, 70, 229, 0.34);
        }

        .section-title {
            margin: 1.7rem 0 0.75rem;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .agent-note {
            margin: 0.8rem 0 1.2rem;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-radius: 16px;
            background: rgba(139, 92, 246, 0.075);
            color: #D6D9E5;
            line-height: 1.65;
        }

        .job-card {
            height: 100%;
            min-height: 320px;
            display: flex;
            flex-direction: column;
            padding: 1.15rem;
            margin-bottom: 1rem;
            border: 1px solid var(--stroke);
            border-radius: 20px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.018)),
                rgba(13, 18, 33, 0.84);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }

        .job-card:hover {
            transform: translateY(-3px);
            border-color: rgba(139, 92, 246, 0.46);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
        }

        .job-top {
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
        }

        .company-mark {
            width: 46px;
            height: 46px;
            flex: 0 0 46px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            color: white;
            font-weight: 800;
            background: linear-gradient(135deg, #8B5CF6, #0891B2);
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.22);
        }

        .job-title {
            margin: 0;
            color: #FFFFFF;
            font-size: 1.02rem;
            line-height: 1.4;
            font-weight: 750;
        }

        .company-name {
            margin-top: 0.24rem;
            color: #B7C0D7;
            font-size: 0.88rem;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin: 1rem 0 0.85rem;
        }

        .chip {
            padding: 0.34rem 0.58rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            color: #C7CEE0;
            background: rgba(255, 255, 255, 0.04);
            font-size: 0.73rem;
            font-weight: 600;
        }

        .job-description {
            flex: 1;
            margin: 0;
            color: #9EA8C0;
            font-size: 0.84rem;
            line-height: 1.65;
        }

        .job-footer {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            align-items: center;
            margin-top: 1rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
        }

        .source {
            color: #828CA6;
            font-size: 0.72rem;
        }

        .apply-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.58rem 0.82rem;
            border-radius: 10px;
            color: white !important;
            text-decoration: none !important;
            font-size: 0.78rem;
            font-weight: 750;
            background: linear-gradient(100deg, #7C3AED, #2563EB);
        }

        [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid var(--stroke);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.035);
        }

        .sidebar-brand {
            padding: 0.15rem 0 1.1rem;
        }

        .sidebar-brand strong {
            font-size: 1.15rem;
            letter-spacing: -0.03em;
        }

        .sidebar-brand span {
            display: block;
            margin-top: 0.3rem;
            color: #8F99B2;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        /* ---------- Premium contrast system v1.4 ---------- */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            color: #F8FAFC !important;
        }

        /* Streamlit widget labels can inherit a low-contrast theme color. */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stTextInput"] label p,
        [data-testid="stSelectbox"] label p,
        [data-testid="stSlider"] label p,
        [data-testid="stCheckbox"] label p {
            color: #DCE5F5 !important;
            -webkit-text-fill-color: #DCE5F5 !important;
            opacity: 1 !important;
            font-weight: 650 !important;
        }

        [data-testid="stSidebar"] {
            color: #F8FAFC !important;
            box-shadow: 18px 0 50px rgba(0, 0, 0, 0.26);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] span,
        [data-testid="stSidebar"] [data-testid="stSlider"] p {
            color: #C9D5E8 !important;
            -webkit-text-fill-color: #C9D5E8 !important;
            opacity: 1 !important;
        }

        [data-testid="stCheckbox"] svg {
            color: #F8FAFC !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: linear-gradient(135deg, #A78BFA, #22D3EE) !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 0 0 5px rgba(139, 92, 246, 0.18) !important;
        }

        [data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
        [data-testid="stSlider"] div[style*="transform"] {
            color: #F8FAFC !important;
            -webkit-text-fill-color: #F8FAFC !important;
            font-weight: 800 !important;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        small {
            color: #AEBBD0 !important;
            -webkit-text-fill-color: #AEBBD0 !important;
            opacity: 1 !important;
        }

        .sidebar-brand strong {
            color: #FFFFFF !important;
            text-shadow: 0 0 22px rgba(139, 92, 246, 0.3);
        }

        .sidebar-brand span {
            color: #C5D0E4 !important;
        }

        .sidebar-section-title {
            margin: 0.7rem 0 1rem;
            color: #F8FAFC;
            font-size: 1.02rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .section-title {
            color: #F8FAFC !important;
            font-size: 1.08rem;
            text-shadow: 0 0 22px rgba(139, 92, 246, 0.2);
        }

        .agent-note {
            border: 1px solid rgba(167, 139, 250, 0.40);
            background:
                linear-gradient(135deg, rgba(124, 58, 237, 0.13), rgba(14, 165, 233, 0.065)),
                rgba(15, 23, 42, 0.88);
            color: #EDF2FF !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.045), 0 14px 38px rgba(0, 0, 0, 0.2);
            font-size: 0.96rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 0.8rem;
        }

        .stat-card {
            position: relative;
            overflow: hidden;
            min-height: 108px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(30, 41, 59, 0.92), rgba(13, 20, 36, 0.94));
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .stat-card::after {
            content: "";
            position: absolute;
            width: 90px;
            height: 90px;
            right: -32px;
            top: -34px;
            border-radius: 50%;
            background: rgba(56, 189, 248, 0.11);
            filter: blur(2px);
        }

        .stat-label {
            display: block;
            position: relative;
            z-index: 1;
            margin-bottom: 0.38rem;
            color: #BFCBE0;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.025em;
            text-transform: uppercase;
        }

        .stat-value {
            position: relative;
            z-index: 1;
            color: #FFFFFF;
            font-size: 2rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.045em;
        }

        .search-meta {
            margin: 0.2rem 0 1.25rem;
            color: #AEBBD0;
            font-size: 0.79rem;
            font-weight: 550;
        }

        .search-meta strong {
            color: #DDD6FE;
            font-weight: 750;
        }

        .job-card {
            border-color: rgba(148, 163, 184, 0.17);
            background:
                linear-gradient(145deg, rgba(30, 41, 59, 0.82), rgba(10, 16, 30, 0.94));
        }

        .company-name { color: #D3DCEC !important; }
        .chip {
            color: #E4EAF5 !important;
            border-color: rgba(167, 139, 250, 0.18);
            background: rgba(139, 92, 246, 0.09);
        }
        .job-description { color: #C3CDE0 !important; }
        .source { color: #A4B0C5 !important; }

        .apply-link {
            background: linear-gradient(100deg, #8B5CF6, #2563EB 52%, #0891B2);
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.28);
        }

        .popular-hint {
            color: #B8C4D8;
            font-size: 0.84rem;
            font-weight: 600;
        }

        [data-testid="stDivider"] {
            border-color: rgba(148, 163, 184, 0.15) !important;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero {
                padding: 1.45rem;
                border-radius: 22px;
            }
            .job-card {
                min-height: auto;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    """Create a connection-pooled HTTP session with safe retries."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def safe_url(value: Any) -> str:
    """Allow only standard HTTP(S) links before putting them in HTML."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def first_non_empty(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def compact_text(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_job(raw: dict[str, Any]) -> dict[str, Any]:
    city = raw.get("job_city")
    state = raw.get("job_state")
    country = raw.get("job_country")
    location_parts = [str(item).strip() for item in (city, state, country) if item]
    location = ", ".join(dict.fromkeys(location_parts))
    if not location:
        location = str(first_non_empty(raw, "job_location", "location", default="Not specified"))

    employment_type = first_non_empty(
        raw,
        "job_employment_type",
        "employment_type",
        "job_type",
        default="Not specified",
    )
    if isinstance(employment_type, list):
        employment_type = ", ".join(map(str, employment_type))

    apply_link = first_non_empty(raw, "job_apply_link", "apply_link", "url")
    if not apply_link:
        options = raw.get("apply_options") or raw.get("job_apply_options") or []
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict):
                    apply_link = first_non_empty(option, "apply_link", "link", "url")
                    if apply_link:
                        break

    posted = first_non_empty(
        raw,
        "job_posted_at",
        "job_posted_human_readable",
        "job_posted_at_datetime_utc",
        "posted_at",
        default="Recently",
    )

    salary_min = raw.get("job_min_salary")
    salary_max = raw.get("job_max_salary")
    salary_period = raw.get("job_salary_period")
    salary = ""
    if salary_min is not None or salary_max is not None:
        low = f"{salary_min:,.0f}" if isinstance(salary_min, (int, float)) else str(salary_min or "")
        high = f"{salary_max:,.0f}" if isinstance(salary_max, (int, float)) else str(salary_max or "")
        salary = " – ".join(part for part in (low, high) if part)
        if salary_period:
            salary = f"{salary} / {salary_period}"

    return {
        "title": str(first_non_empty(raw, "job_title", "title", default="Untitled role")),
        "company": str(first_non_empty(raw, "employer_name", "company_name", "company", default="Company not listed")),
        "location": location,
        "job_type": str(employment_type),
        "posted": str(posted),
        "apply_link": safe_url(apply_link),
        "description": compact_text(first_non_empty(raw, "job_description", "description"), 430),
        "publisher": str(first_non_empty(raw, "job_publisher", "publisher", "source", default="Job listing")),
        "remote": bool(first_non_empty(raw, "job_is_remote", "is_remote", default=False)),
        "salary": salary,
    }


def extract_raw_jobs(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    candidates: Any = payload.get("data")
    if candidates is None:
        candidates = payload.get("jobs")
    if candidates is None:
        candidates = payload.get("results")

    if isinstance(candidates, dict):
        candidates = candidates.get("jobs") or candidates.get("data") or candidates.get("results") or []

    if not isinstance(candidates, list):
        return []

    return [item for item in candidates if isinstance(item, dict)]


@st.cache_data(ttl=300, show_spinner=False, max_entries=100)
def search_jobs_api(
    query: str,
    country: str = "in",
    date_posted: str = "today",
    remote_only: bool = False,
    max_results: int = 12,
) -> dict[str, Any]:
    """Call JSearch and return a small, normalized response safe for the UI/LLM."""
    api_key = get_secret("JSEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("JSEARCH_API_KEY is missing.")

    clean_query = " ".join(query.split()).strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")

    params: dict[str, Any] = {
        "query": clean_query,
        "country": country,
        "date_posted": date_posted,
    }
    if remote_only:
        params["work_from_home"] = "true"

    try:
        response = get_http_session().get(
            JSEARCH_URL,
            headers={"X-API-KEY": api_key},
            params=params,
            timeout=(5, 25),
        )
    except requests.Timeout as exc:
        raise RuntimeError("Job API timed out. Please retry.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Could not connect to the job search service.") from exc

    if response.status_code == 401:
        raise RuntimeError("JSearch rejected the API key.")
    if response.status_code == 429:
        raise RuntimeError("JSearch rate limit reached. Please try again shortly.")
    if response.status_code >= 400:
        raise RuntimeError(f"JSearch returned HTTP {response.status_code}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("JSearch returned an invalid JSON response.") from exc

    jobs = [normalize_job(item) for item in extract_raw_jobs(payload)]
    max_results = max(1, min(int(max_results), 20))

    return {
        "query": clean_query,
        "total": len(jobs[:max_results]),
        "jobs": jobs[:max_results],
    }


@tool
def search_job(
    query: str,
    country: str = "in",
    date_posted: str = "today",
    remote_only: bool = False,
    max_results: int = 12,
) -> str:
    """
    Search current job postings.

    Args:
        query: Job title/skills plus location, for example "Data Analyst in Bengaluru, India".
        country: Two-letter country code. Use "in" for India.
        date_posted: One of today, 3days, week, month, or all.
        remote_only: True to return only remote/work-from-home jobs.
        max_results: Number of results to return, from 1 to 20.
    """
    result = search_jobs_api(
        query=query,
        country=country,
        date_posted=date_posted,
        remote_only=remote_only,
        max_results=max_results,
    )
    return json.dumps(result, ensure_ascii=False)


@st.cache_resource(show_spinner=False)
def get_agent(model_name: str):
    api_key = get_secret("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing.")

    # ChatNVIDIA reads NVIDIA_API_KEY from the environment.
    os.environ["NVIDIA_API_KEY"] = api_key

    llm = ChatNVIDIA(
        model=model_name,
        temperature=0.1,
        top_p=0.9,
        max_tokens=900,
    )

    system_prompt = """
    You are HireLens, a precise job-search agent.

    Rules:
    1. For every job-search request, call the search_job tool exactly once.
    2. Preserve the requested role, skills, location, country, posted-date filter,
       remote preference, and result count.
    3. Never invent jobs, companies, links, locations, or dates.
    4. After the tool result, write only a concise 2-4 sentence overview:
       mention the number of matches, notable patterns, and one practical search tip.
    5. Do not repeat the full job list because the UI renders job cards separately.
    """
    return create_agent(model=llm, tools=[search_job], system_prompt=system_prompt)


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    pieces.append(str(text))
        return "\n".join(pieces).strip()
    return str(content or "").strip()


def jobs_from_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        class_name = message.__class__.__name__.lower()
        if message_type == "tool" or "toolmessage" in class_name:
            content = message_text(message)
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
                return payload["jobs"]
    return []


def summary_from_agent_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        class_name = message.__class__.__name__.lower()
        if message_type in {"ai", "assistant"} or "aimessage" in class_name:
            text = message_text(message)
            tool_calls = getattr(message, "tool_calls", None)
            if text and not tool_calls:
                return text
    return ""


def run_search(
    query: str,
    date_posted: str,
    remote_only: bool,
    max_results: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Agent-first search with deterministic direct-API fallback.
    Returns jobs, overview, execution mode.
    """
    nvidia_key = get_secret("NVIDIA_API_KEY")

    if nvidia_key:
        prompt = f"""
        Find current jobs using these exact parameters:
        - query: {query}
        - country: in
        - date_posted: {date_posted}
        - remote_only: {remote_only}
        - max_results: {max_results}
        """
        try:
            result = get_agent(model_name).invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            messages = result.get("messages", [])
            jobs = jobs_from_agent_messages(messages)
            overview = summary_from_agent_messages(messages)
            if jobs:
                if not overview:
                    overview = f"Found {len(jobs)} matching roles. Review the cards below and open the strongest matches in a new tab."
                return jobs, overview, "AI agent"
        except Exception:
            # Deliberately fall through to the deterministic API path.
            pass

    direct = search_jobs_api(
        query=query,
        country="in",
        date_posted=date_posted,
        remote_only=remote_only,
        max_results=max_results,
    )
    jobs = direct["jobs"]
    overview = (
        f"Found {len(jobs)} matching role{'s' if len(jobs) != 1 else ''}. "
        "Results were loaded through the fast API fallback, so every card comes directly from the jobs feed."
    )
    return jobs, overview, "Fast API fallback"


def render_job_card(job: dict[str, Any]) -> str:
    title = html.escape(str(job.get("title") or "Untitled role"))
    company = html.escape(str(job.get("company") or "Company not listed"))
    location = html.escape(str(job.get("location") or "Not specified"))
    job_type = html.escape(str(job.get("job_type") or "Not specified"))
    posted = html.escape(str(job.get("posted") or "Recently"))
    publisher = html.escape(str(job.get("publisher") or "Job listing"))
    description = html.escape(str(job.get("description") or "Open the job listing to view the full description."))
    salary = html.escape(str(job.get("salary") or ""))
    remote = bool(job.get("remote"))
    apply_link = safe_url(job.get("apply_link"))
    initial = html.escape((company[:1] or "J").upper())

    chips = [
        f'<span class="chip">📍 {location}</span>',
        f'<span class="chip">💼 {job_type}</span>',
        f'<span class="chip">🕒 {posted}</span>',
    ]
    if remote:
        chips.append('<span class="chip">🌐 Remote</span>')
    if salary:
        chips.append(f'<span class="chip">₹ {salary}</span>')

    if apply_link:
        apply_html = (
            f'<a class="apply-link" href="{html.escape(apply_link, quote=True)}" '
            'target="_blank" rel="noopener noreferrer">Apply now ↗</a>'
        )
    else:
        apply_html = '<span class="source">Apply link unavailable</span>'

    return f"""
    <article class="job-card">
        <div class="job-top">
            <div class="company-mark">{initial}</div>
            <div>
                <h3 class="job-title">{title}</h3>
                <div class="company-name">{company}</div>
            </div>
        </div>
        <div class="chips">{''.join(chips)}</div>
        <p class="job-description">{description}</p>
        <div class="job-footer">
            <span class="source">Source: {publisher}</span>
            {apply_html}
        </div>
    </article>
    """


def render_results(jobs: list[dict[str, Any]], overview: str, mode: str) -> None:
    st.markdown('<div class="section-title">AI overview</div>', unsafe_allow_html=True)
    clean_overview = overview.replace("**", "").replace("__", "").strip()
    st.markdown(
        f'<div class="agent-note">{html.escape(clean_overview)}</div>',
        unsafe_allow_html=True,
    )

    companies = len({job.get("company") for job in jobs if job.get("company")})
    remote_count = sum(bool(job.get("remote")) for job in jobs)

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Matching roles</span>
                <span class="stat-value">{len(jobs)}</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Companies</span>
                <span class="stat-value">{companies}</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Remote roles</span>
                <span class="stat-value">{remote_count}</span>
            </div>
        </div>
        <div class="search-meta">Search mode: <strong>{html.escape(mode)}</strong> · Results cached for 5 minutes</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Open positions</div>', unsafe_allow_html=True)

    if not jobs:
        st.info("No jobs matched these filters. Try a broader location or a wider posted-date range.")
        return

    columns = st.columns(2)
    for index, job in enumerate(jobs):
        with columns[index % 2]:
            st.markdown(render_job_card(job), unsafe_allow_html=True)


def main() -> None:
    inject_css()

    st.markdown(
        """
        <section class="hero">
            <span class="eyebrow">✦  ConsoleFlare</span>
            <h1>Find work that feels <span class="gradient-text">made for you.</span></h1>
            <p>
                Search fresh roles across India with intelligent matching and a resilient live-jobs API -
                presented in a clean, application-ready workspace.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(            """
            <div class="sidebar-brand">
                <strong>✨ AI - Powered Job Discovery </strong>
                <span>Premium job search powered by live listings by AI agent.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-title">Search settings</div>', unsafe_allow_html=True)
        location = st.text_input("Preferred location", value="India", placeholder="Bengaluru, Pune, Remote…")
        date_label = st.selectbox("Posted", list(DATE_OPTIONS.keys()), index=0)
        remote_only = st.checkbox("Remote jobs only", value=False)
        max_results = st.slider("Results", min_value=4, max_value=20, value=12, step=2)

        st.divider()
        if st.button("Clear results", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.rerun()

    with st.form("job_search_form", clear_on_submit=False):
        col1, col2 = st.columns([3.2, 1])
        with col1:
            role = st.text_input(
                "Role, skills, or keywords",
                value="Data Analyst",
                placeholder="e.g. Data Analyst, SQL, Power BI",
            )
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Search jobs ✦", use_container_width=True)

    if submitted:
        if not role.strip():
            st.warning("Enter a job title or skill.")
        elif not get_secret("JSEARCH_API_KEY"):
            st.error(
                "JSEARCH_API_KEY is missing. Add it to `.env` locally or Streamlit Secrets in deployment."
            )
        else:
            location_text = location.strip() or "India"
            full_query = role.strip()
            if location_text.lower() not in full_query.lower():
                full_query = f"{full_query} in {location_text}"
            if "india" not in full_query.lower():
                full_query = f"{full_query}, India"

            try:
                with st.spinner("Searching live roles and preparing your shortlist…"):
                    jobs, overview, mode = run_search(
                        query=full_query,
                        date_posted=DATE_OPTIONS[date_label],
                        remote_only=remote_only,
                        max_results=max_results,
                        model_name=get_secret("NVIDIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
                    )
                st.session_state["last_result"] = {
                    "jobs": jobs,
                    "overview": overview,
                    "mode": mode,
                }
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get("last_result")
    if result:
        render_results(
            jobs=result["jobs"],
            overview=result["overview"],
            mode=result["mode"],
        )
    else:
        st.markdown('<div class="section-title">Popular searches</div>', unsafe_allow_html=True)
        quick_cols = st.columns(4)
        suggestions = ["Data Analyst", "Python Developer", "Business Analyst", "UI/UX Designer"]
        for col, suggestion in zip(quick_cols, suggestions):
            with col:
                st.markdown(
                    f'<div class="popular-hint">Try “{html.escape(suggestion)}”</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
