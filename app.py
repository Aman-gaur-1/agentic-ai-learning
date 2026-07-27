from __future__ import annotations

import ast
import html
import json
import os
from datetime import datetime, timezone
from textwrap import dedent
from typing import Annotated, Any
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="ConsoleFlare Jobs",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"
COUNTRY_CODE = "in"
LANGUAGE_CODE = "en"
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
REQUEST_TIMEOUT = (5, 25)
CACHE_TTL_SECONDS = 15 * 60
MAX_API_PAGES = 6

DATE_OPTIONS = {
    "Today": "today",
    "Last 3 days": "3days",
    "Last week": "week",
    "Last month": "month",
    "Any time": "all",
}


class AppConfigurationError(RuntimeError):
    """Raised when required application secrets are missing."""


class JobSearchError(RuntimeError):
    """Raised when a user-friendly job search error should be displayed."""


# -----------------------------------------------------------------------------
# Secrets and HTTP helpers
# -----------------------------------------------------------------------------
def read_secret(name: str) -> str | None:
    """Read Streamlit Secrets first, then environment variables/.env."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    value = value or os.getenv(name)
    return str(value).strip() if value else None


def configure_credentials() -> None:
    missing: list[str] = []

    for name in ("JSEARCH_API_KEY", "NVIDIA_API_KEY"):
        value = read_secret(name)
        if value:
            os.environ[name] = value
        else:
            missing.append(name)

    if missing:
        raise AppConfigurationError(
            "Required API keys are missing. Add JSEARCH_API_KEY and "
            "NVIDIA_API_KEY to your .env file or Streamlit Secrets."
        )


def create_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def raise_friendly_http_error(response: requests.Response) -> None:
    status = response.status_code

    if status in (401, 403):
        message = "The job API rejected the configured credentials."
    elif status == 429:
        message = "The job API rate limit was reached. Please try again shortly."
    elif status == 400:
        message = "The job API could not understand this search. Try simpler filters."
    elif 500 <= status <= 599:
        message = "The job provider is temporarily unavailable."
    else:
        message = f"The job search request failed with status code {status}."

    raise JobSearchError(message)


def extract_raw_jobs(payload: Any) -> list[dict[str, Any]]:
    """
    Extract jobs from the different response shapes returned by JSearch.

    Supported examples:
    - {"data": [...]}
    - {"data": {"jobs": [...]}}
    - {"jobs": [...]}
    - {"results": [...]}
    """
    if not isinstance(payload, dict):
        return []

    candidates: Any = payload.get("data")

    if candidates is None:
        candidates = payload.get("jobs")

    if candidates is None:
        candidates = payload.get("results")

    if isinstance(candidates, dict):
        candidates = (
            candidates.get("jobs")
            or candidates.get("data")
            or candidates.get("results")
            or []
        )

    if not isinstance(candidates, list):
        return []

    return [job for job in candidates if isinstance(job, dict)]


def extract_cursor(payload: dict[str, Any]) -> str | None:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    parameters = (
        payload.get("parameters")
        if isinstance(payload.get("parameters"), dict)
        else {}
    )

    values = (
        payload.get("cursor"),
        payload.get("next_cursor"),
        payload.get("next_page_cursor"),
        meta.get("cursor"),
        parameters.get("cursor"),
        parameters.get("next_cursor"),
    )

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=100, show_spinner=False)
def fetch_jobs_from_api(
    query: str,
    country: str,
    date_posted: str,
    work_from_home: bool,
    limit: int,
) -> dict[str, Any]:
    """Fetch, retry, paginate and deduplicate jobs from JSearch."""
    api_key = os.getenv("JSEARCH_API_KEY", "").strip()
    if not api_key:
        raise AppConfigurationError("JSEARCH_API_KEY is not configured.")

    session = create_http_session()
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = None

    try:
        for _ in range(MAX_API_PAGES):
            params: dict[str, Any] = {
                "query": query,
                "country": country,
                "date_posted": date_posted,
            }

            if work_from_home:
                params["work_from_home"] = "true"
            if cursor:
                params["cursor"] = cursor

            try:
                response = session.get(
                    JSEARCH_URL,
                    headers={"X-API-KEY": api_key},
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.Timeout as exc:
                raise JobSearchError(
                    "The job provider took too long to respond. Please try again."
                ) from exc
            except requests.ConnectionError as exc:
                raise JobSearchError(
                    "The job provider could not be reached. Check your internet connection."
                ) from exc
            except requests.RequestException as exc:
                raise JobSearchError(
                    "A network error interrupted the search. Please try again."
                ) from exc

            if not response.ok:
                raise_friendly_http_error(response)

            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise JobSearchError(
                    "The job provider returned an unreadable response."
                ) from exc

            if not isinstance(payload, dict):
                raise JobSearchError("The job provider returned an unexpected response.")

            page_jobs = extract_raw_jobs(payload)

            for job in page_jobs:
                if not isinstance(job, dict):
                    continue

                identity = str(
                    job.get("job_id")
                    or job.get("job_apply_link")
                    or f"{job.get('job_title', '')}|{job.get('employer_name', '')}|{job.get('job_city', '')}"
                )

                if identity in seen:
                    continue

                seen.add(identity)
                jobs.append(job)

                if len(jobs) >= limit:
                    break

            if len(jobs) >= limit:
                break

            next_cursor = extract_cursor(payload)
            if not page_jobs or not next_cursor or next_cursor == cursor:
                break

            cursor = next_cursor
    finally:
        session.close()

    return {
        "status": "OK",
        "data": jobs[:limit],
        "result_count": len(jobs[:limit]),
    }


# -----------------------------------------------------------------------------
# Data normalization
# -----------------------------------------------------------------------------
def first_text(*values: Any, default: str = "Not specified") -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def compact_text(value: Any, limit: int = 260) -> str:
    """Collapse whitespace and shorten long card text."""
    value = " ".join(str(value or "").split())

    if not value:
        return ""

    if len(value) <= limit:
        return value

    return value[: limit - 1].rstrip() + "…"


def build_location(job: dict[str, Any]) -> str:
    direct = first_text(job.get("job_location"), job.get("location"), default="")
    if direct:
        return direct

    parts = [
        str(job.get(key, "")).strip()
        for key in ("job_city", "job_state", "job_country")
        if str(job.get(key, "")).strip()
    ]
    return ", ".join(dict.fromkeys(parts)) or "Location not specified"


def build_posted_time(job: dict[str, Any]) -> str:
    relative = first_text(
        job.get("job_posted_at"),
        job.get("job_posted_human_readable"),
        default="",
    )
    if relative:
        return relative

    iso_value = first_text(
        job.get("job_posted_at_datetime_utc"),
        job.get("job_posted_at_date"),
        default="",
    )
    if iso_value:
        try:
            return datetime.fromisoformat(iso_value.replace("Z", "+00:00")).strftime(
                "%d %b %Y"
            )
        except ValueError:
            return iso_value

    timestamp = job.get("job_posted_at_timestamp")
    try:
        if timestamp:
            return datetime.fromtimestamp(
                int(timestamp), tz=timezone.utc
            ).strftime("%d %b %Y")
    except (TypeError, ValueError, OSError):
        pass

    return "Posted time unavailable"


def normalize_job(job: dict[str, Any]) -> dict[str, Any]:
    employment_type = first_text(
        job.get("job_employment_type"),
        job.get("employment_type"),
        job.get("job_type"),
        default="",
    )

    if not employment_type:
        employment_types = job.get("job_employment_types", [])
        if isinstance(employment_types, list) and employment_types:
            employment_type = ", ".join(str(item) for item in employment_types)

    if not employment_type:
        employment_type = "Not specified"

    arrangement = str(
        first_text(
            job.get("work_arrangement"),
            job.get("job_work_arrangement"),
            default="",
        )
    ).lower()

    remote_value = job.get("job_is_remote", job.get("is_remote", False))
    remote = bool(remote_value) or arrangement in {
        "remote",
        "work from home",
        "work_from_home",
    }

    apply_link = first_text(
        job.get("job_apply_link"),
        job.get("apply_link"),
        job.get("url"),
        default="",
    )

    if not apply_link:
        options = job.get("apply_options") or job.get("job_apply_options") or []
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, dict):
                    continue

                apply_link = first_text(
                    option.get("apply_link"),
                    option.get("link"),
                    option.get("url"),
                    default="",
                )

                if apply_link:
                    break

    salary_min = job.get("job_min_salary")
    salary_max = job.get("job_max_salary")
    salary_period = first_text(job.get("job_salary_period"), default="")

    salary_parts: list[str] = []
    for value in (salary_min, salary_max):
        if isinstance(value, (int, float)):
            salary_parts.append(f"{value:,.0f}")
        elif value not in (None, ""):
            salary_parts.append(str(value))

    salary = " – ".join(salary_parts)
    if salary and salary_period:
        salary = f"{salary} / {salary_period}"

    return {
        "id": first_text(
            job.get("job_id"),
            apply_link,
            default="",
        ),
        "title": first_text(
            job.get("job_title"),
            job.get("title"),
            default="Untitled role",
        ),
        "company": first_text(
            job.get("employer_name"),
            job.get("company_name"),
            job.get("company"),
            default="Company not specified",
        ),
        "location": build_location(job),
        "job_type": employment_type,
        "posted_time": build_posted_time(job),
        "apply_link": apply_link,
        "remote": remote,
        "platform": first_text(
            job.get("job_publisher"),
            job.get("publisher"),
            job.get("source"),
            default="Job listing",
        ),
        "description": compact_text(
            first_text(
                job.get("job_description"),
                job.get("description"),
                default="",
            ),
            280,
        ),
        "salary": salary,
    }


def compact_payload(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    raw_jobs = payload.get("data", [])
    raw_jobs = raw_jobs if isinstance(raw_jobs, list) else []

    jobs = [
        normalize_job(job) for job in raw_jobs if isinstance(job, dict)
    ][:limit]

    return {"status": payload.get("status", "OK"), "data": jobs, "result_count": len(jobs)}


# -----------------------------------------------------------------------------
# LangChain tool and agent
# -----------------------------------------------------------------------------
@tool
def search_jobs(
    query: Annotated[str, Field(description="Job role and location query.")],
    country: Annotated[str, Field(description="Two-letter country code.")] = COUNTRY_CODE,
    date_posted: Annotated[
        str, Field(description="today, 3days, week, month or all")
    ] = "today",
    work_from_home: Annotated[
        bool, Field(description="Return only remote roles when true.")
    ] = False,
    limit: Annotated[int, Field(ge=1, le=30, description="Maximum results.")] = 10,
) -> dict[str, Any]:
    """Search current job postings using the selected filters."""
    payload = fetch_jobs_from_api(
        query=query,
        country=country,
        date_posted=date_posted,
        work_from_home=work_from_home,
        limit=limit,
    )

    compact = compact_payload(payload, limit)
    compact["filters"] = {
        "query": query,
        "country": country,
        "date_posted": date_posted,
        "work_from_home": work_from_home,
        "limit": limit,
    }
    return compact


def get_job_agent() -> Any:
    if "job_agent" not in st.session_state:
        llm = ChatNVIDIA(
            model=NVIDIA_MODEL,
            temperature=0,
            max_completion_tokens=500,
        )

        st.session_state.job_agent = create_agent(
            model=llm,
            tools=[search_jobs],
            system_prompt=(
                "You are a job search agent. Call search_jobs exactly once using "
                "the values provided by the user. Never invent jobs, companies, "
                "locations, dates or application links."
            ),
        )

    return st.session_state.job_agent


def decode_payload(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        for item in content:
            candidate = item.get("text") if isinstance(item, dict) else item
            result = decode_payload(candidate)
            if result:
                return result
        return None

    if not isinstance(content, str) or not content.strip():
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(content)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass

    return None


def extract_tool_payload(agent_result: dict[str, Any]) -> dict[str, Any] | None:
    messages = agent_result.get("messages", [])
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) != "search_jobs":
            continue

        artifact = decode_payload(getattr(message, "artifact", None))
        if artifact:
            return artifact

        content = decode_payload(message.content)
        if content:
            return content

    return None


def run_agent_search(
    role: str,
    location: str,
    date_posted: str,
    remote_only: bool,
    limit: int,
) -> dict[str, Any]:
    query = f"{role} in {location}"

    expected_filters = {
        "query": query,
        "country": COUNTRY_CODE,
        "date_posted": date_posted,
        "work_from_home": remote_only,
        "limit": limit,
    }

    prompt = dedent(
        f"""
        Call search_jobs exactly once with these values:
        query={query!r}
        country={COUNTRY_CODE!r}
        date_posted={date_posted!r}
        work_from_home={remote_only!r}
        limit={limit!r}
        Do not change any value.
        """
    ).strip()

    try:
        result = get_job_agent().with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=True,
        ).invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 8},
        )
    except (AppConfigurationError, JobSearchError):
        raise
    except Exception as exc:
        raise JobSearchError(
            "The AI search service could not complete the request. Please try again."
        ) from exc

    payload = extract_tool_payload(result)
    if payload and payload.get("filters") == expected_filters:
        return payload

    # Reliable deterministic fallback using the same LangChain tool.
    fallback = search_jobs.invoke(expected_filters)

    if isinstance(fallback, dict):
        payload = fallback
    else:
        payload = decode_payload(fallback)

    if not payload:
        raise JobSearchError("The job provider returned an unexpected response.")

    return payload


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def validate(role: str, location: str, limit: int) -> list[str]:
    errors: list[str] = []

    if len(role.strip()) < 2:
        errors.append("Enter a valid job role.")
    if len(role.strip()) > 80:
        errors.append("Keep the job role under 80 characters.")
    if len(location.strip()) < 2:
        errors.append("Enter a valid location.")
    if len(location.strip()) > 80:
        errors.append("Keep the location under 80 characters.")
    if not 1 <= limit <= 30:
        errors.append("Choose between 1 and 30 results.")

    return errors


def safe_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return html.escape(url, quote=True)


def generate_overview(jobs: list[dict[str, Any]], role: str, location: str) -> str:
    if not jobs:
        return (
            f"No matching {role} roles were returned for {location}. Try a broader "
            "location, a wider date range, or disable the remote-only filter."
        )

    companies = {job.get("company") for job in jobs if job.get("company")}
    remote_count = sum(1 for job in jobs if job.get("remote"))
    locations = [job.get("location") for job in jobs if job.get("location")]
    top_locations = list(dict.fromkeys(locations))[:3]
    location_text = ", ".join(top_locations) if top_locations else location

    return (
        f"Found {len(jobs)} {role} roles across {location_text}. "
        f"The results include {len(companies)} companies and {remote_count} remote "
        "role(s). Review the job cards below and open the application link for the "
        "most relevant opportunities."
    )


def render_job_card(job: dict[str, Any]) -> None:
    title = html.escape(str(job.get("title", "Untitled role")))
    company = html.escape(str(job.get("company", "Company not specified")))
    location = html.escape(str(job.get("location", "Location not specified")))
    job_type = html.escape(str(job.get("job_type", "Not specified")))
    posted = html.escape(str(job.get("posted_time", "Posted time unavailable")))
    platform = html.escape(str(job.get("platform", "Job listing")))
    description = html.escape(
        str(
            job.get("description")
            or "Open the listing to view the complete job description."
        )
    )
    salary = html.escape(str(job.get("salary", "")))
    url = safe_url(str(job.get("apply_link", "")))

    remote_badge = (
        '<span class="remote-chip">REMOTE</span>'
        if job.get("remote")
        else ""
    )

    salary_chip = (
        f'<span class="salary-chip">₹ {salary}</span>'
        if salary
        else ""
    )

    action = (
        f'<a class="apply-btn" href="{url}" target="_blank" '
        'rel="noopener noreferrer">Apply now ↗</a>'
        if url
        else '<span class="apply-disabled">Apply link unavailable</span>'
    )

    card_html = f"""<article class="job-card">
<div class="job-card-top">
<div>
<div class="job-title">{title}</div>
<div class="company-name">{company}</div>
</div>
<div class="chip-row">
<span class="type-chip">{job_type}</span>
{remote_badge}
{salary_chip}
</div>
</div>
<div class="job-meta">
<span>⌖ {location}</span>
<span>◷ {posted}</span>
<span>◉ Platform: {platform}</span>
</div>
<p class="job-description">{description}</p>
<div class="job-card-footer">
<span class="platform-label">Listed on {platform}</span>
{action}
</div>
</article>"""

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {
        "jobs": [],
        "last_search": None,
        "search_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_css() -> None:
    st.markdown(
        dedent(
            """
            <style>
            :root {
                --bg: #07101f;
                --panel: rgba(16, 25, 46, 0.92);
                --panel-2: rgba(20, 31, 55, 0.94);
                --border: rgba(148, 163, 184, 0.20);
                --text: #f8fafc;
                --muted: #aab6cb;
                --purple: #7c3aed;
                --blue: #1d9bf0;
                --cyan: #51d7f2;
            }

            html, body, [class*="css"] {
                font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                    BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            .stApp {
                color: var(--text);
                background:
                    radial-gradient(circle at 12% 0%, rgba(124, 58, 237, 0.20), transparent 28%),
                    radial-gradient(circle at 100% 0%, rgba(8, 145, 178, 0.16), transparent 25%),
                    linear-gradient(180deg, #08111f 0%, #050a13 100%);
            }

            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stMainBlockContainer"] {
                max-width: 1080px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1120, #070b15);
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.8rem;
            }

            .sidebar-brand {
                margin: 0.6rem 0 0.2rem;
                font-size: 1.05rem;
                font-weight: 850;
            }

            .sidebar-copy {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.55;
                margin-bottom: 1.4rem;
            }

            .hero {
                position: relative;
                overflow: hidden;
                padding: 2.25rem 2.2rem;
                margin-bottom: 1.25rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 26px;
                background: linear-gradient(135deg, rgba(38, 35, 79, 0.96), rgba(10, 42, 56, 0.92));
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
            }

            .hero::after {
                content: "";
                position: absolute;
                width: 250px;
                height: 250px;
                right: -80px;
                top: -120px;
                border-radius: 50%;
                background: rgba(114, 86, 255, 0.26);
                filter: blur(6px);
            }

            .eyebrow {
                display: inline-block;
                position: relative;
                z-index: 1;
                padding: 0.45rem 0.75rem;
                margin-bottom: 1.05rem;
                border: 1px solid rgba(167, 139, 250, 0.35);
                border-radius: 999px;
                color: #ddd6fe;
                background: rgba(124, 58, 237, 0.12);
                font-size: 0.73rem;
                font-weight: 850;
                letter-spacing: 0.08em;
            }

            .hero h1 {
                position: relative;
                z-index: 1;
                max-width: 860px;
                margin: 0;
                font-size: clamp(2.35rem, 5vw, 4rem);
                line-height: 1.04;
                letter-spacing: -0.055em;
            }

            .gradient-text {
                color: transparent;
                background: linear-gradient(90deg, #c4b5fd, #7dd3fc, #67e8f9);
                background-clip: text;
                -webkit-background-clip: text;
            }

            .hero p {
                position: relative;
                z-index: 1;
                max-width: 760px;
                margin: 1.25rem 0 0;
                color: var(--muted);
                line-height: 1.65;
                font-size: 1rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: rgba(148, 163, 184, 0.18) !important;
                border-radius: 20px !important;
                background: rgba(8, 14, 28, 0.72) !important;
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
            }

            label, [data-testid="stWidgetLabel"] p {
                color: #e5e7eb !important;
                font-size: 0.85rem !important;
                font-weight: 750 !important;
            }

            [data-baseweb="input"] > div,
            [data-baseweb="select"] > div {
                min-height: 3rem;
                border-radius: 11px !important;
                border-color: rgba(148, 163, 184, 0.22) !important;
                background: rgba(248, 250, 252, 0.96) !important;
            }

            [data-baseweb="input"] input,
            [data-baseweb="select"] span {
                color: #111827 !important;
            }

            .stButton > button,
            [data-testid="stFormSubmitButton"] > button {
                min-height: 3rem;
                border: 0 !important;
                border-radius: 11px !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                background: linear-gradient(95deg, #7c3aed, #4f46e5 55%, #0891b2) !important;
                box-shadow: 0 12px 30px rgba(79, 70, 229, 0.24);
            }

            .section-heading {
                margin: 1.65rem 0 0.85rem;
                font-size: 1rem;
                font-weight: 850;
            }

            .overview-box {
                padding: 1.15rem 1.25rem;
                border: 1px solid rgba(139, 92, 246, 0.48);
                border-radius: 16px;
                color: #dbe4f2;
                background: linear-gradient(135deg, rgba(36, 43, 78, 0.92), rgba(12, 39, 58, 0.92));
                line-height: 1.65;
            }

            .metric-box {
                position: relative;
                overflow: hidden;
                min-height: 115px;
                padding: 1rem 1.1rem;
                border: 1px solid var(--border);
                border-radius: 18px;
                background: rgba(22, 34, 57, 0.94);
            }

            .metric-box::after {
                content: "";
                position: absolute;
                width: 80px;
                height: 80px;
                right: -25px;
                top: -25px;
                border-radius: 50%;
                background: rgba(14, 165, 233, 0.12);
            }

            .metric-label {
                color: #aeb9cc;
                font-size: 0.75rem;
                font-weight: 850;
                letter-spacing: 0.04em;
            }

            .metric-value {
                margin-top: 0.55rem;
                font-size: 1.8rem;
                font-weight: 900;
            }

            .job-card {
                padding: 1.25rem 1.3rem;
                margin-bottom: 0.85rem;
                border: 1px solid var(--border);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(18, 28, 49, 0.95), rgba(8, 16, 31, 0.95));
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
            }

            .job-card-top {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
            }

            .job-title { font-size: 1.12rem; font-weight: 850; }
            .company-name { margin-top: 0.32rem; color: #c4b5fd; font-weight: 700; }
            .chip-row { display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: flex-end; }

            .type-chip, .remote-chip {
                padding: 0.36rem 0.58rem;
                border-radius: 999px;
                font-size: 0.68rem;
                font-weight: 850;
                border: 1px solid rgba(167, 139, 250, 0.25);
                background: rgba(124, 58, 237, 0.12);
                color: #ddd6fe;
            }

            .remote-chip {
                color: #a7f3d0;
                border-color: rgba(16, 185, 129, 0.25);
                background: rgba(16, 185, 129, 0.10);
            }

            .salary-chip {
                padding: 0.36rem 0.58rem;
                border-radius: 999px;
                font-size: 0.68rem;
                font-weight: 850;
                color: #fde68a;
                border: 1px solid rgba(245, 158, 11, 0.28);
                background: rgba(245, 158, 11, 0.10);
            }

            .job-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem 1.1rem;
                margin: 0.95rem 0 1rem;
                color: var(--muted);
                font-size: 0.84rem;
            }

            .job-description {
                margin: 0 0 1rem;
                color: #bdc8dc;
                font-size: 0.88rem;
                line-height: 1.65;
            }

            .job-card-footer {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding-top: 0.9rem;
                border-top: 1px solid rgba(148, 163, 184, 0.12);
            }

            .platform-label {
                color: #8794aa;
                font-size: 0.76rem;
                font-weight: 650;
            }

            .apply-btn, .apply-disabled {
                display: inline-flex;
                min-height: 2.5rem;
                align-items: center;
                justify-content: center;
                padding: 0 0.9rem;
                border-radius: 10px;
                font-size: 0.82rem;
                font-weight: 850;
                text-decoration: none !important;
            }

            .apply-btn { color: white !important; background: linear-gradient(95deg, #7c3aed, #2563eb); }
            .apply-disabled { color: #7c8799; border: 1px solid rgba(148, 163, 184, 0.15); }

            .empty-state {
                padding: 2rem 1rem;
                text-align: center;
                color: var(--muted);
                border: 1px dashed rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                background: rgba(15, 23, 42, 0.38);
            }

            .footer {
                margin-top: 2.5rem;
                text-align: center;
                color: #6f7b8f;
                font-size: 0.76rem;
            }

            @media (max-width: 768px) {
                [data-testid="stMainBlockContainer"] { padding: 1rem; }
                .hero { padding: 1.55rem 1.2rem; }
                .hero h1 { font-size: 2.35rem; }
                .job-card-top { flex-direction: column; }
                .chip-row { justify-content: flex-start; }
                .job-card-footer { align-items: flex-start; flex-direction: column; }
                .apply-btn, .apply-disabled { width: 100%; box-sizing: border-box; }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
def main() -> None:
    initialize_state()
    inject_css()

    with st.sidebar:
        st.markdown(
            dedent(
                """
                <div class="sidebar-brand">✨ AI-Powered Job Discovery</div>
                <div class="sidebar-copy">
                    Premium job search powered by live listings and a LangChain agent.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### Search settings")
        location = st.text_input("Preferred location", value="India", max_chars=80)
        date_label = st.selectbox("Posted", options=list(DATE_OPTIONS), index=0)
        remote_only = st.checkbox("Remote jobs only", value=False)
        limit = st.slider("Results", min_value=5, max_value=30, value=10, step=1)

        st.divider()
        clear_clicked = st.button("Clear results", use_container_width=True)

    if clear_clicked:
        st.session_state.jobs = []
        st.session_state.last_search = None
        st.session_state.search_error = None
        st.rerun()

    st.markdown(
        dedent(
            """
            <section class="hero">
                <div class="eyebrow">✦ CONSOLEFLARE</div>
                <h1>
                    Find work that feels made<br>
                    <span class="gradient-text">for you.</span>
                </h1>
                <p>
                    Search fresh roles across India with intelligent matching and a
                    resilient live-jobs API — presented in a clean, application-ready workspace.
                </p>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("job_search_form", clear_on_submit=False):
            search_col, button_col = st.columns([4.3, 1.3], vertical_alignment="bottom")

            with search_col:
                role = st.text_input(
                    "Role, skills, or keywords",
                    value="Data Analyst",
                    placeholder="e.g. Python Developer, Data Analyst, GenAI Engineer",
                    max_chars=80,
                )

            with button_col:
                submitted = st.form_submit_button(
                    "Search jobs ✦",
                    use_container_width=True,
                    type="primary",
                )

    if submitted:
        errors = validate(role, location, limit)

        if errors:
            st.session_state.search_error = " ".join(errors)
        else:
            try:
                configure_credentials()

                with st.spinner("Searching live job openings..."):
                    payload = run_agent_search(
                        role=role.strip(),
                        location=location.strip(),
                        date_posted=DATE_OPTIONS[date_label],
                        remote_only=remote_only,
                        limit=limit,
                    )

                jobs = payload.get("data", [])
                st.session_state.jobs = [
                    job for job in jobs if isinstance(job, dict)
                ][:limit]
                st.session_state.last_search = {
                    "role": role.strip(),
                    "location": location.strip(),
                    "date": date_label,
                    "remote": remote_only,
                }
                st.session_state.search_error = None

            except (AppConfigurationError, JobSearchError) as exc:
                st.session_state.search_error = str(exc)
            except Exception:
                st.session_state.search_error = (
                    "Something unexpected interrupted the search. Please try again."
                )

    if st.session_state.search_error:
        st.error(st.session_state.search_error)

    jobs: list[dict[str, Any]] = st.session_state.jobs
    search = st.session_state.last_search

    if search:
        st.markdown('<div class="section-heading">AI overview</div>', unsafe_allow_html=True)
        overview = generate_overview(jobs, search["role"], search["location"])
        st.markdown(
            f'<div class="overview-box">{html.escape(overview)}</div>',
            unsafe_allow_html=True,
        )

        matching_roles = len(jobs)
        companies = len({job.get("company") for job in jobs if job.get("company")})
        remote_roles = sum(1 for job in jobs if job.get("remote"))

        metric_cols = st.columns(3)
        metrics = (
            ("MATCHING ROLES", matching_roles),
            ("COMPANIES", companies),
            ("REMOTE ROLES", remote_roles),
        )

        for column, (label, value) in zip(metric_cols, metrics):
            with column:
                st.markdown(
                    dedent(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-heading">Latest opportunities</div>', unsafe_allow_html=True)

        if jobs:
            for job in jobs:
                render_job_card(job)
        else:
            st.markdown(
                dedent(
                    """
                    <div class="empty-state">
                        <strong>No matching jobs were returned.</strong><br>
                        Try a broader location, wider date range, or disable remote-only filtering.
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            dedent(
                """
                <div class="empty-state">
                    <strong>Your job matches will appear here.</strong><br>
                    Enter a role, choose filters from the sidebar, and start searching.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">Live listings may change or expire on the publisher website.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()






