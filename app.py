# ============================================================
# BC Crime Explorer: Major Crimes in British Columbia
# ============================================================
# requirements.txt:
#   streamlit>=1.28.0
#   pandas>=2.0.0
#   plotly>=5.17.0
#   openpyxl>=3.1.0
#   numpy>=1.24.0
#
# Run: streamlit run app.py
#
# Data expected in ./datasets/ folder:
#   - crimedata_csv_all_years.csv  (Vancouver 2003–2021)
#   - appendix_f_-_crime_statistics_in_bc_2023.xlsx  (BC 2022–2023)
#
# To extend: add more CSV/XLSX files to ./datasets/ or upload via sidebar.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, glob, warnings, io
from outlier import run_ensemble, run_multivariate, STRICTNESS_PRESETS, SKLEARN_AVAILABLE
warnings.filterwarnings("ignore")

# ── Page Config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="BC Crime Explorer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "BC Crime Explorer — Interactive crime analytics for British Columbia."},
)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & REFERENCE DATA
# ═══════════════════════════════════════════════════════════════

CRIME_CATEGORY_MAP = {
    "Homicide": "Homicide",
    "Offence Against a Person": "Violent",
    "Vehicle Collision or Pedestrian Struck (with Fatality)": "Violent",
    "Vehicle Collision or Pedestrian Struck (with Injury)": "Traffic",
    "Break and Enter Commercial": "Property",
    "Break and Enter Residential/Other": "Property",
    "Theft from Vehicle": "Property",
    "Theft of Vehicle": "Property",
    "Theft of Bicycle": "Property",
    "Other Theft": "Property",
    "Mischief": "Property",
}

CSI_WEIGHTS = {
    "Homicide": 50.0,
    "Offence Against a Person": 20.0,
    "Vehicle Collision or Pedestrian Struck (with Fatality)": 15.0,
    "Vehicle Collision or Pedestrian Struck (with Injury)": 5.0,
    "Break and Enter Residential/Other": 4.0,
    "Break and Enter Commercial": 3.0,
    "Theft of Vehicle": 2.5,
    "Theft from Vehicle": 1.5,
    "Other Theft": 1.5,
    "Theft of Bicycle": 1.0,
    "Mischief": 1.0,
}

NEIGHBOURHOOD_REGION_MAP = {
    "Central Business District": "Downtown Core",
    "West End": "Downtown Core",
    "Strathcona": "Downtown Core",
    "Stanley Park": "Downtown Core",
    "South Cambie": "West Side",
    "Fairview": "West Side",
    "Mount Pleasant": "West Side",
    "Oakridge": "West Side",
    "Kerrisdale": "West Side",
    "Shaughnessy": "West Side",
    "Dunbar-Southlands": "West Side",
    "West Point Grey": "West Side",
    "Kitsilano": "West Side",
    "Arbutus Ridge": "West Side",
    "Kensington-Cedar Cottage": "East Side",
    "Grandview-Woodland": "East Side",
    "Hastings-Sunrise": "East Side",
    "Renfrew-Collingwood": "East Side",
    "Riley Park": "South Vancouver",
    "Sunset": "South Vancouver",
    "Victoria-Fraserview": "South Vancouver",
    "Killarney": "South Vancouver",
    "Marpole": "South Vancouver",
    "Musqueam": "South Vancouver",
}

NEIGHBOURHOOD_COORDS = {
    "Central Business District": (49.2820, -123.1171),
    "West End": (49.2844, -123.1344),
    "Strathcona": (49.2790, -123.0891),
    "Stanley Park": (49.3017, -123.1417),
    "Fairview": (49.2700, -123.1250),
    "Mount Pleasant": (49.2630, -123.1010),
    "South Cambie": (49.2450, -123.1190),
    "Kensington-Cedar Cottage": (49.2480, -123.0723),
    "Grandview-Woodland": (49.2730, -123.0695),
    "Hastings-Sunrise": (49.2790, -123.0440),
    "Renfrew-Collingwood": (49.2440, -123.0440),
    "Riley Park": (49.2430, -123.1010),
    "Sunset": (49.2210, -123.0860),
    "Victoria-Fraserview": (49.2200, -123.0505),
    "Killarney": (49.2210, -123.0261),
    "Marpole": (49.2080, -123.1347),
    "Oakridge": (49.2300, -123.1160),
    "Kerrisdale": (49.2308, -123.1594),
    "Shaughnessy": (49.2490, -123.1400),
    "Dunbar-Southlands": (49.2380, -123.1820),
    "West Point Grey": (49.2704, -123.2050),
    "Kitsilano": (49.2680, -123.1620),
    "Arbutus Ridge": (49.2510, -123.1560),
    "Musqueam": (49.2200, -123.1920),
}

CRIME_COLORS = {
    "Property": "#00b4d8",
    "Violent": "#dc143c",
    "Homicide": "#8b0000",
    "Traffic": "#d4af37",
    "Other": "#8892a4",
}

REGION_COLORS = {
    "Downtown Core": "#dc143c",
    "East Side": "#00b4d8",
    "West Side": "#d4af37",
    "South Vancouver": "#9b59b6",
    "Other": "#8892a4",
}

BC_CRIME_TIMELINE = [
    {
        "year": 1982,
        "title": "Clifford Olson Convicted",
        "category": "Homicide",
        "color": "#8b0000",
        "description": "Clifford Olson, BC's most notorious serial killer, pleaded guilty to 11 murders of children and teenagers. His case exposed major gaps in inter-agency information sharing among BC police forces.",
    },
    {
        "year": 1985,
        "title": "Air India Flight 182 Bombing",
        "category": "Terrorism",
        "color": "#dc143c",
        "description": "On June 23, 1985, a bomb destroyed Air India Flight 182, killing all 329 aboard. The device was planted at Vancouver International Airport — Canada's deadliest terrorist attack. The subsequent investigation spanned two decades.",
    },
    {
        "year": 1997,
        "title": "Missing Women Investigation Begins",
        "category": "Homicide",
        "color": "#8b0000",
        "description": "Families of women missing from Vancouver's Downtown Eastside began formally pressing police for a coordinated investigation. Systemic failures would later be documented by the Missing Women Commission of Inquiry.",
    },
    {
        "year": 2002,
        "title": "Robert Pickton Farm Investigation",
        "category": "Homicide",
        "color": "#8b0000",
        "description": "Robert Pickton was arrested in February 2002 after a search of his Port Coquitlam pig farm revealed remains of missing women. He was eventually charged with 27 counts of murder, with evidence of at least 33 victims.",
    },
    {
        "year": 2007,
        "title": "Surrey Six Murders",
        "category": "Gang Crime",
        "color": "#9b59b6",
        "description": "On October 19, 2007, six men were murdered in a Surrey highrise apartment by Red Scorpions gang members — one of the worst gang killings in BC history. The case triggered sweeping anti-gang legislation and task forces.",
    },
    {
        "year": 2007,
        "title": "Pickton Verdict",
        "category": "Homicide",
        "color": "#8b0000",
        "description": "Robert Pickton was found guilty of six counts of second-degree murder. The verdict, and subsequent public inquiry, led to landmark reforms in how police handle missing persons cases, especially for marginalized women.",
    },
    {
        "year": 2009,
        "title": "Lower Mainland Gang Violence Escalates",
        "category": "Gang Crime",
        "color": "#9b59b6",
        "description": "A surge in gang-related shootings across Metro Vancouver claimed dozens of lives in 2009. The BC government established the Combined Forces Special Enforcement Unit (CFSEU) in direct response to organized crime violence.",
    },
    {
        "year": 2014,
        "title": "Fentanyl Enters BC's Drug Supply",
        "category": "Drug Crisis",
        "color": "#d4af37",
        "description": "Illicitly manufactured fentanyl — 50–100× more potent than heroin — began appearing in BC's drug supply in measurable quantities, triggering a catastrophic rise in overdose deaths that would reshape public health policy.",
    },
    {
        "year": 2016,
        "title": "BC Declares Opioid Public Health Emergency",
        "category": "Drug Crisis",
        "color": "#d4af37",
        "description": "On April 14, 2016, BC declared a province-wide public health emergency due to the opioid crisis — the first such declaration in Canadian history. That year, nearly 1,000 people died from illicit drug toxicity in BC.",
    },
    {
        "year": 2020,
        "title": "COVID-19 Reshapes Crime Patterns",
        "category": "Social Impact",
        "color": "#00b4d8",
        "description": "Pandemic lockdowns caused dramatic shifts: street-level property crime declined as public spaces emptied, while domestic violence, cybercrime, and frauds surged. The opioid crisis accelerated amid supply chain disruptions.",
    },
    {
        "year": 2021,
        "title": "Record Overdose Deaths in BC",
        "category": "Drug Crisis",
        "color": "#d4af37",
        "description": "BC recorded over 2,200 overdose deaths in 2021 — the highest annual total in the province's history. This represented a public health crisis eclipsing road deaths, homicides, and suicides combined.",
    },
    {
        "year": 2023,
        "title": "BC Crime Statistics: 2023 Snapshot",
        "category": "Statistics",
        "color": "#00b4d8",
        "description": "BC reported 123 homicides (−21% from 2022), 43,390 assault offences (+4%), a violent crime rate of 15.6 per 1,000 residents, and a Crime Severity Index of 96.89 for violent crime — slightly below baseline.",
    },
]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ═══════════════════════════════════════════════════════════════
# CSS — DARK PREMIUM THEME
# ═══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Root & Background ── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background-color: #0a0e1a;
            background-image:
                radial-gradient(ellipse at 15% 40%, rgba(220,20,60,0.05) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 15%, rgba(0,180,216,0.05) 0%, transparent 55%),
                radial-gradient(ellipse at 50% 90%, rgba(212,175,55,0.03) 0%, transparent 50%);
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: #060a13 !important;
            border-right: 1px solid rgba(212,175,55,0.12) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }

        /* ── Hide Streamlit's fixed top chrome so it can't cover the tab bar ── */
        [data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }

        /* ── Main content area ── */
        .block-container { padding-top: 0.5rem !important; max-width: 100% !important; }

        /* ── Text ── */
        h1, h2, h3, h4 { color: #e8eaf0 !important; }
        p, li { color: #c4cad8; }
        label { color: #8892a4 !important; }

        /* ── Streamlit native metric ── */
        [data-testid="stMetricValue"] { color: #d4af37 !important; font-size: 1.9rem !important; font-weight: 800 !important; }
        [data-testid="stMetricLabel"] { color: #8892a4 !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }
        [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }
        [data-testid="stMetric"] { background: rgba(15,22,40,0.85); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 1rem 1.25rem !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #0a0e1a;
            padding: 0.5rem 0 0 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            position: sticky;
            top: 0;
            z-index: 999;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            color: #8892a4;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 0.55rem 1.1rem;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { background: rgba(212,175,55,0.06); color: #d4af37; }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(212,175,55,0.14), rgba(212,175,55,0.04)) !important;
            border-color: rgba(212,175,55,0.28) !important;
            color: #d4af37 !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 0 0; }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, rgba(220,20,60,0.85), rgba(139,0,0,0.9));
            color: white; border: none; border-radius: 8px;
            font-weight: 600; font-size: 0.85rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(220,20,60,0.25);
        }
        .stButton > button:hover {
            box-shadow: 0 6px 22px rgba(220,20,60,0.45);
            transform: translateY(-1px);
        }
        .stDownloadButton > button {
            background: linear-gradient(135deg, rgba(0,150,180,0.8), rgba(0,100,140,0.9)) !important;
            color: white !important; border: none !important;
            box-shadow: 0 4px 15px rgba(0,180,216,0.25) !important;
        }

        /* ── Inputs & Selects ── */
        .stMultiSelect [data-baseweb="select"] div {
            background: rgba(12,17,32,0.9) !important;
            border-color: rgba(212,175,55,0.18) !important;
            color: #e8eaf0 !important;
        }
        .stSelectbox div[data-baseweb="select"] div {
            background: rgba(12,17,32,0.9) !important;
            border-color: rgba(212,175,55,0.18) !important;
        }
        div[data-testid="stSlider"] .rc-slider-track { background: #dc143c; }
        div[data-testid="stSlider"] .rc-slider-handle { border-color: #dc143c; background: #dc143c; }

        /* ── Dataframe ── */
        .stDataFrame { border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; overflow: hidden; }
        iframe[title="st.iframe"] { border-radius: 10px; }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background: rgba(15,22,40,0.6); border: 1px dashed rgba(212,175,55,0.25);
            border-radius: 12px; padding: 0.5rem;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: rgba(15,22,40,0.7) !important;
            border-radius: 8px !important;
            color: #e8eaf0 !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
        }
        .streamlit-expanderContent { background: rgba(10,14,26,0.5) !important; border-radius: 0 0 8px 8px !important; }

        /* ── Alert / info boxes ── */
        [data-testid="stAlert"] { background: rgba(0,180,216,0.08); border-color: rgba(0,180,216,0.25); border-radius: 10px; }

        /* ── Divider ── */
        hr { border-color: rgba(255,255,255,0.06) !important; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(212,175,55,0.45); }

        /* ── Custom KPI Cards ── */
        .kpi-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
        .kpi-card {
            flex: 1; min-width: 160px;
            background: linear-gradient(145deg, rgba(15,22,40,0.92), rgba(10,14,26,0.97));
            border: 1px solid rgba(255,255,255,0.055);
            border-radius: 16px; padding: 1.35rem 1.5rem;
            position: relative; overflow: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
        }
        .kpi-card::before {
            content: ''; position: absolute;
            top: 0; left: 0; right: 0; height: 2.5px;
        }
        .kpi-card.c-red::before   { background: linear-gradient(90deg, #dc143c 0%, rgba(220,20,60,0) 100%); }
        .kpi-card.c-gold::before  { background: linear-gradient(90deg, #d4af37 0%, rgba(212,175,55,0) 100%); }
        .kpi-card.c-teal::before  { background: linear-gradient(90deg, #00b4d8 0%, rgba(0,180,216,0) 100%); }
        .kpi-card.c-purple::before{ background: linear-gradient(90deg, #9b59b6 0%, rgba(155,89,182,0) 100%); }
        .kpi-card:hover { border-color: rgba(212,175,55,0.18); transform: translateY(-3px); box-shadow: 0 14px 40px rgba(0,0,0,0.45); }
        .kpi-icon   { font-size: 1.6rem; margin-bottom: 0.6rem; display: block; }
        .kpi-label  { font-size: 0.72rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; margin-bottom: 0.35rem; }
        .kpi-value  { font-size: 2rem; font-weight: 800; color: #e8eaf0; line-height: 1; margin-bottom: 0.4rem; }
        .kpi-delta  { font-size: 0.8rem; font-weight: 600; color: #8892a4; }
        .kpi-delta.pos { color: #00b4d8; }
        .kpi-delta.neg { color: #dc143c; }

        /* ── Section header ── */
        .section-hdr {
            font-size: 1.05rem; font-weight: 700; color: #e8eaf0;
            border-left: 3px solid #d4af37;
            padding-left: 0.75rem; margin: 1.5rem 0 0.85rem 0;
        }

        /* ── Divider accent ── */
        .divider-accent {
            height: 1px; margin: 1.5rem 0;
            background: linear-gradient(90deg, transparent, rgba(212,175,55,0.3), transparent);
        }

        /* ── Badge ── */
        .badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:20px; font-size:0.72rem; font-weight:600; }
        .badge-red    { background:rgba(220,20,60,0.15);   color:#dc143c; border:1px solid rgba(220,20,60,0.25); }
        .badge-gold   { background:rgba(212,175,55,0.15);  color:#d4af37; border:1px solid rgba(212,175,55,0.25); }
        .badge-teal   { background:rgba(0,180,216,0.15);   color:#00b4d8; border:1px solid rgba(0,180,216,0.25); }
        .badge-purple { background:rgba(155,89,182,0.15);  color:#9b59b6; border:1px solid rgba(155,89,182,0.25); }

        /* ── Page title ── */
        .page-title {
            font-size: 2.4rem; font-weight: 900; line-height: 1.1;
            background: linear-gradient(135deg, #dc143c 0%, #d4af37 50%, #00b4d8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .page-subtitle { color: #8892a4; font-size: 0.95rem; margin-top: 0.3rem; margin-bottom: 1.2rem; }

        /* ── Timeline ── */
        .tl-container { position: relative; padding: 0.5rem 0 0.5rem 0; }
        .tl-container::before {
            content: ''; position: absolute; left: 78px; top: 0; bottom: 0;
            width: 2px;
            background: linear-gradient(180deg, #dc143c 0%, #d4af37 50%, #00b4d8 100%);
        }
        .tl-item { display: flex; gap: 1.2rem; margin-bottom: 1.25rem; align-items: flex-start; }
        .tl-year  { min-width: 68px; text-align: right; color: #d4af37; font-weight: 800; font-size: 1rem; padding-top: 0.8rem; }
        .tl-dot   { flex-shrink: 0; width: 12px; height: 12px; border-radius: 50%; margin-top: 0.95rem; box-shadow: 0 0 10px currentColor; }
        .tl-card  {
            flex: 1; background: rgba(15,22,40,0.78); border: 1px solid rgba(255,255,255,0.055);
            border-radius: 12px; padding: 0.9rem 1.1rem;
            transition: all 0.25s ease;
        }
        .tl-card:hover { border-color: rgba(212,175,55,0.2); box-shadow: 0 6px 24px rgba(0,0,0,0.35); }
        .tl-cat   { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem; }
        .tl-title { font-weight: 700; color: #e8eaf0; font-size: 0.95rem; margin-bottom: 0.4rem; }
        .tl-desc  { color: #8892a4; font-size: 0.82rem; line-height: 1.65; }

        /* ── Insight cards ── */
        .insight-card {
            background: rgba(15,22,40,0.75); border: 1px solid rgba(212,175,55,0.12);
            border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
            display: flex; gap: 0.75rem; align-items: flex-start;
        }
        .insight-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 0.1rem; }
        .insight-text { color: #c4cad8; font-size: 0.9rem; line-height: 1.65; }
        .insight-text strong { color: #d4af37; }

        /* ── Sidebar logo ── */
        .sidebar-logo { text-align:center; padding:1.2rem 0 1rem 0; }
        .sidebar-logo .logo-icon { font-size: 2.8rem; }
        .sidebar-logo .logo-title { color:#d4af37; font-weight:800; font-size:1rem; letter-spacing:0.06em; margin-top:0.3rem; }
        .sidebar-logo .logo-sub   { color:#8892a4; font-size:0.7rem; margin-top:0.15rem; }

        /* ── Filter count badge ── */
        .filter-count {
            background: rgba(212,175,55,0.1); border: 1px solid rgba(212,175,55,0.22);
            border-radius: 10px; padding: 0.7rem 1rem; text-align: center; margin-top: 0.5rem;
        }
        .filter-count .fc-num { color: #d4af37; font-size: 1.5rem; font-weight: 800; }
        .filter-count .fc-lbl { color: #8892a4; font-size: 0.72rem; margin-top: 0.1rem; }

        /* ── About data source cards ── */
        .source-card {
            background: rgba(15,22,40,0.75); border: 1px solid rgba(255,255,255,0.055);
            border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.6rem;
        }
        .source-card h4 { color: #d4af37; font-size: 0.9rem; margin-bottom: 0.35rem; }
        .source-card p  { color: #8892a4; font-size: 0.82rem; line-height: 1.6; margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading crime data…")
def load_crime_data(data_dir: str = "datasets"):
    """Scan data_dir for CSV/XLSX files and build the unified df_crime DataFrame."""
    dfs, loaded_files = [], []
    csv_files  = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if {"TYPE", "YEAR", "MONTH", "NEIGHBOURHOOD"}.issubset(df.columns):
                df = _process_vancouver_csv(df, fname)
                dfs.append(df)
                loaded_files.append({"File": fname, "Format": "CSV", "Rows": f"{len(df):,}", "Status": "✅ Loaded"})
            else:
                loaded_files.append({"File": fname, "Format": "CSV", "Rows": f"{len(df):,}", "Status": "⚠️ Unknown schema"})
        except Exception as e:
            loaded_files.append({"File": fname, "Format": "CSV", "Rows": "—", "Status": f"❌ {e}"})

    # XLSX files — handled separately for BC-wide stats
    for fpath in xlsx_files:
        fname = os.path.basename(fpath)
        loaded_files.append({"File": fname, "Format": "XLSX", "Rows": "—", "Status": "✅ Loaded (BC stats)"})

    df_crime = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df_crime, loaded_files


def _process_vancouver_csv(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "TYPE": "Crime_Type", "YEAR": "Year", "MONTH": "Month",
        "DAY": "Day", "HOUR": "Hour", "MINUTE": "Minute",
        "HUNDRED_BLOCK": "Block", "NEIGHBOURHOOD": "Neighbourhood",
        "X": "UTM_X", "Y": "UTM_Y",
    })

    # Coerce numeric
    for col in ["Year", "Month", "Day", "Hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing year or clearly invalid years
    df = df.dropna(subset=["Year"])
    df = df[(df["Year"] >= 1900) & (df["Year"] <= 2100)]
    df["Year"]  = df["Year"].astype(int)
    df["Month"] = df["Month"].clip(1, 12).fillna(1).astype(int)
    df["Day"]   = df["Day"].clip(1, 31).fillna(1).astype(int)
    df["Hour"]  = df["Hour"].fillna(0).astype(int)

    df["Neighbourhood"] = df["Neighbourhood"].fillna("Unknown")
    df["Crime_Type"]    = df["Crime_Type"].fillna("Other")

    # Derived columns
    df["Crime_Category"] = df["Crime_Type"].map(CRIME_CATEGORY_MAP).fillna("Other")
    df["Region"]         = df["Neighbourhood"].map(NEIGHBOURHOOD_REGION_MAP).fillna("Other")
    df["Decade"]         = (df["Year"] // 10 * 10).astype(str) + "s"
    df["CSI_Weight"]     = df["Crime_Type"].map(CSI_WEIGHTS).fillna(1.0)

    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Fall",   10: "Fall",  11: "Fall",
    }).fillna("Unknown")

    # Approximate day of week
    try:
        df["Date"] = pd.to_datetime(
            df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"}),
            errors="coerce",
        )
        df["Day_of_Week"] = df["Date"].dt.day_name().fillna("Unknown")
    except Exception:
        df["Day_of_Week"] = "Unknown"

    # Map neighbourhood centroids for mapping
    df["Lat"] = df["Neighbourhood"].map(lambda x: NEIGHBOURHOOD_COORDS.get(x, (49.262, -123.13))[0])
    df["Lon"] = df["Neighbourhood"].map(lambda x: NEIGHBOURHOOD_COORDS.get(x, (49.262, -123.13))[1])

    df["Source_File"] = source_file
    return df


@st.cache_data(show_spinner=False)
def load_bc_excel_stats(data_dir: str = "datasets"):
    """Parse Table 1 (offences) and Table 5 (CSI) from the BC stats XLSX."""
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not xlsx_files:
        return None, {}

    fpath = xlsx_files[0]
    try:
        # ── Table 1 ─────────────────────────────────────────
        raw1 = pd.read_excel(fpath, sheet_name="Table 1", header=None)
        rows = []
        for i in range(4, len(raw1)):
            r = raw1.iloc[i]
            cat = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
            if not cat or cat.lower().startswith(("note", "source", "nan", "table")):
                continue
            try:
                rows.append({
                    "Category":      cat,
                    "Count_2022":    _safe_num(r.iloc[1]),
                    "Count_2023":    _safe_num(r.iloc[2]),
                    "Pct_Change":    _safe_num(r.iloc[3]),
                    "Rate_2022":     _safe_num(r.iloc[4]),
                    "Rate_2023":     _safe_num(r.iloc[5]),
                    "Cleared_2022":  _safe_num(r.iloc[7]),
                    "Cleared_2023":  _safe_num(r.iloc[8]),
                })
            except Exception:
                pass
        df_t1 = pd.DataFrame(rows).dropna(subset=["Count_2023"])

        # ── Tables 3-5: CSI & Clearance rates ───────────────
        raw35 = pd.read_excel(fpath, sheet_name="Tables 3 to 5", header=None)
        kv = {}
        for i in range(len(raw35)):
            r = raw35.iloc[i]
            lbl = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
            if "Violent crime severity" in lbl and "Youth" not in lbl:
                kv["violent_csi_2022"], kv["violent_csi_2023"] = _safe_num(r.iloc[1]), _safe_num(r.iloc[2])
            elif "Non-violent crime severity" in lbl and "Youth" not in lbl:
                kv["nonviol_csi_2022"], kv["nonviol_csi_2023"] = _safe_num(r.iloc[1]), _safe_num(r.iloc[2])
            elif "Violent crime rate" in lbl:
                kv["violent_rate_2022"], kv["violent_rate_2023"] = _safe_num(r.iloc[1]), _safe_num(r.iloc[2])
            elif "Property crime rate" in lbl:
                kv["property_rate_2022"], kv["property_rate_2023"] = _safe_num(r.iloc[1]), _safe_num(r.iloc[2])
            elif "Overall weighted clearance rate" in lbl:
                kv["clearance_2022"], kv["clearance_2023"] = _safe_num(r.iloc[1]), _safe_num(r.iloc[2])

        return df_t1, kv
    except Exception as e:
        return None, {}


def _safe_num(val):
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════

def _chart_layout(fig, title="", height=400, legend=True):
    """Apply the standard dark theme layout to any Plotly figure."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="Inter, sans-serif", size=12),
        title=dict(text=title, font=dict(size=14, color="#e8eaf0"), x=0.01, xanchor="left", pad=dict(b=10)),
        height=height,
        margin=dict(l=50, r=20, t=52 if title else 20, b=45),
        showlegend=legend,
        legend=dict(
            bgcolor="rgba(10,14,26,0.75)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="#8892a4", size=11),
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)", color="#8892a4", linecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)", color="#8892a4", linecolor="rgba(255,255,255,0.08)"),
    )
    return fig


def _plotly(fig, title="", height=400, legend=True, key=None):
    _chart_layout(fig, title, height, legend)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            """<div class="sidebar-logo">
                <div class="logo-icon">⚖️</div>
                <div class="logo-title">BC CRIME EXPLORER</div>
                <div class="logo-sub">Interactive Crime Analytics Dashboard</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Time Range ──────────────────────────────────────
        st.markdown("**🗓 Time Range**")
        y_min, y_max = int(df["Year"].min()), int(df["Year"].max())
        year_range = st.slider("Year", y_min, y_max, (y_min, y_max), label_visibility="collapsed", key="yr")

        # ── Crime Category ───────────────────────────────────
        st.markdown("**🔍 Crime Category**")
        all_cats = sorted(df["Crime_Category"].unique())
        sel_cats = st.multiselect("Category", all_cats, default=all_cats, label_visibility="collapsed", key="cat")

        # ── Region ──────────────────────────────────────────
        st.markdown("**📍 Region**")
        all_regions = sorted(df["Region"].unique())
        sel_regions = st.multiselect("Region", all_regions, default=all_regions, label_visibility="collapsed", key="reg")

        # ── Neighbourhood ───────────────────────────────────
        st.markdown("**🏘 Neighbourhood**")
        all_nb = sorted(df["Neighbourhood"].unique())
        sel_nb = st.multiselect("Neighbourhood", all_nb, default=[], placeholder="All (leave blank)", label_visibility="collapsed", key="nb")

        # ── CSI Severity Threshold ───────────────────────────
        st.markdown("**⚖️ Crime Severity Weight**")
        csi_min = float(df["CSI_Weight"].min())
        csi_max = float(df["CSI_Weight"].max())
        csi_range = st.slider(
            "CSI Weight", csi_min, csi_max, (csi_min, csi_max),
            label_visibility="collapsed", key="csi",
        )

        # ── Display Options ──────────────────────────────────
        st.divider()
        st.markdown("**⚙️ Display Options**")
        top_n = st.slider("Top N Neighbourhoods", 5, 24, 10, key="topn")
        show_pct = st.checkbox("Show percentages on charts", value=False, key="pct")

        # ── File Upload ──────────────────────────────────────
        st.divider()
        st.markdown("**📂 Add More Data**")
        uploaded = st.file_uploader(
            "Upload CSV or XLSX", type=["csv", "xlsx"],
            label_visibility="collapsed", key="upload",
        )

        # ── Apply filters ────────────────────────────────────
        mask = (
            (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
            & (df["Crime_Category"].isin(sel_cats))
            & (df["Region"].isin(sel_regions))
            & (df["CSI_Weight"] >= csi_range[0]) & (df["CSI_Weight"] <= csi_range[1])
        )
        if sel_nb:
            mask &= df["Neighbourhood"].isin(sel_nb)
        df_f = df[mask].copy()

        n = len(df_f)
        pct = n / len(df) * 100 if len(df) > 0 else 0
        st.markdown(
            f"""<div class="filter-count">
                <div class="fc-num">{n:,}</div>
                <div class="fc-lbl">Incidents matching filters ({pct:.1f}% of dataset)</div>
            </div>""",
            unsafe_allow_html=True,
        )

        return df_f, top_n, show_pct, uploaded


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════

def render_overview(df: pd.DataFrame, df_bc_t1, bc_kv: dict):
    # Page title
    st.markdown(
        """<div class="page-title">BC Crime Explorer</div>
        <div class="page-subtitle">
            Comprehensive crime analytics for Vancouver &amp; British Columbia &mdash;
            covering 2003–2023 from open police records and provincial statistics.
        </div>""",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No data matches the current filters. Try widening your selection in the sidebar.")
        return

    # ── KPI cards ──────────────────────────────────────────
    total  = len(df)
    annual = df.groupby("Year").size()
    peak_y = int(annual.idxmax()); peak_n = int(annual.max())
    low_y  = int(annual.idxmin()); low_n  = int(annual.min())
    avg_yr = total / df["Year"].nunique()
    top_crime = df["Crime_Type"].value_counts().index[0]
    top_pct   = df["Crime_Type"].value_counts().iloc[0] / total * 100
    yr_range  = f"{df['Year'].min()}–{df['Year'].max()}"

    st.markdown(
        f"""<div class="kpi-row">
            <div class="kpi-card c-red">
                <span class="kpi-icon">🔴</span>
                <div class="kpi-label">Total Incidents</div>
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-delta">{avg_yr:,.0f} avg per year</div>
            </div>
            <div class="kpi-card c-gold">
                <span class="kpi-icon">📅</span>
                <div class="kpi-label">Peak Year</div>
                <div class="kpi-value">{peak_y}</div>
                <div class="kpi-delta neg">{peak_n:,} incidents</div>
            </div>
            <div class="kpi-card c-teal">
                <span class="kpi-icon">📉</span>
                <div class="kpi-label">Lowest Year</div>
                <div class="kpi-value">{low_y}</div>
                <div class="kpi-delta pos">{low_n:,} incidents</div>
            </div>
            <div class="kpi-card c-purple">
                <span class="kpi-icon">🔎</span>
                <div class="kpi-label">Most Common Crime</div>
                <div class="kpi-value" style="font-size:1rem;padding-top:0.4rem">{top_crime}</div>
                <div class="kpi-delta">{top_pct:.1f}% of total</div>
            </div>
            <div class="kpi-card c-gold">
                <span class="kpi-icon">📊</span>
                <div class="kpi-label">Data Coverage</div>
                <div class="kpi-value" style="font-size:1.3rem">{yr_range}</div>
                <div class="kpi-delta">{df['Year'].nunique()} years | {df['Neighbourhood'].nunique()} areas</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── BC-Wide stats row ───────────────────────────────────
    if bc_kv:
        st.markdown('<div class="section-hdr">BC-Wide Statistics (2023 vs 2022)</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        bc_stats = [
            ("Violent CSI",    bc_kv.get("violent_csi_2023", "—"),    bc_kv.get("violent_csi_2022")),
            ("Non-Violent CSI",bc_kv.get("nonviol_csi_2023", "—"),   bc_kv.get("nonviol_csi_2022")),
            ("Violent Rate/1k",bc_kv.get("violent_rate_2023", "—"),   bc_kv.get("violent_rate_2022")),
            ("Property Rate/1k",bc_kv.get("property_rate_2023","—"),  bc_kv.get("property_rate_2022")),
            ("Clearance Rate", f"{bc_kv.get('clearance_2023',0)*100:.1f}%" if bc_kv.get('clearance_2023') else "—", None),
        ]
        for col, (lbl, val, prev) in zip(cols, bc_stats):
            delta = None
            if prev and isinstance(val, float):
                delta = f"{(val - prev) / prev * 100:+.1f}%"
            col.metric(lbl, f"{val:.2f}" if isinstance(val, float) else val, delta)

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    # ── Charts row ──────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        cat_vc = df["Crime_Category"].value_counts()
        fig = go.Figure(go.Pie(
            labels=cat_vc.index,
            values=cat_vc.values,
            hole=0.58,
            marker_colors=[CRIME_COLORS.get(c, "#8892a4") for c in cat_vc.index],
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>%{value:,} incidents<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(
            annotations=[dict(text="Category", x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#e8eaf0")],
        )
        _plotly(fig, "Crime by Category", 360)

    with col2:
        ann = df.groupby(["Year", "Crime_Category"]).size().reset_index(name="Count")
        fig = px.area(
            ann, x="Year", y="Count", color="Crime_Category",
            color_discrete_map=CRIME_COLORS,
            template="plotly_dark",
            hover_data={"Crime_Category": True},
        )
        fig.update_traces(line_width=0.8)
        _plotly(fig, "Annual Incident Trend by Category", 360)

    # ── Top crimes table ────────────────────────────────────
    st.markdown('<div class="section-hdr">Top Crime Types (Filtered Period)</div>', unsafe_allow_html=True)
    top_crimes = (
        df.groupby("Crime_Type").agg(
            Incidents=("Crime_Type", "count"),
            CSI_Weight=("CSI_Weight", "first"),
        )
        .sort_values("Incidents", ascending=False)
        .reset_index()
    )
    top_crimes["Share (%)"] = (top_crimes["Incidents"] / top_crimes["Incidents"].sum() * 100).round(1)
    st.dataframe(
        top_crimes.rename(columns={"Crime_Type": "Crime Type"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Incidents": st.column_config.ProgressColumn("Incidents", format="%d", min_value=0, max_value=int(top_crimes["Incidents"].max())),
            "Share (%)": st.column_config.ProgressColumn("Share (%)", format="%.1f%%", min_value=0, max_value=100),
            "CSI_Weight": st.column_config.NumberColumn("CSI Weight", format="%.1f"),
        },
    )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — TIME TRENDS
# ═══════════════════════════════════════════════════════════════

def render_time_trends(df: pd.DataFrame):
    if df.empty:
        st.info("No data for current filters.")
        return

    st.markdown('<div class="section-hdr">Annual Crime Trends</div>', unsafe_allow_html=True)

    mode = st.radio("View mode", ["By Category", "By Crime Type", "Overall"], horizontal=True, key="tt_mode")

    if mode == "By Category":
        ann = df.groupby(["Year", "Crime_Category"]).size().reset_index(name="Count")
        fig = px.line(ann, x="Year", y="Count", color="Crime_Category",
                      color_discrete_map=CRIME_COLORS, markers=True, template="plotly_dark",
                      hover_data={"Crime_Category": True})
    elif mode == "By Crime Type":
        ann = df.groupby(["Year", "Crime_Type"]).size().reset_index(name="Count")
        fig = px.line(ann, x="Year", y="Count", color="Crime_Type", markers=True, template="plotly_dark")
    else:
        ann = df.groupby("Year").size().reset_index(name="Count")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ann["Year"], y=ann["Count"], mode="lines+markers",
            fill="tozeroy", line=dict(color="#dc143c", width=2.5),
            fillcolor="rgba(220,20,60,0.12)",
            hovertemplate="<b>%{x}</b><br>%{y:,} incidents<extra></extra>",
        ))
        # Add rolling avg
        ann["Rolling"] = ann["Count"].rolling(3, center=True).mean()
        fig.add_trace(go.Scatter(
            x=ann["Year"], y=ann["Rolling"], mode="lines",
            line=dict(color="#d4af37", width=2, dash="dot"),
            name="3-Year Rolling Avg",
        ))

    _plotly(fig, "Annual Incidents Over Time", 420)

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    # ── Monthly heatmap ─────────────────────────────────────
    st.markdown('<div class="section-hdr">Monthly Pattern Heatmap (Year × Month)</div>', unsafe_allow_html=True)
    heat = df.groupby(["Year", "Month"]).size().unstack(fill_value=0)
    heat.columns = [MONTH_NAMES[c - 1] for c in heat.columns]
    fig = px.imshow(
        heat, color_continuous_scale="Reds", aspect="auto",
        labels=dict(x="Month", y="Year", color="Incidents"),
        template="plotly_dark",
    )
    fig.update_xaxes(side="bottom")
    _plotly(fig, "", 380)

    # ── Hour & Day of week ──────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Crimes by Hour of Day</div>', unsafe_allow_html=True)
        hr = df.groupby("Hour").size().reset_index(name="Count")
        fig = px.bar(hr, x="Hour", y="Count", template="plotly_dark",
                     color="Count", color_continuous_scale="Reds",
                     hover_data={"Hour": True})
        fig.update_coloraxes(showscale=False)
        _plotly(fig, "", 340)

    with col2:
        st.markdown('<div class="section-hdr">Crimes by Day of Week</div>', unsafe_allow_html=True)
        dow = df.groupby("Day_of_Week").size().reset_index(name="Count")
        dow["Day_of_Week"] = pd.Categorical(dow["Day_of_Week"], categories=DAY_ORDER, ordered=True)
        dow = dow.sort_values("Day_of_Week")
        fig = px.bar(dow, x="Day_of_Week", y="Count", template="plotly_dark",
                     color="Count", color_continuous_scale="Blues")
        fig.update_coloraxes(showscale=False)
        _plotly(fig, "", 340)

    # ── Season breakdown ────────────────────────────────────
    st.markdown('<div class="section-hdr">Crime by Season</div>', unsafe_allow_html=True)
    seas = df.groupby(["Season", "Crime_Category"]).size().reset_index(name="Count")
    fig = px.bar(seas, x="Season", y="Count", color="Crime_Category",
                 color_discrete_map=CRIME_COLORS, barmode="group", template="plotly_dark",
                 category_orders={"Season": ["Spring", "Summer", "Fall", "Winter"]})
    _plotly(fig, "", 360)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC VIEW
# ═══════════════════════════════════════════════════════════════

def render_geographic(df: pd.DataFrame, top_n: int):
    if df.empty:
        st.info("No data for current filters.")
        return

    st.markdown('<div class="section-hdr">Interactive Crime Map — Vancouver Neighbourhoods</div>', unsafe_allow_html=True)
    st.caption("Bubble size = incident count · Colour = region · Use the Play ▶ button to animate over years")

    map_mode = st.radio("Map animation", ["All years combined", "Animated by year"], horizontal=True, key="map_mode")

    # Aggregate for map
    df_geo = (
        df.groupby(["Neighbourhood", "Year", "Region", "Lat", "Lon"])
        .agg(Count=("Crime_Type", "count"))
        .reset_index()
    )
    df_geo_all = df.groupby(["Neighbourhood", "Region", "Lat", "Lon"]).agg(Count=("Crime_Type", "count")).reset_index()

    try:
        if map_mode == "All years combined":
            fig = px.scatter_mapbox(
                df_geo_all, lat="Lat", lon="Lon", size="Count",
                color="Region", color_discrete_map=REGION_COLORS,
                hover_name="Neighbourhood",
                hover_data={"Count": True, "Lat": False, "Lon": False},
                size_max=55, zoom=11,
                center={"lat": 49.262, "lon": -123.13},
                mapbox_style="carto-darkmatter",
                opacity=0.85,
                template="plotly_dark",
            )
        else:
            fig = px.scatter_mapbox(
                df_geo, lat="Lat", lon="Lon", size="Count",
                color="Region", color_discrete_map=REGION_COLORS,
                hover_name="Neighbourhood",
                animation_frame="Year",
                size_max=55, zoom=11,
                center={"lat": 49.262, "lon": -123.13},
                mapbox_style="carto-darkmatter",
                opacity=0.85,
                template="plotly_dark",
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=520, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Map could not render ({e}). Showing regional bar chart instead.")

    # ── Regional breakdown bar ───────────────────────────────
    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-hdr">Incidents by Region</div>', unsafe_allow_html=True)
        reg = df.groupby(["Region", "Crime_Category"]).size().reset_index(name="Count")
        fig = px.bar(reg, x="Region", y="Count", color="Crime_Category",
                     color_discrete_map=CRIME_COLORS, template="plotly_dark", barmode="stack")
        _plotly(fig, "", 360)

    with col2:
        st.markdown(f'<div class="section-hdr">Top {top_n} Neighbourhoods by Crime Volume</div>', unsafe_allow_html=True)
        nb_top = df["Neighbourhood"].value_counts().head(top_n).reset_index()
        nb_top.columns = ["Neighbourhood", "Count"]
        fig = px.bar(
            nb_top.sort_values("Count"), x="Count", y="Neighbourhood",
            orientation="h", template="plotly_dark",
            color="Count", color_continuous_scale="Reds",
        )
        fig.update_coloraxes(showscale=False)
        _plotly(fig, "", 360)

    # ── Neighbourhood trend over time ───────────────────────
    st.markdown('<div class="section-hdr">Year-over-Year Crime Trends — Top Neighbourhoods</div>', unsafe_allow_html=True)
    top_nb_list = df["Neighbourhood"].value_counts().head(top_n).index.tolist()
    nb_yr = df[df["Neighbourhood"].isin(top_nb_list)].groupby(["Year", "Neighbourhood"]).size().reset_index(name="Count")
    fig = px.line(nb_yr, x="Year", y="Count", color="Neighbourhood", markers=False, template="plotly_dark")
    fig.update_traces(line_width=1.8)
    _plotly(fig, "", 420)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════

def render_data_explorer(df: pd.DataFrame, show_pct: bool):
    if df.empty:
        st.info("No data for current filters.")
        return

    st.markdown('<div class="section-hdr">Interactive Data Table</div>', unsafe_allow_html=True)

    # Aggregated view toggle
    view = st.radio("View", ["Aggregated Summary", "Raw Incident Records"], horizontal=True, key="de_view")

    if view == "Aggregated Summary":
        group_by = st.selectbox(
            "Group by", ["Neighbourhood", "Crime_Type", "Crime_Category", "Year", "Region", "Season", "Day_of_Week"],
            key="de_grp",
        )
        agg = (
            df.groupby(group_by)
            .agg(
                Incidents=("Crime_Type", "count"),
                Avg_CSI=("CSI_Weight", "mean"),
                Unique_Types=("Crime_Type", "nunique"),
            )
            .sort_values("Incidents", ascending=False)
            .reset_index()
        )
        agg["Share (%)"] = (agg["Incidents"] / agg["Incidents"].sum() * 100).round(2)
        agg["Avg_CSI"] = agg["Avg_CSI"].round(2)

        display_df = agg
        download_df = agg
    else:
        cols_show = ["Year", "Month", "Day", "Hour", "Crime_Type", "Crime_Category", "Neighbourhood", "Region", "Block", "CSI_Weight"]
        cols_show = [c for c in cols_show if c in df.columns]
        display_df = df[cols_show].head(50_000)
        download_df = df[cols_show]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    # Export
    csv_buf = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Filtered Data as CSV",
        data=csv_buf,
        file_name=f"bc_crime_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.caption(f"Showing {min(len(display_df), 50_000):,} of {len(df):,} records matching current filters.")

    # ── Summary stats ────────────────────────────────────────
    st.markdown('<div class="section-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.intersection(["Year", "Month", "Hour", "CSI_Weight"])
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — VISUAL ANALYTICS
# ═══════════════════════════════════════════════════════════════

def render_visual_analytics(df: pd.DataFrame, top_n: int):
    if df.empty:
        st.info("No data for current filters.")
        return

    # ── 1. Neighbourhood × Crime Type Heatmap ───────────────
    st.markdown('<div class="section-hdr">Crime Intensity Heatmap — Neighbourhood × Crime Type</div>', unsafe_allow_html=True)
    heat2 = df.groupby(["Neighbourhood", "Crime_Type"]).size().unstack(fill_value=0)
    fig = px.imshow(
        heat2, color_continuous_scale="Reds", aspect="auto",
        template="plotly_dark",
        labels=dict(x="Crime Type", y="Neighbourhood", color="Incidents"),
    )
    fig.update_xaxes(tickangle=-35)
    _plotly(fig, "", 460, key="hm2")

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    # ── 2. Sunburst — Region → Category → Type ──────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Crime Hierarchy — Sunburst</div>', unsafe_allow_html=True)
        df_sun = df.groupby(["Region", "Crime_Category", "Crime_Type"]).size().reset_index(name="Count")
        fig = px.sunburst(
            df_sun, path=["Region", "Crime_Category", "Crime_Type"],
            values="Count",
            color="Crime_Category",
            color_discrete_map=CRIME_COLORS,
            template="plotly_dark",
        )
        fig.update_traces(textfont_size=12)
        _plotly(fig, "", 460, key="sun")

    with col2:
        st.markdown('<div class="section-hdr">Crime Hierarchy — Treemap</div>', unsafe_allow_html=True)
        fig = px.treemap(
            df_sun, path=["Region", "Crime_Category", "Crime_Type"],
            values="Count",
            color="Crime_Category",
            color_discrete_map=CRIME_COLORS,
            template="plotly_dark",
        )
        _plotly(fig, "", 460, key="tree")

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    # ── 3. 3D Scatter ────────────────────────────────────────
    st.markdown('<div class="section-hdr">3D Analytics — Year × Month × Crime Volume</div>', unsafe_allow_html=True)
    st.caption("Rotate the 3D plot by dragging · Scroll to zoom · Hover for details")

    df_3d = df.groupby(["Year", "Month", "Crime_Category"]).size().reset_index(name="Count")

    fig3d = go.Figure()
    for cat, grp in df_3d.groupby("Crime_Category"):
        fig3d.add_trace(go.Scatter3d(
            x=grp["Year"], y=grp["Month"], z=grp["Count"],
            mode="markers",
            name=cat,
            marker=dict(
                size=np.clip(grp["Count"] / grp["Count"].max() * 12, 3, 14),
                color=CRIME_COLORS.get(cat, "#8892a4"),
                opacity=0.8,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{cat}</b><br>Year: %{{x}}<br>Month: %{{y}}<br>"
                "Incidents: %{z:,}<extra></extra>"
            ),
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title="Year",  backgroundcolor="rgba(10,14,26,0.8)", gridcolor="rgba(255,255,255,0.06)", color="#8892a4"),
            yaxis=dict(title="Month", backgroundcolor="rgba(10,14,26,0.8)", gridcolor="rgba(255,255,255,0.06)", color="#8892a4",
                       tickvals=list(range(1, 13)), ticktext=MONTH_NAMES),
            zaxis=dict(title="Incidents", backgroundcolor="rgba(10,14,26,0.8)", gridcolor="rgba(255,255,255,0.06)", color="#8892a4"),
            bgcolor="rgba(5,8,18,0.95)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#e8eaf0"),
        legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    # ── 4. Animated Bar — Top neighbourhoods by year ─────────
    st.markdown(f'<div class="section-hdr">Top {top_n} Neighbourhoods — Animated by Year</div>', unsafe_allow_html=True)
    st.caption("Press the ▶ Play button to animate crime counts year by year.")

    top_nb = df["Neighbourhood"].value_counts().head(top_n).index.tolist()
    df_race = df[df["Neighbourhood"].isin(top_nb)].groupby(["Year", "Neighbourhood"]).size().reset_index(name="Count")

    fig = px.bar(
        df_race.sort_values(["Year", "Count"], ascending=[True, False]),
        x="Count", y="Neighbourhood",
        animation_frame="Year",
        orientation="h",
        color="Count", color_continuous_scale="Reds",
        range_x=[0, df_race["Count"].max() * 1.08],
        template="plotly_dark",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=420, font=dict(color="#e8eaf0"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 5. BC-wide bar (Excel table 1) ───────────────────────
    # (displayed in Overview; here we show crime category polar chart)
    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Crime Volume by Decade & Category</div>', unsafe_allow_html=True)
    dec = df.groupby(["Decade", "Crime_Category"]).size().reset_index(name="Count")
    fig = px.bar(dec, x="Decade", y="Count", color="Crime_Category",
                 color_discrete_map=CRIME_COLORS, barmode="group", template="plotly_dark")
    _plotly(fig, "", 360, key="dec")


# ═══════════════════════════════════════════════════════════════
# TAB 6 — MAJOR CRIMES TIMELINE
# ═══════════════════════════════════════════════════════════════

def render_timeline():
    st.markdown('<div class="section-hdr">Major BC Crime Events — Historical Timeline</div>', unsafe_allow_html=True)
    st.caption("A curated record of landmark crime events shaping British Columbia's criminal justice landscape.")

    # Decade filter
    decades = sorted({(e["year"] // 10) * 10 for e in BC_CRIME_TIMELINE})
    sel_dec = st.multiselect(
        "Filter by Decade",
        [f"{d}s" for d in decades],
        default=[f"{d}s" for d in decades],
        key="tl_dec",
    )
    sel_dec_nums = {int(d[:-1]) for d in sel_dec}

    # Category color map
    cat_color = {
        "Homicide": "#8b0000",
        "Terrorism": "#dc143c",
        "Gang Crime": "#9b59b6",
        "Drug Crisis": "#d4af37",
        "Social Impact": "#00b4d8",
        "Statistics": "#00b4d8",
    }

    html_parts = ['<div class="tl-container">']
    for ev in BC_CRIME_TIMELINE:
        decade = (ev["year"] // 10) * 10
        if decade not in sel_dec_nums:
            continue
        col = ev.get("color", cat_color.get(ev["category"], "#8892a4"))
        badge_class = {
            "Homicide": "badge-red", "Terrorism": "badge-red",
            "Gang Crime": "badge-purple", "Drug Crisis": "badge-gold",
            "Social Impact": "badge-teal", "Statistics": "badge-teal",
        }.get(ev["category"], "badge-teal")

        html_parts.append(
            f'<div class="tl-item">'
            f'<div class="tl-year">{ev["year"]}</div>'
            f'<div class="tl-dot" style="background:{col}; color:{col};"></div>'
            f'<div class="tl-card">'
            f'<div class="tl-cat"><span class="badge {badge_class}">{ev["category"]}</span></div>'
            f'<div class="tl-title">{ev["title"]}</div>'
            f'<div class="tl-desc">{ev["description"]}</div>'
            f'</div></div>'
        )

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 7 — INSIGHTS
# ═══════════════════════════════════════════════════════════════

def render_insights(df: pd.DataFrame, df_bc_t1, bc_kv: dict):
    if df.empty:
        st.info("No data for current filters.")
        return

    st.markdown('<div class="section-hdr">AI-Style Dynamic Insights — Based on Current Filters</div>', unsafe_allow_html=True)
    st.caption("Observations update automatically as you adjust the sidebar filters.")

    total = len(df)
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    annual = df.groupby("Year").size()
    peak_y, low_y = int(annual.idxmax()), int(annual.idxmin())
    top_crime = df["Crime_Type"].value_counts().index[0]
    top_pct   = df["Crime_Type"].value_counts().iloc[0] / total * 100
    top_nb    = df["Neighbourhood"].value_counts().index[0]
    top_nb_n  = df["Neighbourhood"].value_counts().iloc[0]
    top_hr    = int(df.groupby("Hour").size().idxmax())
    prop_pct  = len(df[df["Crime_Category"] == "Property"]) / total * 100
    violent_n = len(df[df["Crime_Category"] == "Violent"])
    prop_n    = len(df[df["Crime_Category"] == "Property"])
    ratio     = prop_n / max(violent_n, 1)

    if len(annual) >= 3:
        trend_pct = (annual.iloc[-1] - annual.iloc[0]) / max(annual.iloc[0], 1) * 100
        trend_dir = "increased" if trend_pct > 0 else "decreased"
    else:
        trend_pct, trend_dir = 0, "unchanged"

    dow_peak = df.groupby("Day_of_Week").size()
    if not dow_peak.empty:
        dow_peak = dow_peak.reindex([d for d in DAY_ORDER if d in dow_peak.index])
        peak_dow = dow_peak.idxmax() if not dow_peak.empty else "Unknown"
    else:
        peak_dow = "Unknown"

    top_region   = df.groupby("Region").size().idxmax()
    top_region_n = df.groupby("Region").size().max()

    insights = [
        ("📊", f"In the selected period <strong>{yr_min}–{yr_max}</strong>, Vancouver recorded "
               f"<strong>{total:,} crime incidents</strong> spanning {df['Crime_Type'].nunique()} distinct crime types "
               f"across {df['Neighbourhood'].nunique()} neighbourhoods."),
        ("🔴", f"<strong>{top_crime}</strong> is the most prevalent crime, accounting for "
               f"<strong>{top_pct:.1f}%</strong> of all incidents — far ahead of the next closest category."),
        ("📅", f"Crime peaked in <strong>{peak_y}</strong> with {annual[peak_y]:,} incidents, while "
               f"<strong>{low_y}</strong> recorded the fewest at {annual[low_y]:,}. "
               f"Overall, crime has <strong>{trend_dir} by {abs(trend_pct):.1f}%</strong> across the full period."),
        ("📍", f"The <strong>{top_nb}</strong> neighbourhood recorded the highest volume "
               f"with <strong>{top_nb_n:,} incidents</strong>, making it the primary crime hotspot. "
               f"The <strong>{top_region}</strong> region overall accounts for {top_region_n:,} incidents."),
        ("🕐", f"Crimes are most frequent at <strong>{top_hr}:00</strong> ({df.groupby('Hour').size()[top_hr]:,} incidents). "
               f"The <strong>{peak_dow}</strong> is the most crime-prone day of the week."),
        ("🏠", f"Property crime dominates at <strong>{prop_pct:.1f}%</strong> of all incidents. "
               f"For every violent incident, there are approximately <strong>{ratio:.1f} property crimes</strong> — "
               f"consistent with national Canadian trends."),
    ]

    # BC-wide insights from Excel
    if bc_kv:
        vcsi = bc_kv.get("violent_csi_2023")
        nvsi = bc_kv.get("nonviol_csi_2023")
        clr  = bc_kv.get("clearance_2023")
        if vcsi:
            insights.append(("⚖️", f"At the provincial level (BC 2023), the <strong>Violent Crime Severity Index</strong> "
                                    f"stood at <strong>{vcsi:.1f}</strong>, while the Non-Violent CSI reached {nvsi:.1f} "
                                    f"— reflecting growing property and fraud offences province-wide."))
        if clr:
            insights.append(("🔍", f"BC's overall weighted <strong>clearance rate</strong> was just "
                                    f"<strong>{clr*100:.1f}%</strong> in 2023, meaning fewer than 1 in 5 crimes "
                                    f"resulted in a charged suspect — highlighting enforcement capacity challenges."))

    for icon, text in insights:
        st.markdown(
            f'<div class="insight-card"><div class="insight-icon">{icon}</div>'
            f'<div class="insight-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    # Year-over-year delta chart
    st.markdown('<div class="section-hdr">Year-over-Year % Change in Total Incidents</div>', unsafe_allow_html=True)
    yoy = annual.pct_change() * 100
    yoy_df = yoy.reset_index()
    yoy_df.columns = ["Year", "Pct_Change"]
    yoy_df = yoy_df.dropna()
    yoy_df["Color"] = yoy_df["Pct_Change"].apply(lambda x: "#dc143c" if x > 0 else "#00b4d8")
    fig = go.Figure(go.Bar(
        x=yoy_df["Year"], y=yoy_df["Pct_Change"],
        marker_color=yoy_df["Color"],
        hovertemplate="<b>%{x}</b><br>Change: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    _plotly(fig, "", 340, legend=False, key="yoy")


# ═══════════════════════════════════════════════════════════════
# TAB 8 — ABOUT & DATA SOURCES
# ═══════════════════════════════════════════════════════════════

def render_about(df: pd.DataFrame, loaded_files: list, df_xl, bc_kv: dict, uploaded):
    st.markdown('<div class="section-hdr">About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="insight-card">
        <div class="insight-icon">ℹ️</div>
        <div class="insight-text">
            <strong>BC Crime Explorer</strong> is an open-source interactive analytics dashboard built with
            Streamlit and Plotly. It visualises police-reported crime data from Vancouver and BC-wide
            provincial statistics. All data is sourced from publicly available government datasets.
            The dashboard is intended for educational, research, and policy-analysis purposes only.
        </div></div>""",
        unsafe_allow_html=True,
    )

    # Loaded files
    st.markdown('<div class="section-hdr">Loaded Datasets</div>', unsafe_allow_html=True)
    if loaded_files:
        st.dataframe(pd.DataFrame(loaded_files), use_container_width=True, hide_index=True)
    else:
        st.warning("No datasets loaded. Place CSV or XLSX files in the `datasets/` folder.")

    # Data sources
    st.markdown('<div class="section-hdr">Data Sources & Methodology</div>', unsafe_allow_html=True)

    sources_html = """
    <div class="source-card">
        <h4>📂 Vancouver Open Crime Data (crimedata_csv_all_years.csv)</h4>
        <p>Police-reported crime incidents in Vancouver from 2003 to 2021. Contains ~794,000 records
        with crime type, date/time, and neighbourhood. Source: City of Vancouver Open Data Portal.
        <em>Limitations:</em> Vancouver only (not all of BC), no victim demographics, some records
        have missing neighbourhoods (~"Offence Against a Person" geocoding gaps).</p>
    </div>
    <div class="source-card">
        <h4>📊 BC Crime Statistics 2023 (appendix_f_...xlsx)</h4>
        <p>Provincial-level crime statistics published by BC's Ministry of Public Safety, including
        Criminal Code offence counts, rates per 1,000 residents, clearance rates, and Crime Severity
        Index (CSI) for 2022–2023. <em>Limitations:</em> Two-year snapshot only; cannot be directly
        merged with incident-level Vancouver data for multi-year CSI trends.</p>
    </div>
    <div class="source-card">
        <h4>⚖️ Crime Severity Index (CSI) Weights</h4>
        <p>The CSI_Weight column uses simplified weights derived from Statistics Canada's UCR survey
        methodology: Homicide=50, Offence Against Person=20, Trafficking=15, B&E Residential=4,
        B&E Commercial=3, Motor Vehicle Theft=2.5, Theft=1.5, Mischief=1. These are approximations
        for relative severity scoring — not official Statistics Canada CSI values.</p>
    </div>
    <div class="source-card">
        <h4>🗺️ Geographic Coordinates</h4>
        <p>Neighbourhood centroid coordinates are approximate values manually derived from known
        Vancouver geography. The original dataset uses UTM Zone 10N (EPSG:32610) X/Y coordinates;
        these have been replaced with WGS84 lat/lon centroids per neighbourhood for mapping purposes.</p>
    </div>
    """
    st.markdown(sources_html, unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="section-hdr">Known Limitations</div>', unsafe_allow_html=True)
    with st.expander("Click to expand known data limitations"):
        st.markdown("""
        - **Geographic scope**: The incident-level dataset covers Vancouver city only — not Metro Vancouver, the Lower Mainland, or the rest of BC.
        - **Pre-2003 data gap**: No standardised digital records are available for earlier periods in this dataset.
        - **Crime reporting gap**: Only police-reported crimes are included. Unreported crimes (estimated at 2–3× higher for some categories) are not captured.
        - **Classification changes**: Crime type definitions may have shifted over the 2003–2021 period, affecting year-over-year comparability.
        - **Population normalization**: Crime rates per capita are not computed for Vancouver neighbourhoods in this app due to lack of annual neighbourhood population data.
        - **XLSX scope**: The BC-wide Excel statistics only cover 2022–2023 and cannot be merged with the incident-level time series.
        """)

    # Uploaded file preview
    st.markdown('<div class="section-hdr">Upload Additional Dataset</div>', unsafe_allow_html=True)
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            st.success(f"Preview of **{uploaded.name}** ({len(df_up):,} rows × {len(df_up.columns)} columns):")
            st.dataframe(df_up.head(200), use_container_width=True)
            st.info("To permanently add this data, save the file to the `datasets/` folder and restart the app.")
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}")
    else:
        st.info("Use the sidebar uploader to preview any additional CSV or XLSX file.")

    # Tech stack
    st.markdown('<div class="section-hdr">Tech Stack</div>', unsafe_allow_html=True)
    tech_html = """
    <div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:0.5rem;">
        <span class="badge badge-teal">Streamlit</span>
        <span class="badge badge-gold">Pandas</span>
        <span class="badge badge-red">Plotly</span>
        <span class="badge badge-purple">NumPy</span>
        <span class="badge badge-teal">OpenPyXL</span>
        <span class="badge badge-gold">Python 3.12</span>
    </div>
    """
    st.markdown(tech_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB — OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════

VERDICT_COLORS = {"normal": "#3a4358", "possible": "#d4af37", "high": "#dc143c"}
VERDICT_LABELS = {"normal": "Normal", "possible": "Possible", "high": "High-confidence"}

_TIME_DIMS = {"Year", "Year-Month", "Decade", "Hour"}


def _build_series(df: pd.DataFrame, group_by: str, metric: str):
    """Aggregate the filtered incidents into an ordered (labels, values) series."""
    if metric == "Incident count":
        agg_kw = dict(value=("Crime_Type", "count"))
    elif metric == "Total CSI weight":
        agg_kw = dict(value=("CSI_Weight", "sum"))
    else:  # Average CSI weight
        agg_kw = dict(value=("CSI_Weight", "mean"))

    if group_by == "Year-Month":
        g = df.groupby(["Year", "Month"]).agg(**agg_kw).reset_index()
        g = g.sort_values(["Year", "Month"])
        g["label"] = g["Year"].astype(str) + "-" + g["Month"].astype(int).map(lambda m: f"{m:02d}")
    else:
        g = df.groupby(group_by).agg(**agg_kw).reset_index()
        g = g.rename(columns={group_by: "label"})
        if group_by == "Decade":
            g["_ord"] = g["label"].str.rstrip("s").astype(int)
            g = g.sort_values("_ord").drop(columns="_ord")
        elif group_by == "Hour":
            g = g.sort_values("label")
        elif group_by == "Day_of_Week":
            order = {d: i for i, d in enumerate(DAY_ORDER)}
            g["_ord"] = g["label"].map(lambda d: order.get(d, 99))
            g = g.sort_values("_ord").drop(columns="_ord")
        elif group_by in _TIME_DIMS:
            g = g.sort_values("label")
        else:
            g = g.sort_values("value", ascending=False)

    g["label"] = g["label"].astype(str)
    return g[["label", "value"]].reset_index(drop=True)


_MV_FEATURES = ["Incidents", "Total_CSI", "Avg_CSI", "Unique_Crime_Types"]


def _build_feature_table(df: pd.DataFrame, group_by: str):
    """Build a per-group numeric feature matrix for multivariate analysis."""
    if group_by == "Year-Month":
        keys = ["Year", "Month"]
        g = df.groupby(keys).agg(
            Incidents=("Crime_Type", "count"),
            Total_CSI=("CSI_Weight", "sum"),
            Avg_CSI=("CSI_Weight", "mean"),
            Unique_Crime_Types=("Crime_Type", "nunique"),
        ).reset_index().sort_values(keys)
        g["label"] = g["Year"].astype(str) + "-" + g["Month"].astype(int).map(lambda m: f"{m:02d}")
    else:
        g = df.groupby(group_by).agg(
            Incidents=("Crime_Type", "count"),
            Total_CSI=("CSI_Weight", "sum"),
            Avg_CSI=("CSI_Weight", "mean"),
            Unique_Crime_Types=("Crime_Type", "nunique"),
        ).reset_index().rename(columns={group_by: "label"})
    g["label"] = g["label"].astype(str)
    g["Avg_CSI"] = g["Avg_CSI"].round(3)
    return g[["label"] + _MV_FEATURES].reset_index(drop=True)


def _outlier_callout(labels, values, verdict, value_name: str):
    """Render an explicit, human-readable list of the detected outliers."""
    pairs = list(zip(labels, values, verdict))
    highs = [(l, v) for l, v, vd in pairs if vd == "high"]
    poss = [(l, v) for l, v, vd in pairs if vd == "possible"]

    def fmt(items):
        return ", ".join(f"**{l}** ({float(v):,.2f})" for l, v in items)

    st.markdown('<div class="section-hdr">Detected Outliers</div>', unsafe_allow_html=True)
    if not highs and not poss:
        st.success("No anomalies detected at the current settings — every group voted Normal.")
        return
    if highs:
        st.markdown(
            f"🔴 **High-confidence ({len(highs)})** — flagged by ≥2 detectors · "
            f"{value_name}: {fmt(highs)}"
        )
    if poss:
        st.markdown(
            f"🟡 **Possible ({len(poss)})** — flagged by exactly 1 detector · "
            f"{value_name}: {fmt(poss)}"
        )


def render_outlier_detection(df: pd.DataFrame):
    if df.empty:
        st.info("No data for current filters.")
        return

    st.markdown('<div class="section-hdr">Anomaly Detection — Hybrid Voting Ensemble</div>', unsafe_allow_html=True)
    st.caption(
        "Filtered incidents are aggregated by group and screened by four detectors — "
        "Modified Z-Score (MAD), Tukey IQR fences, Isolation Forest, and iterative mean/std "
        "trimming — reconciled by a majority vote into Normal, Possible, and High-confidence "
        "verdicts. Univariate screens one metric; multivariate screens several features at once."
    )

    if not SKLEARN_AVAILABLE:
        st.warning(
            "scikit-learn is not installed, so the Isolation Forest detector is disabled — "
            "results below use a three-detector vote. Run `pip install scikit-learn` to enable it."
        )

    mode = st.radio(
        "Analysis mode",
        ["Univariate (one metric)", "Multivariate (multiple features)"],
        horizontal=True, key="od_mode",
    )
    is_multi = mode.startswith("Multivariate")

    c1, c2 = st.columns(2)
    with c1:
        group_by = st.selectbox(
            "Aggregate by",
            ["Year", "Year-Month", "Decade", "Neighbourhood", "Region",
             "Crime_Type", "Crime_Category", "Season", "Day_of_Week", "Hour"],
            key="od_grp",
        )
    with c2:
        if is_multi:
            st.selectbox("Features", [" + ".join(_MV_FEATURES)], disabled=True, key="od_feat")
        else:
            metric = st.selectbox(
                "Metric", ["Incident count", "Total CSI weight", "Average CSI weight"],
                key="od_metric",
            )

    # ── Tuning controls ──────────────────────────────────────
    t1, t2, t3 = st.columns([1.2, 1, 1])
    with t1:
        strictness_label = st.radio(
            "Strictness (MAD & IQR)",
            ["Lenient", "Balanced", "Strict"], index=1, horizontal=True, key="od_strict",
        )
    with t2:
        contamination = st.slider(
            "Contamination (Isolation Forest)", 0.01, 0.20, 0.10, 0.01, key="od_contam",
        )
    with t3:
        iterations = st.slider(
            "Iterations (iterative trimmer)", 1, 100, 20, 1, key="od_iter",
        )
    strictness = strictness_label.lower()

    if is_multi:
        feat = _build_feature_table(df, group_by)
        if len(feat) < 3:
            st.info(
                f"Only {len(feat)} group(s) for this aggregation — at least 3 are needed. "
                "Widen the filters or choose a finer aggregation."
            )
            return
        matrix = feat[_MV_FEATURES].to_numpy(dtype=float)
        result = run_multivariate(matrix, _MV_FEATURES, strictness, contamination, iterations)
        labels = feat["label"].tolist()
        verdict = result["vote"]["verdict"]
        votes = result["vote"]["count"]
        tallies = result["vote"]["tallies"]
        per_method = result["vote"]["per_method"]

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 High-confidence", tallies["high"])
        m2.metric("🟡 Possible", tallies["possible"])
        m3.metric("⚪ Normal", tallies["normal"])
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("MAD (any feature)", per_method["mad"])
        p2.metric("IQR (any feature)", per_method["iqr"])
        p3.metric("Isolation Forest", per_method["iso"])
        p4.metric("Iterative (any feature)", per_method["itr"])

        _outlier_callout(labels, feat["Incidents"], verdict, "Incidents")

        # PCA / 2-D scatter coloured by verdict
        pca = np.asarray(result["pca"], dtype=float)
        fig = go.Figure()
        for v in ("normal", "possible", "high"):
            idx = [i for i, vv in enumerate(verdict) if vv == v]
            if not idx:
                continue
            fig.add_trace(go.Scatter(
                x=pca[idx, 0], y=pca[idx, 1], mode="markers",
                name=VERDICT_LABELS[v],
                marker=dict(color=VERDICT_COLORS[v], size=10,
                            line=dict(width=1, color="rgba(255,255,255,0.25)")),
                text=[labels[i] for i in idx],
                hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            ))
        _plotly(fig, title=f"Feature-space projection by {group_by} — outliers highlighted",
                height=440, key="od_mv_chart")

        detail = feat.rename(columns={"label": group_by}).copy()
        detail["MAD"] = result["row_flags"]["mad"]
        detail["IQR"] = result["row_flags"]["iqr"]
        detail["IsoForest"] = result["row_flags"]["iso"]
        detail["Iterative"] = result["row_flags"]["itr"]
        detail["Votes"] = votes
        detail["Verdict"] = [VERDICT_LABELS[v] for v in verdict]
        detail = detail.sort_values(["Votes", "Incidents"], ascending=[False, False]).reset_index(drop=True)

        outliers = detail[detail["Verdict"] != "Normal"].reset_index(drop=True)
        st.markdown('<div class="section-hdr">Outlier Entries</div>', unsafe_allow_html=True)
        if outliers.empty:
            st.success("No outlier entries at the current settings — every group voted Normal.")
        else:
            st.dataframe(outliers, use_container_width=True, hide_index=True,
                         height=min(420, 80 + 35 * len(outliers)))

        st.markdown('<div class="section-hdr">Per-Group Results (All Groups)</div>', unsafe_allow_html=True)
        st.dataframe(detail, use_container_width=True, hide_index=True, height=420)
        report = detail
        n_groups = len(feat)
    else:
        series = _build_series(df, group_by, metric)
        if len(series) < 3:
            st.info(
                f"Only {len(series)} group(s) for this aggregation — at least 3 data points are "
                "needed to detect outliers. Widen the filters or choose a finer aggregation."
            )
            return
        values = series["value"].to_numpy(dtype=float)
        result = run_ensemble(values, strictness, contamination, iterations)
        labels = series["label"].tolist()
        verdict = result["vote"]["verdict"]
        votes = result["vote"]["count"]
        tallies = result["vote"]["tallies"]
        per_method = result["vote"]["per_method"]

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 High-confidence", tallies["high"])
        m2.metric("🟡 Possible", tallies["possible"])
        m3.metric("⚪ Normal", tallies["normal"])
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("MAD flags", per_method["mad"])
        p2.metric("IQR flags", per_method["iqr"])
        p3.metric("Isolation Forest", per_method["iso"])
        p4.metric("Iterative (vote-eligible)", per_method["itr"])

        k_flat = result["itr"]["stats"].get("K_flat")
        if k_flat is not None and iterations > k_flat:
            st.caption(
                f"⚠️ The iterative band goes effectively flat at iteration K={k_flat}; "
                f"running {iterations} iterations is past its useful working range."
            )

        _outlier_callout(labels, series["value"], verdict, metric)

        colors = [VERDICT_COLORS[v] for v in verdict]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=series["label"], y=series["value"],
            marker=dict(color=colors, line=dict(width=0)),
            customdata=np.stack([votes, [VERDICT_LABELS[v] for v in verdict]], axis=-1),
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.2f}<br>"
                          "Votes: %{customdata[0]}<br>Verdict: %{customdata[1]}<extra></extra>",
            showlegend=False,
        ))
        for v in ("high", "possible", "normal"):
            fig.add_trace(go.Bar(
                x=[None], y=[None], name=VERDICT_LABELS[v],
                marker=dict(color=VERDICT_COLORS[v]),
            ))
        fig.update_layout(barmode="overlay")
        _plotly(fig, title=f"{metric} by {group_by} — outliers highlighted",
                height=440, key="od_chart")

        detail = pd.DataFrame({
            group_by: series["label"],
            metric: series["value"].round(2),
            "MAD": result["mad"]["flags"],
            "IQR": result["iqr"]["flags"],
            "IsoForest": result["iso"]["flags"],
            "Iterative": result["itr"]["flags"],
            "Votes": votes,
            "Verdict": [VERDICT_LABELS[v] for v in verdict],
        })
        detail = detail.sort_values(["Votes", metric], ascending=[False, False]).reset_index(drop=True)

        outliers = detail[detail["Verdict"] != "Normal"].reset_index(drop=True)
        st.markdown('<div class="section-hdr">Outlier Entries</div>', unsafe_allow_html=True)
        if outliers.empty:
            st.success("No outlier entries at the current settings — every group voted Normal.")
        else:
            st.dataframe(outliers, use_container_width=True, hide_index=True,
                         height=min(420, 80 + 35 * len(outliers)))

        st.markdown('<div class="section-hdr">Per-Group Results (All Groups)</div>', unsafe_allow_html=True)
        st.dataframe(detail, use_container_width=True, hide_index=True, height=420)
        report = detail
        n_groups = len(series)

    csv_buf = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Outlier Report as CSV",
        data=csv_buf,
        file_name=f"bc_crime_outliers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
    st.caption(
        f"{tallies['high']} high-confidence and {tallies['possible']} possible anomalies "
        f"across {n_groups} groups · {mode} · strictness={strictness_label} "
        f"(τ={STRICTNESS_PRESETS[strictness]['tau']}, k={STRICTNESS_PRESETS[strictness]['k']}), "
        f"contamination={contamination}, iterations={iterations}."
    )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Load data ─────────────────────────────────────────────
    df_crime, loaded_files = load_crime_data("datasets")
    df_bc_t1, bc_kv        = load_bc_excel_stats("datasets")

    if df_crime.empty:
        st.error(
            "⚠️ No crime data loaded. Please place the CSV dataset in the `datasets/` folder "
            "and restart the app."
        )
        st.stop()

    # ── Sidebar (returns filtered df + options) ───────────────
    df_f, top_n, show_pct, uploaded = render_sidebar(df_crime)

    # ── Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Overview",
        "📈 Time Trends",
        "🗺 Geographic View",
        "📋 Data Explorer",
        "🔬 Visual Analytics",
        "🚨 Outlier Detection",
        "📅 Timeline",
        "💡 Insights",
        "ℹ️ About",
    ])

    with tabs[0]:
        render_overview(df_f, df_bc_t1, bc_kv)

    with tabs[1]:
        render_time_trends(df_f)

    with tabs[2]:
        render_geographic(df_f, top_n)

    with tabs[3]:
        render_data_explorer(df_f, show_pct)

    with tabs[4]:
        render_visual_analytics(df_f, top_n)

    with tabs[5]:
        render_outlier_detection(df_f)

    with tabs[6]:
        render_timeline()

    with tabs[7]:
        render_insights(df_f, df_bc_t1, bc_kv)

    with tabs[8]:
        render_about(df_crime, loaded_files, df_bc_t1, bc_kv, uploaded)


if __name__ == "__main__":
    main()
