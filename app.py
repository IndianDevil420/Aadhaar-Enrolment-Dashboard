import io
import zipfile
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from difflib import SequenceMatcher

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(
    page_title="Aadhaar Enrolment Dashboard",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: white !important;
        font-weight: 600;
    }
    .stMetric .metric-value {
        color: white !important;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 700;
    }
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
    }
    /* Custom icon styles */
    .icon-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Expected canonical column names
CANONICAL_COLS = {
    "date": ["date", "enrolment_date", "date_of_enrolment", "enrollment_date"],
    "state": ["state", "state_name"],
    "district": ["district", "district_name"],
    "pincode": ["pincode", "pin", "pin_code", "postal_code"],
    "age_0_5": ["0_5", "age_0_5", "age_0to5", "age_0-5", "age_0_5_count"],
    "age_5_17": ["5_17", "age_5_17", "age_5to17", "age_5-17", "age_5_17_count"],
    "age_18_plus": ["18_plus", "age_18_plus", "age_18+", "age_18plus", "age_18_plus_count"],
}

AGE_GROUP_KEYS = ["age_0_5", "age_5_17", "age_18_plus"]
AGE_GROUP_LABELS = {
    "age_0_5": "0–5 years",
    "age_5_17": "5–17 years",
    "age_18_plus": "18+ years",
}

# Modern color palette
COLOR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']

# -----------------------------
# Utility functions
# -----------------------------

def normalize_name(name: str) -> str:
    """Normalize a name by removing extra spaces, special chars and converting to lowercase."""
    if pd.isna(name) or not name:
        return ""
    # Remove all non-alphanumeric chars and convert to lowercase
    normalized = ''.join(c.lower() for c in str(name) if c.isalnum())
    return normalized


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two normalized strings."""
    n1 = normalize_name(s1)
    n2 = normalize_name(s2)
    if not n1 or not n2:
        return 0.0
    return SequenceMatcher(None, n1, n2).ratio()


def build_state_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping of state name variations to their canonical form.
    Uses aggressive string normalization (removes spaces, special chars).
    """
    if "state" not in df.columns:
        return {}
    
    # Get all state names
    all_states = df["state"].dropna().astype(str).str.strip().tolist()
    
    # Count occurrences
    state_counts = Counter(all_states)
    
    # Get unique states
    unique_states = sorted(list(set(all_states)), key=lambda x: state_counts[x], reverse=True)
    
    # Group by normalized name
    normalized_groups = {}
    
    for state in unique_states:
        normalized = normalize_name(state)
        if not normalized:
            continue
            
        if normalized not in normalized_groups:
            normalized_groups[normalized] = []
        normalized_groups[normalized].append(state)
    
    # Build mapping - map all variations to the most frequent one
    state_map = {}
    for normalized, group in normalized_groups.items():
        # Pick the most frequent as canonical
        canonical = max(group, key=lambda x: state_counts[x])
        for state in group:
            state_map[state] = canonical
    
    return state_map


def build_district_mapping(df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """
    Build a mapping of district name variations to their canonical form within each state.
    Uses aggressive string normalization.
    """
    if "state" not in df.columns or "district" not in df.columns:
        return {}
    
    district_map = {}
    
    # Process each state separately
    for state in df["state"].dropna().unique():
        state_data = df[df["state"] == state]
        all_districts = state_data["district"].dropna().astype(str).str.strip().tolist()
        
        # Count occurrences
        district_counts = Counter(all_districts)
        
        # Get unique districts
        unique_districts = list(set(all_districts))
        
        # Group by normalized name
        normalized_groups = {}
        
        for district in unique_districts:
            normalized = normalize_name(district)
            if not normalized:
                continue
                
            if normalized not in normalized_groups:
                normalized_groups[normalized] = []
            normalized_groups[normalized].append(district)
        
        # Build mapping for this state
        for normalized, group in normalized_groups.items():
            canonical = max(group, key=lambda x: district_counts[x])
            for district in group:
                district_map[(state, district)] = canonical
    
    return district_map


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to canonical keys defined in CANONICAL_COLS."""
    cols = {c.lower().strip(): c for c in df.columns}
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})

    rename_map = {}
    for canon, variants in CANONICAL_COLS.items():
        for v in variants:
            if v in df.columns:
                rename_map[v] = canon
                break
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for required in ["date", "state", "district", "pincode"] + AGE_GROUP_KEYS:
        if required not in df.columns:
            df[required] = np.nan

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column robustly."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    return df


def clean_strings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Trim and standardize string columns."""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": np.nan})
            )
    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce age group columns to numeric, fill missing with 0."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].fillna(0).astype(int)
    return df


def load_zip_to_dfs(zip_bytes: bytes) -> List[pd.DataFrame]:
    """Load CSVs from a ZIP file-like bytes object."""
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    dfs.append(df)
    return dfs


def unify_and_merge(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Standardize, clean, and concatenate multiple CSVs logically."""
    cleaned = []
    for df in dfs:
        df = standardize_columns(df)
        df = parse_dates(df)
        df = clean_strings(df, ["state", "district"])
        df["pincode"] = df["pincode"].astype(str).str.strip().replace({"nan": np.nan})
        df = coerce_numeric(df, AGE_GROUP_KEYS)
        cleaned.append(df)

    # Concatenate first
    combined = pd.concat(cleaned, ignore_index=True)
    
    # Build state mapping and apply it
    state_map = build_state_mapping(combined)
    combined["state"] = combined["state"].apply(lambda x: state_map.get(x, x) if pd.notna(x) else x)
    
    # Build district mapping and apply it
    district_map = build_district_mapping(combined)
    combined["district"] = combined.apply(
        lambda row: district_map.get((row["state"], row["district"]), row["district"])
        if pd.notna(row["state"]) and pd.notna(row["district"])
        else row["district"],
        axis=1
    )
    
    essential = ["date", "state", "district"]
    combined = combined.dropna(subset=essential)

    dedup_keys = ["date", "state", "district", "pincode"]
    combined = (
        combined.sort_values("date")
        .drop_duplicates(subset=dedup_keys, keep="last")
        .reset_index(drop=True)
    )

    combined["total_enrolments"] = combined[AGE_GROUP_KEYS].sum(axis=1)
    return combined


@st.cache_data(show_spinner=False)
def prepare_data(zip_bytes: bytes) -> pd.DataFrame:
    """End-to-end data preparation pipeline."""
    dfs = load_zip_to_dfs(zip_bytes)
    if not dfs:
        raise ValueError("No CSV files found in the ZIP.")
    data = unify_and_merge(dfs)

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.to_period("M").astype(str)
    return data


def filter_data(
    df: pd.DataFrame,
    states: List[str],
    districts: List[str],
    pincode: str,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    age_group_choice: str,
) -> pd.DataFrame:
    """Apply all filters to the dataset."""
    f = df.copy()

    if states:
        f = f[f["state"].isin(states)]
    if districts:
        f = f[f["district"].isin(districts)]
    if pincode:
        f = f[f["pincode"] == str(pincode).strip()]

    if date_range and all(date_range):
        start, end = date_range
        f = f[(f["date"] >= start) & (f["date"] <= end)]

    return f


def compute_kpis(df: pd.DataFrame, age_group_choice: str) -> Dict[str, any]:
    """Compute KPI metrics."""
    kpis = {}
    if df.empty:
        return {
            "total": 0,
            "age_group_percent": {},
            "growth_rate": None,
            "highest_region": None,
            "lowest_region": None,
        }

    if age_group_choice == "All":
        total = int(df["total_enrolments"].sum())
    else:
        total = int(df[age_group_choice].sum())
    kpis["total"] = total

    total_all = df["total_enrolments"].sum()
    if total_all > 0:
        kpis["age_group_percent"] = {
            AGE_GROUP_LABELS[k]: round(100 * df[k].sum() / total_all, 2)
            for k in AGE_GROUP_KEYS
        }
    else:
        kpis["age_group_percent"] = {AGE_GROUP_LABELS[k]: 0.0 for k in AGE_GROUP_KEYS}

    monthly = df.groupby("month")["total_enrolments"].sum().sort_index()
    if len(monthly) >= 2:
        last, prev = monthly.iloc[-1], monthly.iloc[-2]
        kpis["growth_rate"] = round(((last - prev) / prev) * 100, 2) if prev != 0 else None
    else:
        yearly = df.groupby("year")["total_enrolments"].sum().sort_index()
        if len(yearly) >= 2:
            last, prev = yearly.iloc[-1], yearly.iloc[-2]
            kpis["growth_rate"] = round(((last - prev) / prev) * 100, 2) if prev != 0 else None
        else:
            kpis["growth_rate"] = None

    by_district = df.groupby(["state", "district"])["total_enrolments"].sum().reset_index()
    if not by_district.empty:
        highest = by_district.sort_values("total_enrolments", ascending=False).iloc[0]
        lowest = by_district.sort_values("total_enrolments", ascending=True).iloc[0]
        kpis["highest_region"] = f"{highest['district']}, {highest['state']} ({int(highest['total_enrolments'])})"
        kpis["lowest_region"] = f"{lowest['district']}, {lowest['state']} ({int(lowest['total_enrolments'])})"
    else:
        kpis["highest_region"] = None
        kpis["lowest_region"] = None

    return kpis


def generate_insights(df: pd.DataFrame) -> List[str]:
    """Rule-based insights."""
    insights = []
    if df.empty:
        return ["No data available for the selected filters."]

    monthly_district = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby(["state", "district", "month"])["total_enrolments"]
        .sum()
        .reset_index()
    )
    medians = monthly_district.groupby(["state", "district"])["total_enrolments"].median().reset_index()
    if not medians.empty:
        threshold = medians["total_enrolments"].quantile(0.1)
        low_regions = medians[medians["total_enrolments"] <= threshold]
        if not low_regions.empty:
            regions_list = ", ".join(
                [f"{r['district']}, {r['state']}" for _, r in low_regions.iterrows()]
            )
            insights.append(f"Consistently low enrolment observed in: {regions_list}.")

    monthly_age = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month")[AGE_GROUP_KEYS]
        .sum()
        .sort_index()
    )
    if len(monthly_age) >= 6:
        recent = monthly_age.iloc[-3:].sum()
        prior = monthly_age.iloc[-6:-3].sum()
        for k in AGE_GROUP_KEYS:
            if prior[k] > 0:
                growth = (recent[k] - prior[k]) / prior[k]
                if growth < 0.05:
                    insights.append(f"Slow growth in {AGE_GROUP_LABELS[k]} over the last quarter (~{round(growth*100,2)}%).")

    mom = monthly_age.sum(axis=1)
    if len(mom) >= 3:
        pct_change = mom.pct_change().dropna()
        spikes = pct_change[pct_change > 0.4]
        drops = pct_change[pct_change < -0.4]
        for m in spikes.index:
            insights.append(f"Spike detected in {m}: +{round(pct_change.loc[m]*100,2)}% month-over-month.")
        for m in drops.index:
            insights.append(f"Drop detected in {m}: {round(pct_change.loc[m]*100,2)}% month-over-month.")

    if not insights:
        insights.append("No notable anomalies or slowdowns detected in the selected period.")
    return insights


def make_time_series(df: pd.DataFrame, age_group_choice: str):
    """Enhanced time-series chart with area fill."""
    if df.empty:
        return go.Figure()

    ts = (
        df.groupby("date")[AGE_GROUP_KEYS]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    
    fig = go.Figure()
    
    for i, col in enumerate(AGE_GROUP_KEYS):
        fig.add_trace(go.Scatter(
            x=ts["date"],
            y=ts[col],
            name=AGE_GROUP_LABELS[col],
            mode='lines',
            line=dict(width=3, color=COLOR_PALETTE[i]),
            fill='tonexty' if i > 0 else 'tozeroy',
            fillcolor=f'rgba({int(COLOR_PALETTE[i][1:3], 16)}, {int(COLOR_PALETTE[i][3:5], 16)}, {int(COLOR_PALETTE[i][5:7], 16)}, 0.3)'
        ))
    
    fig.update_layout(
        title=dict(text="Enrolment Trends Over Time", font=dict(size=20, color="#1f2937")),
        xaxis_title="Date",
        yaxis_title="Enrolments",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        height=450
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    return fig


def make_geo_bars(df: pd.DataFrame, level: str = "state"):
    """Enhanced geographic bar chart with gradient colors."""
    if df.empty:
        return go.Figure()

    if level == "state":
        grp = df.groupby("state")["total_enrolments"].sum().reset_index()
        title = "Total Enrolments by State"
        xcol = "state"
    else:
        grp = df.groupby(["state", "district"])["total_enrolments"].sum().reset_index()
        grp["label"] = grp["district"] + ", " + grp["state"]
        title = "Total Enrolments by District"
        xcol = "label"

    grp = grp.sort_values("total_enrolments", ascending=False).head(30)
    
    fig = go.Figure(data=[
        go.Bar(
            x=grp[xcol],
            y=grp["total_enrolments"],
            marker=dict(
                color=grp["total_enrolments"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Enrolments")
            ),
            text=grp["total_enrolments"],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#1f2937")),
        xaxis_title=level.title(),
        yaxis_title="Enrolments",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        height=500,
        xaxis_tickangle=-45
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    return fig


def make_age_distribution(df: pd.DataFrame, group_by: str = "state"):
    """Enhanced stacked bar chart for age distribution."""
    if df.empty:
        return go.Figure()

    grp = df.groupby(group_by)[AGE_GROUP_KEYS].sum().reset_index()
    grp = grp.sort_values(AGE_GROUP_KEYS[0], ascending=False).head(20)
    
    fig = go.Figure()
    
    for i, col in enumerate(AGE_GROUP_KEYS):
        fig.add_trace(go.Bar(
            name=AGE_GROUP_LABELS[col],
            x=grp[group_by],
            y=grp[col],
            marker_color=COLOR_PALETTE[i],
            text=grp[col],
            texttemplate='%{text:,.0f}',
            textposition='inside'
        ))
    
    fig.update_layout(
        title=dict(text=f"👥 Age Group Distribution by {group_by.title()}", font=dict(size=20, color="#1f2937")),
        xaxis_title=group_by.title(),
        yaxis_title="Enrolments",
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_tickangle=-45
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    return fig


def make_heatmap(df: pd.DataFrame, level: str = "district"):
    """Enhanced heatmap visualization."""
    if df.empty:
        return go.Figure()

    if level == "state":
        grp = df.groupby("state")["total_enrolments"].sum().reset_index()
        grp = grp.sort_values("total_enrolments", ascending=False).head(20)
        z_data = [grp["total_enrolments"].values]
        y_labels = ["Enrolments"]
        x_labels = grp["state"].tolist()
    else:
        grp = df.groupby(["state", "district"])["total_enrolments"].sum().reset_index()
        grp = grp.sort_values("total_enrolments", ascending=False).head(30)
        grp["label"] = grp["district"] + ", " + grp["state"]
        z_data = [grp["total_enrolments"].values]
        y_labels = ["Enrolments"]
        x_labels = grp["label"].tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='Portland',
        showscale=True,
        colorbar=dict(title="Count"),
        text=z_data,
        texttemplate='%{text:,.0f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=dict(text=f"Enrolment Intensity Heatmap ({level.title()})", font=dict(size=20, color="#1f2937")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def region_ranking_table(df: pd.DataFrame, level: str = "district") -> pd.DataFrame:
    """Create a ranking table for regions."""
    if df.empty:
        return pd.DataFrame(columns=["Region", "Total Enrolments", "Rank"])

    if level == "state":
        grp = df.groupby("state")["total_enrolments"].sum().reset_index()
        grp = grp.rename(columns={"state": "Region", "total_enrolments": "Total Enrolments"})
    else:
        grp = df.groupby(["state", "district"])["total_enrolments"].sum().reset_index()
        grp["Region"] = grp["district"] + ", " + grp["state"]
        grp = grp[["Region", "total_enrolments"]].rename(columns={"total_enrolments": "Total Enrolments"})

    grp["Rank"] = grp["Total Enrolments"].rank(method="dense", ascending=False).astype(int)
    grp = grp.sort_values(["Rank", "Total Enrolments"], ascending=[True, False])
    return grp


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Data & Filters")

uploaded_zip = st.sidebar.file_uploader(
    "Upload ZIP containing CSV files",
    type=["zip"],
    help="The ZIP should contain aggregated Aadhaar enrolment CSVs.",
)

if uploaded_zip is None:
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h1 style='color: #667eea;'>Aadhaar Enrolment Dashboard</h1>
        <p style='font-size: 1.2rem; color: #6b7280;'>Upload a ZIP file to begin exploring enrolment data</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Prepare data
try:
    with st.spinner("Loading and processing data..."):
        data = prepare_data(uploaded_zip.getvalue())
        
        # Debug: Show state consolidation info
        with st.sidebar.expander("Debug Info", expanded=False):
            unique_states = data["state"].unique()
            st.write(f"**Total unique states found:** {len(unique_states)}")
            st.write("**States in data:**")
            for state in sorted(unique_states):
                count = len(data[data["state"] == state])
                normalized = normalize_name(state)
                st.write(f"- {state} ({count} rows) → normalized: '{normalized}'")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Dynamic filters
all_states = sorted(data["state"].dropna().unique().tolist())
state_sel = st.sidebar.multiselect("State", options=all_states, default=all_states[:3] if len(all_states) >= 3 else all_states)

if state_sel:
    districts_filtered = data[data["state"].isin(state_sel)]
else:
    districts_filtered = data
all_districts = sorted(districts_filtered["district"].dropna().unique().tolist())
district_sel = st.sidebar.multiselect("District", options=all_districts)

pincode_sel = st.sidebar.text_input("PIN code (optional)", value="")

min_date, max_date = data["date"].min(), data["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

age_group_choice = st.sidebar.radio(
    "Age group",
    options=["All"] + AGE_GROUP_KEYS,
    format_func=lambda x: "All" if x == "All" else AGE_GROUP_LABELS[x],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Compare Regions")
compare_level = st.sidebar.selectbox("Level", options=["state", "district"])
compare_a = st.sidebar.selectbox(
    "Region A",
    options=sorted(data[compare_level].dropna().unique().tolist()),
)
compare_b = st.sidebar.selectbox(
    "Region B",
    options=sorted(data[compare_level].dropna().unique().tolist()),
)

# Apply filters
filtered = filter_data(
    data,
    states=state_sel,
    districts=district_sel,
    pincode=pincode_sel,
    date_range=(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])),
    age_group_choice=age_group_choice,
)

# -----------------------------
# Main Dashboard
# -----------------------------
st.title("Aadhaar Enrolment Dashboard")
st.markdown("---")

kpis = compute_kpis(filtered, age_group_choice)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Enrolments", f"{kpis['total']:,}")
if kpis["growth_rate"] is not None:
    col2.metric("Growth Rate", f"{kpis['growth_rate']}%", delta=f"{kpis['growth_rate']}%")
else:
    col2.metric("Growth Rate", "N/A")
col3.metric("Highest Region", kpis["highest_region"] or "N/A")
col4.metric("Lowest Region", kpis["lowest_region"] or "N/A")

with st.expander("Age Group Distribution", expanded=True):
    cols = st.columns(3)
    for i, k in enumerate(AGE_GROUP_KEYS):
        label = AGE_GROUP_LABELS[k]
        val = kpis["age_group_percent"].get(label, 0.0)
        cols[i].metric(label, f"{val}%")

st.markdown("---")

# Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Geographic", "Age Groups", "Heatmap"])

with tab1:
    ts_fig = make_time_series(filtered, age_group_choice)
    st.plotly_chart(ts_fig, use_container_width=True)

with tab2:
    geo_level = st.radio("View by:", options=["state", "district"], horizontal=True, key="geo_level")
    geo_fig = make_geo_bars(filtered, level=geo_level)
    st.plotly_chart(geo_fig, use_container_width=True)

with tab3:
    dist_level = st.radio("Group by:", options=["state", "district"], horizontal=True, key="age_dist_level")
    age_fig = make_age_distribution(filtered, group_by=dist_level)
    st.plotly_chart(age_fig, use_container_width=True)

with tab4:
    heat_level = st.radio("Level:", options=["state", "district"], horizontal=True, key="heat_level")
    heat_fig = make_heatmap(filtered, level=heat_level)
    st.plotly_chart(heat_fig, use_container_width=True)

st.markdown("---")

# Insights
st.subheader("Insight Panel")
insights = generate_insights(filtered)
for i in insights:
    st.info(f"• {i}")

st.markdown("---")

# Rankings
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Region Rankings")
    rank_level = st.radio("Ranking by:", options=["state", "district"], horizontal=True, key="rank_level")
    rank_df = region_ranking_table(filtered, level=rank_level)
    st.dataframe(rank_df, use_container_width=True, height=400)

with col2:
    st.subheader("Export Data")
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="aadhaar_enrolment_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# Comparison
st.subheader(f"Region Comparison: {compare_a} vs {compare_b}")

def compare_regions(df: pd.DataFrame, level: str, a: str, b: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Metric", "Region A", "Region B"])
    if level == "state":
        df_a = df[df["state"] == a]
        df_b = df[df["state"] == b]
    else:
        df_a = df[df["district"] == a]
        df_b = df[df["district"] == b]

    metrics = {
        "Total enrolments": (df_a["total_enrolments"].sum(), df_b["total_enrolments"].sum()),
        "0–5 years": (df_a["age_0_5"].sum(), df_b["age_0_5"].sum()),
        "5–17 years": (df_a["age_5_17"].sum(), df_b["age_5_17"].sum()),
        "18+ years": (df_a["age_18_plus"].sum(), df_b["age_18_plus"].sum()),
    }
    out = pd.DataFrame(
        [{"Metric": k, "Region A": int(v[0]), "Region B": int(v[1])} for k, v in metrics.items()]
    )
    return out

comp_df = compare_regions(filtered, compare_level, compare_a, compare_b)

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(comp_df, use_container_width=True)

with col2:
    comp_melt = comp_df.melt(id_vars="Metric", var_name="Region", value_name="Enrolments")
    comp_fig = go.Figure()
    
    for i, region in enumerate(["Region A", "Region B"]):
        region_data = comp_melt[comp_melt["Region"] == region]
        comp_fig.add_trace(go.Bar(
            name=f"{compare_a if i == 0 else compare_b}",
            x=region_data["Metric"],
            y=region_data["Enrolments"],
            marker_color=COLOR_PALETTE[i],
            text=region_data["Enrolments"],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        ))
    
    comp_fig.update_layout(
        title=dict(text=f"Comparison: {compare_a} vs {compare_b}", font=dict(size=18, color="#1f2937")),
        xaxis_title="Metric",
        yaxis_title="Enrolments",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    comp_fig.update_xaxes(showgrid=False)
    comp_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    st.plotly_chart(comp_fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with Python, Pandas, NumPy, Plotly, and Streamlit.")
st.caption("Created by Ankan and Sudip ❤️.")
