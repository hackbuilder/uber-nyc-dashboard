"""
Uber NYC — GM Operations Dashboard
4-tab Streamlit app  |  TLC HVFHS data 2025-03 → 2026-03 + 2015 historical

Tab 1: Uber Operations   — KPIs, trend, heatmap, borough, WoW, zone map
Tab 2: Market Share      — Uber vs Lyft by zone, borough, hour, fare, wait
Tab 3: Demand Drivers    — Weather, holiday, seasonality, operational metrics, search trends
Tab 3: Demand Drivers    — Weather, holiday, seasonality, operational metrics, search trends
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
import streamlit as st
from datetime import timedelta

try:
    from pytrends.request import TrendReq
    _PYTRENDS_OK = True
except ImportError:
    _PYTRENDS_OK = False

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber NYC — GM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BOROUGH_ORDER  = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BOROUGH_COLORS = {
    "Manhattan": "#06C167", "Brooklyn": "#276EF1",
    "Queens": "#FF6B00", "Bronx": "#E11900", "Staten Island": "#7356BF",
}
UBER_COLOR = "#000000"
LYFT_COLOR = "#FF00BF"
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR_TICKS = dict(
    tickmode="array", tickvals=list(range(0, 24, 3)),
    ticktext=["12am", "3am", "6am", "9am", "12pm", "3pm", "6pm", "9pm"],
)
MAP_STYLE  = "carto-positron"
MAP_CENTER = {"lat": 40.72, "lon": -73.97}

# Holidays in data range (2025-03 → 2026-03)
HOLIDAYS = pd.DataFrame({
    "date": pd.to_datetime([
        "2025-03-17", "2025-05-26", "2025-06-19", "2025-07-04",
        "2025-09-01", "2025-10-13", "2025-10-31", "2025-11-02",
        "2025-11-11", "2025-11-27", "2025-11-28", "2025-12-24",
        "2025-12-25", "2025-12-31", "2026-01-01", "2026-01-19",
        "2026-02-14", "2026-02-16", "2026-03-17",
    ]),
    "name": [
        "St. Patrick's Day", "Memorial Day", "Juneteenth", "Independence Day",
        "Labor Day", "Columbus Day", "Halloween", "NYC Marathon",
        "Veterans Day", "Thanksgiving", "Black Friday", "Christmas Eve",
        "Christmas Day", "New Year's Eve", "New Year's Day", "MLK Day",
        "Valentine's Day", "Presidents Day", "St. Patrick's Day '26",
    ],
})

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading trip data…")
def load_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "trips_combined_2025.parquet"))
    df["request_date"] = pd.to_datetime(df["request_date"])
    return df

@st.cache_data(show_spinner=False)
def load_zone_summary():
    return pd.read_parquet(os.path.join(DATA_DIR, "trips_zone_summary.parquet"))

@st.cache_data(show_spinner=False)
def load_zone_geojson():
    with open(os.path.join(DATA_DIR, "taxi_zones.geojson")) as f:
        return json.load(f)

@st.cache_data(ttl=86400, show_spinner=False)
def load_weather(start: str, end: str):
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude=40.7128&longitude=-74.0060"
            f"&start_date={start}&end_date={end}"
            "&daily=precipitation_sum,temperature_2m_max,snowfall_sum"
            "&timezone=America%2FNew_York"
        )
        r = requests.get(url, timeout=15).json()
        return pd.DataFrame({
            "date":      pd.to_datetime(r["daily"]["time"]),
            "precip_mm": r["daily"]["precipitation_sum"],
            "temp_c":    r["daily"]["temperature_2m_max"],
            "snow_cm":   r["daily"]["snowfall_sum"],
        })
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_google_trends():
    """Pull weekly search interest for 'uber' and 'lyft' in NYC DMA.
    Returns None on rate-limit or if pytrends unavailable."""
    if not _PYTRENDS_OK:
        return None
    try:
        pt = TrendReq(hl="en-US", tz=-300, timeout=(10, 25))
        pt.build_payload(
            ["uber", "lyft"],
            cat=0,
            timeframe="2025-03-01 2026-03-31",
            geo="US-NY-501",  # NYC DMA
        )
        df = pt.interest_over_time()
        if df.empty:
            return None
        df = df.drop(columns=["isPartial"], errors="ignore").reset_index()
        df = df.rename(columns={"date": "week"})
        return df
    except Exception:
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────

def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = (
        df.groupby("request_date")
        .agg(trips=("trip_count","sum"), fare_sum=("base_fare_sum","sum"),
             wait_sum=("wait_time_sum","sum"), miles_sum=("trip_miles_sum","sum"))
        .reset_index().sort_values("request_date")
    )
    d["avg_fare"]     = d["fare_sum"]  / d["trips"]
    d["avg_wait_min"] = d["wait_sum"]  / d["trips"] / 60
    d["avg_miles"]    = d["miles_sum"] / d["trips"]
    return d

def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["week"] = d["request_date"] - pd.to_timedelta(d["request_date"].dt.dayofweek, unit="D")
    w = (
        d.groupby("week")
        .agg(trips=("trips","sum"), fare_sum=("fare_sum","sum"),
             wait_sum=("wait_sum","sum"), miles_sum=("miles_sum","sum"))
        .reset_index()
    )
    w["avg_fare"] = w["fare_sum"] / w["trips"]
    return w

def linear_forecast(series: pd.Series, n_ahead: int = 4):
    y = series.values.astype(float)
    x = np.arange(len(y))
    coeffs   = np.polyfit(x, y, 1)
    std      = (y - np.polyval(coeffs, x)).std()
    yhat     = np.polyval(coeffs, np.arange(len(y), len(y) + n_ahead))
    return yhat, float(std)

def fmt_delta(now, prev, inverse=False):
    if prev and prev != 0:
        pct = (now - prev) / abs(prev) * 100
        s   = f"{'+' if pct>=0 else ''}{pct:.1f}% WoW"
        col = ("normal" if pct>=0 else "inverse") if not inverse else ("inverse" if pct>=0 else "normal")
        return s, col
    return "N/A", "off"

def time_segment(hour):
    if   hour in range(7, 10):  return "AM Rush (7–9am)"
    elif hour in range(16, 20): return "PM Rush (4–7pm)"
    elif hour in [22, 23, 0, 1, 2]: return "Late Night (10pm–2am)"
    elif hour in range(10, 16): return "Midday (10am–3pm)"
    else:                       return "Early/Overnight"

SEG_ORDER = ["AM Rush (7–9am)", "Midday (10am–3pm)", "PM Rush (4–7pm)",
             "Late Night (10pm–2am)", "Early/Overnight"]
SEG_COLORS = {
    "AM Rush (7–9am)":      "#276EF1",
    "Midday (10am–3pm)":    "#06C167",
    "PM Rush (4–7pm)":      "#FF6B00",
    "Late Night (10pm–2am)":"#7356BF",
    "Early/Overnight":      "#AAAAAA",
}

def scatter_with_regression(x, y, xlab, ylab, color=None, title=""):
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    slope, intercept, r, p, _ = stats.linregress(xc, yc)
    x_line = np.linspace(xc.min(), xc.max(), 100)
    y_line = slope * x_line + intercept

    fig = go.Figure()
    if color is not None:
        for cat, grp in pd.DataFrame({"x": x, "y": y, "c": color}).groupby("c"):
            fig.add_trace(go.Scatter(
                x=grp["x"], y=grp["y"], mode="markers", name=str(cat),
                marker=dict(size=5, opacity=0.55),
            ))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="",
                                 marker=dict(size=5, color="#555", opacity=0.5), showlegend=False))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=f"Trend  R²={r**2:.2f}",
        line=dict(color="#E53935", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13, x=0, xanchor="left"),
        xaxis_title=xlab, yaxis_title=ylab,
        height=340, margin=dict(t=45, b=70, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0,
                    font_size=11, itemsizing="constant"),
    )
    return fig, r**2

# ── Load data ─────────────────────────────────────────────────────────────────
df_all   = load_data()
zone_sum = load_zone_summary()
zone_gj  = load_zone_geojson()
max_date = df_all["request_date"].max()
min_date = df_all["request_date"].min()
weather  = load_weather(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
# Google Trends loaded lazily inside Tab 3 to avoid blocking startup

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Uber NYC Dashboard")
    st.caption(f"TLC HVFHS  {min_date.strftime('%b %Y')} – {max_date.strftime('%b %Y')}")
    st.markdown("---")
    boroughs_sel = st.multiselect("Filter Boroughs", BOROUGH_ORDER, default=BOROUGH_ORDER)
    show_precip  = st.toggle("Show precipitation overlay (Tab 1)", value=False)
    st.markdown("---")
    st.caption("HV0003 = Uber · HV0005 = Lyft")

df_f = df_all[df_all["Borough"].isin(boroughs_sel)]

cur_end   = max_date
cur_start = cur_end   - timedelta(days=6)
prv_end   = cur_start - timedelta(days=1)
prv_start = prv_end   - timedelta(days=6)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UBER OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📈 Uber Operations", "🥊 Market Share", "🌦️ Demand Drivers",
])

with tab1:
    uber_f = df_f[df_f["provider"] == "Uber"]
    daily  = to_daily(uber_f)

    cur_d = daily[daily["request_date"].between(cur_start, cur_end)]
    prv_d = daily[daily["request_date"].between(prv_start, prv_end)]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.subheader(f"Operations Overview — Last 7 Days  (through {max_date.strftime('%b %d, %Y')})")
    k1, k2, k3, k4 = st.columns(4)

    def safe_avg(d, num_col, den_col):
        n, denom = d[num_col].sum(), d[den_col].sum()
        return n / denom if denom else 0.0

    cur_trips = int(cur_d["trips"].sum());  prv_trips = int(prv_d["trips"].sum())
    d_s, d_c  = fmt_delta(cur_trips, prv_trips)
    k1.metric("Total Rides", f"{cur_trips:,.0f}", d_s, delta_color=d_c)

    cur_fare = safe_avg(cur_d, "fare_sum", "trips");  prv_fare = safe_avg(prv_d, "fare_sum", "trips")
    d_s, d_c = fmt_delta(cur_fare, prv_fare)
    k2.metric("Avg Fare / Ride", f"${cur_fare:.2f}", d_s, delta_color=d_c)

    cur_wait = safe_avg(cur_d, "wait_sum", "trips") / 60;  prv_wait = safe_avg(prv_d, "wait_sum", "trips") / 60
    d_s, d_c = fmt_delta(cur_wait, prv_wait, inverse=True)
    k3.metric("Avg Wait Time", f"{cur_wait:.1f} min", d_s, delta_color=d_c)

    cur_mi = safe_avg(cur_d, "miles_sum", "trips");  prv_mi = safe_avg(prv_d, "miles_sum", "trips")
    d_s, d_c = fmt_delta(cur_mi, prv_mi)
    k4.metric("Avg Trip Miles", f"{cur_mi:.1f} mi", d_s, delta_color=d_c)

    st.divider()

    # ── Time Series + Demand Heatmap ──────────────────────────────────────────
    col_ts, col_hm = st.columns([3, 2])

    with col_ts:
        st.markdown("**Weekly Ride Volume & 4-Week Forecast**")
        weekly = to_weekly(daily)
        wc     = weekly.copy()
        days_in_last = (max_date - wc["week"].max()).days + 1
        if days_in_last < 4:
            wc = wc.iloc[:-1]
        wc["rolling4"] = wc["trips"].rolling(4, min_periods=2).mean()

        src                = wc.tail(12)
        yhat, std          = linear_forecast(src["trips"], 4)
        last_wk            = wc["week"].max()
        future_wks         = [last_wk + timedelta(weeks=i+1) for i in range(4)]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=wc["week"], y=wc["trips"],
            name="Weekly Rides", marker_color="#C8E6C9", opacity=0.85,
        ))
        fig_ts.add_trace(go.Scatter(
            x=wc["week"], y=wc["rolling4"], name="4-wk Avg",
            line=dict(color=UBER_COLOR, width=2.5),
        ))
        fig_ts.add_trace(go.Scatter(
            x=future_wks + future_wks[::-1],
            y=list(yhat + std) + list((yhat - std)[::-1]),
            fill="toself", fillcolor="rgba(6,193,103,0.18)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig_ts.add_trace(go.Scatter(
            x=future_wks, y=yhat, name="4-wk Forecast",
            line=dict(color="#06C167", width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=6),
        ))
        if show_precip and weather is not None:
            ww = weather.copy()
            ww["week"] = ww["date"] - pd.to_timedelta(ww["date"].dt.dayofweek, unit="D")
            ww_agg = ww.groupby("week")["precip_mm"].sum().reset_index()
            fig_ts.add_trace(go.Bar(
                x=ww_agg["week"], y=ww_agg["precip_mm"],
                name="Precip (mm)", yaxis="y2",
                marker_color="rgba(30,100,200,0.22)",
            ))
            fig_ts.update_layout(yaxis2=dict(
                title="Precip (mm)", overlaying="y", side="right",
                showgrid=False, rangemode="nonnegative",
            ))
        fig_ts.update_layout(
            xaxis_title=None, yaxis_title="Rides",
            height=360, margin=dict(t=10, b=30, l=50, r=50),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            barmode="overlay",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_hm:
        st.markdown("**When Do People Ride?** *(avg rides/week)*")
        n_wks = max(1, int((max_date - min_date).days / 7))
        hm = uber_f.groupby(["day_of_week","request_hour"])["trip_count"].sum().reset_index()
        hm["avg_rides"] = hm["trip_count"] / n_wks
        piv = (hm.pivot(index="day_of_week", columns="request_hour", values="avg_rides")
               .reindex(index=range(7), columns=range(24)).fillna(0))
        piv.index = DOW_LABELS
        fig_hm = px.imshow(
            piv,
            color_continuous_scale=[[0,"#f7fdf7"],[0.4,"#74c476"],[1,"#00441b"]],
            labels=dict(x="Hour of Day", y="", color="Avg Rides/Wk"),
            aspect="auto",
        )
        fig_hm.update_layout(height=360, margin=dict(t=10,b=30,l=10,r=10), xaxis=HOUR_TICKS)
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Use to time driver incentives and surge pricing windows.")

    st.divider()

    # ── Borough Bar + WoW ─────────────────────────────────────────────────────
    col_boro, col_wow = st.columns(2)

    with col_boro:
        metric_opt = st.selectbox(
            "Borough metric", ["Total Rides","Avg Fare ($)","Avg Wait (min)"],
            key="b_met", label_visibility="collapsed",
        )
        st.markdown(f"**Borough — {metric_opt}** *(Last 7 Days, WoW %)*")

        def boro_agg(mask):
            return (
                uber_f[mask & uber_f["Borough"].isin(BOROUGH_ORDER)]
                .groupby("Borough")
                .agg(trips=("trip_count","sum"), fare_sum=("base_fare_sum","sum"),
                     wait_sum=("wait_time_sum","sum"))
                .reset_index()
                .assign(avg_fare=lambda x: x.fare_sum/x.trips,
                        avg_wait=lambda x: x.wait_sum/x.trips/60)
            )

        bc = boro_agg(uber_f["request_date"].between(cur_start, cur_end))
        bp = boro_agg(uber_f["request_date"].between(prv_start, prv_end)).rename(
            columns={"trips":"trips_p","avg_fare":"fare_p","avg_wait":"wait_p"})
        bdf = bc.merge(bp[["Borough","trips_p","fare_p","wait_p"]], on="Borough", how="left")

        vcol, pcol = {"Total Rides":("trips","trips_p"),"Avg Fare ($)":("avg_fare","fare_p"),
                      "Avg Wait (min)":("avg_wait","wait_p")}[metric_opt]
        bdf["wow"] = (bdf[vcol] - bdf[pcol]) / bdf[pcol].abs() * 100
        bdf = bdf.sort_values(vcol, ascending=True)

        fig_b = go.Figure(go.Bar(
            y=bdf["Borough"], x=bdf[vcol], orientation="h",
            marker_color=[BOROUGH_COLORS.get(b,"#aaa") for b in bdf["Borough"]],
            text=bdf["wow"].apply(lambda v: f"{v:+.1f}%"), textposition="outside",
        ))
        fig_b.update_layout(
            xaxis_title=metric_opt, height=320, showlegend=False,
            margin=dict(t=10,b=30,l=10,r=70),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with col_wow:
        st.markdown("**This Week vs Prior Week** *(rides by day)*")
        daily["dow"] = daily["request_date"].dt.dayofweek
        cw = daily[daily["request_date"].between(cur_start, cur_end)].groupby("dow")["trips"].sum().reindex(range(7), fill_value=0)
        pw = daily[daily["request_date"].between(prv_start, prv_end)].groupby("dow")["trips"].sum().reindex(range(7), fill_value=0)
        fig_wow = go.Figure()
        fig_wow.add_trace(go.Bar(x=DOW_LABELS, y=pw.values, name="Prior Week", marker_color="#CCCCCC"))
        fig_wow.add_trace(go.Bar(x=DOW_LABELS, y=cw.values, name="This Week",  marker_color="#06C167"))
        fig_wow.update_layout(
            barmode="group", yaxis_title="Rides", height=320,
            margin=dict(t=10,b=30), plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_wow, use_container_width=True)

    st.divider()

    # ── Zone-level Map ────────────────────────────────────────────────────────
    st.markdown("**Ride Volume by Pickup Zone** *(Uber, full period)*")

    z_uber = zone_sum[
        (zone_sum["provider"] == "Uber") &
        (zone_sum["Borough"].isin(boroughs_sel))
    ][["LocationID","Zone","Borough","trip_count"]].copy()

    fig_map = px.choropleth_mapbox(
        z_uber,
        geojson=zone_gj,
        locations="LocationID",
        color="trip_count",
        color_continuous_scale=[[0,"#edf8e9"],[0.5,"#74c476"],[1,"#00441b"]],
        mapbox_style=MAP_STYLE,
        center=MAP_CENTER,
        zoom=9.5,
        opacity=0.75,
        hover_name="Zone",
        hover_data={"Borough": True, "trip_count": ":,.0f", "LocationID": False},
        labels={"trip_count": "Rides"},
    )
    fig_map.update_layout(
        height=450, margin=dict(t=0,b=0,l=0,r=0),
        coloraxis_colorbar=dict(title="Total Rides"),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET SHARE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Uber vs Lyft — NYC Rideshare Market")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    by_prov = df_f.groupby("provider")["trip_count"].sum()
    total   = by_prov.sum()
    u_n, l_n = by_prov.get("Uber",0), by_prov.get("Lyft",0)
    u_pct, l_pct = u_n/total*100, l_n/total*100

    def wk_share(s, e):
        sub = df_f[df_f["request_date"].between(s, e)]
        bp  = sub.groupby("provider")["trip_count"].sum()
        t   = bp.sum()
        return bp.get("Uber",0)/t*100 if t else 0

    delta_pp = wk_share(cur_start, cur_end) - wk_share(prv_start, prv_end)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Uber Market Share", f"{u_pct:.1f}%", f"{delta_pp:+.2f} pp this week")
    m2.metric("Lyft Market Share", f"{l_pct:.1f}%", f"{-delta_pp:+.2f} pp this week")
    m3.metric("Uber Rides (period)", f"{u_n:,.0f}")
    m4.metric("Lyft Rides (period)", f"{l_n:,.0f}")

    st.divider()

    # ── Donut + Weekly Share Trend ─────────────────────────────────────────────
    col_d, col_tr = st.columns([1, 2])

    with col_d:
        st.markdown("**2025–26 Overall Split**")
        fig_donut = go.Figure(go.Pie(
            labels=["Uber","Lyft"], values=[u_n, l_n],
            hole=0.58, marker_colors=[UBER_COLOR, LYFT_COLOR],
            textinfo="label+percent", textfont_size=14, insidetextorientation="radial",
        ))
        fig_donut.update_layout(
            height=300, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
            annotations=[dict(text=f"<b>{u_pct:.0f}%</b><br>Uber", x=0.5, y=0.5,
                              font_size=16, showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_tr:
        st.markdown("**Weekly Uber Share (%) — Trend**")
        ws = df_f.groupby(["request_date","provider"])["trip_count"].sum().reset_index()
        ws["week"] = ws["request_date"] - pd.to_timedelta(ws["request_date"].dt.dayofweek, unit="D")
        wp = ws.groupby(["week","provider"])["trip_count"].sum().unstack(fill_value=0).reset_index()
        wp["total"]    = wp.get("Uber",0) + wp.get("Lyft",0)
        wp["uber_pct"] = wp.get("Uber",0) / wp["total"] * 100
        last_r = wp.dropna(subset=["uber_pct"]).iloc[-1]

        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(
            x=wp["week"], y=wp["uber_pct"], fill="tozeroy",
            fillcolor="rgba(0,0,0,0.07)", line=dict(color=UBER_COLOR, width=2.5),
            name="Uber %",
        ))
        fig_tr.add_hline(y=50, line_dash="dash", line_color="#888",
                         annotation_text="50% parity", annotation_position="bottom right")
        fig_tr.add_annotation(x=last_r["week"], y=last_r["uber_pct"],
                               text=f"  {last_r['uber_pct']:.1f}%", showarrow=False,
                               font=dict(size=12, color=UBER_COLOR), xanchor="left")
        fig_tr.update_layout(
            yaxis=dict(range=[0,100]), xaxis_title=None, yaxis_title="Uber Share (%)",
            height=300, margin=dict(t=10,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_tr, use_container_width=True)

    st.divider()

    # ── Zone Market Share Map ──────────────────────────────────────────────────
    st.markdown("**Uber Market Share by Pickup Zone**")
    st.caption("Hover a zone to see Uber %, rides, and which borough it belongs to.")

    z_all = (
        zone_sum[zone_sum["Borough"].isin(boroughs_sel)]
        .groupby(["PULocationID","provider","Zone","Borough"])["trip_count"].sum()
        .reset_index()
    )
    z_pivot = z_all.pivot_table(
        index=["PULocationID","Zone","Borough"], columns="provider",
        values="trip_count", aggfunc="sum", fill_value=0,
    ).reset_index()
    z_pivot.columns.name = None
    if "Uber" not in z_pivot.columns: z_pivot["Uber"] = 0
    if "Lyft" not in z_pivot.columns: z_pivot["Lyft"] = 0
    z_pivot["total"]    = z_pivot["Uber"] + z_pivot["Lyft"]
    z_pivot["uber_pct"] = np.where(z_pivot["total"] > 0, z_pivot["Uber"] / z_pivot["total"] * 100, 50.0)

    fig_zmap = px.choropleth_mapbox(
        z_pivot,
        geojson=zone_gj,
        locations="PULocationID",
        color="uber_pct",
        color_continuous_scale=[[0, LYFT_COLOR],[0.5,"#f0f0f0"],[1, "#333333"]],
        range_color=[45, 90],
        mapbox_style=MAP_STYLE,
        center=MAP_CENTER,
        zoom=9.5,
        opacity=0.78,
        hover_name="Zone",
        hover_data={"Borough": True, "uber_pct": ":.1f", "Uber": ":,.0f",
                    "Lyft": ":,.0f", "PULocationID": False},
        labels={"uber_pct": "Uber %", "Uber": "Uber Rides", "Lyft": "Lyft Rides"},
    )
    fig_zmap.update_layout(
        height=480, margin=dict(t=0,b=0,l=0,r=0),
        coloraxis_colorbar=dict(title="Uber %", ticksuffix="%"),
    )
    st.plotly_chart(fig_zmap, use_container_width=True)

    st.divider()

    # ── Borough Share + Hour/Day Heatmap ──────────────────────────────────────
    col_bs, col_hs = st.columns(2)

    with col_bs:
        st.markdown("**Market Share by Borough**")
        bp2 = (df_f[df_f["Borough"].isin(BOROUGH_ORDER)]
               .groupby(["Borough","provider"])["trip_count"].sum().reset_index())
        bt2 = bp2.groupby("Borough")["trip_count"].sum().reset_index().rename(columns={"trip_count":"total"})
        bp2 = bp2.merge(bt2, on="Borough")
        bp2["share"] = bp2["trip_count"] / bp2["total"] * 100

        fig_bs = px.bar(
            bp2, x="share", y="Borough", color="provider", orientation="h", barmode="stack",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"share":"Share (%)","Borough":""},
            category_orders={"Borough":BOROUGH_ORDER[::-1]},
            text=bp2["share"].apply(lambda v: f"{v:.0f}%"),
        )
        fig_bs.update_traces(textposition="inside", textfont_color="white", textfont_size=11)
        fig_bs.update_layout(
            height=320, margin=dict(t=10,b=30,l=10,r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(range=[0,100], title="Share (%)"),
            legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bs, use_container_width=True)

    with col_hs:
        st.markdown("**Uber vs Lyft: Wait Time Gap by Hour & Day**")
        st.caption("Green = Uber faster · Red = Uber slower than Lyft (minutes)")
        hw = df_f.groupby(["day_of_week","request_hour","provider"]).agg(
            wait_sum=("wait_time_sum","sum"), trips=("trip_count","sum")
        ).reset_index()
        hw["avg_wait_min"] = hw["wait_sum"] / hw["trips"] / 60
        hw_u = hw[hw["provider"]=="Uber"][["day_of_week","request_hour","avg_wait_min"]].rename(columns={"avg_wait_min":"u_wait"})
        hw_l = hw[hw["provider"]=="Lyft"][["day_of_week","request_hour","avg_wait_min"]].rename(columns={"avg_wait_min":"l_wait"})
        hw_gap = hw_u.merge(hw_l, on=["day_of_week","request_hour"], how="inner")
        hw_gap["gap"] = hw_gap["u_wait"] - hw_gap["l_wait"]
        piv_gap = (hw_gap.pivot(index="day_of_week", columns="request_hour", values="gap")
                   .reindex(index=range(7), columns=range(24)).fillna(0))
        piv_gap.index = DOW_LABELS
        absmax = max(abs(piv_gap.values.max()), abs(piv_gap.values.min()), 0.5)
        fig_hs = px.imshow(
            piv_gap,
            color_continuous_scale=[[0,"#06C167"],[0.5,"#f5f5f5"],[1,"#E53935"]],
            zmin=-absmax, zmax=absmax,
            labels=dict(x="Hour of Day", color="Gap (min)"), aspect="auto",
        )
        fig_hs.update_layout(
            height=320, margin=dict(t=10,b=30,l=10,r=10), xaxis=HOUR_TICKS,
            coloraxis_colorbar=dict(title="Uber−Lyft<br>Wait (min)"),
        )
        st.plotly_chart(fig_hs, use_container_width=True)

    st.divider()

    # ── Fare & Wait Comparison ────────────────────────────────────────────────
    st.markdown("**Avg Fare & Wait Time: Uber vs Lyft by Borough**")

    cmp = (
        df_f[df_f["Borough"].isin(BOROUGH_ORDER)]
        .groupby(["Borough","provider"])
        .agg(trips=("trip_count","sum"), fare_sum=("base_fare_sum","sum"),
             wait_sum=("wait_time_sum","sum"))
        .reset_index()
        .assign(avg_fare=lambda x: x.fare_sum/x.trips,
                avg_wait_min=lambda x: x.wait_sum/x.trips/60)
    )

    cf1, cf2 = st.columns(2)
    with cf1:
        fig_fare = px.bar(
            cmp, x="Borough", y="avg_fare", color="provider", barmode="group",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"avg_fare":"Avg Base Fare ($)","Borough":"","provider":""},
            category_orders={"Borough":BOROUGH_ORDER},
        )
        fig_fare.update_layout(
            title=dict(text="Avg Fare per Ride", font_size=14, x=0, xanchor="left"),
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
            margin=dict(t=45, b=70, l=50, r=20),
        )
        st.plotly_chart(fig_fare, use_container_width=True)

    with cf2:
        fig_wait = px.bar(
            cmp, x="Borough", y="avg_wait_min", color="provider", barmode="group",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"avg_wait_min":"Avg Wait Time (min)","Borough":"","provider":""},
            category_orders={"Borough":BOROUGH_ORDER},
        )
        fig_wait.update_layout(
            title=dict(text="Avg Wait Time", font_size=14, x=0, xanchor="left"),
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
            margin=dict(t=45, b=70, l=50, r=20),
        )
        st.plotly_chart(fig_wait, use_container_width=True)

    st.divider()

    # ── Monthly Share Trend by Borough ────────────────────────────────────────
    st.markdown("**How Uber's Share Has Shifted by Borough — Month by Month**")
    st.caption("Each line = one borough. Dashed 50% line = parity with Lyft.")

    mshare = (
        df_f[df_f["Borough"].isin(BOROUGH_ORDER)]
        .assign(month=df_f["request_date"].dt.to_period("M"))
        .groupby(["month","Borough","provider"])["trip_count"].sum()
        .reset_index()
    )
    mshare_tot = mshare.groupby(["month","Borough"])["trip_count"].sum().reset_index().rename(columns={"trip_count":"total"})
    mshare = mshare.merge(mshare_tot, on=["month","Borough"])
    mshare = mshare[mshare["provider"]=="Uber"].copy()
    mshare["uber_pct"] = mshare["trip_count"] / mshare["total"] * 100
    mshare["month_dt"] = mshare["month"].dt.to_timestamp()

    fig_msh = go.Figure()
    for boro in BOROUGH_ORDER:
        sub = mshare[mshare["Borough"]==boro].sort_values("month_dt")
        if sub.empty:
            continue
        fig_msh.add_trace(go.Scatter(
            x=sub["month_dt"], y=sub["uber_pct"],
            mode="lines+markers", name=boro,
            line=dict(color=BOROUGH_COLORS.get(boro,"#aaa"), width=2.5),
            marker=dict(size=6),
        ))
    fig_msh.add_hline(y=50, line_dash="dash", line_color="#aaa",
                      annotation_text="50% parity", annotation_position="bottom right")
    fig_msh.update_layout(
        yaxis=dict(range=[30, 100], title="Uber Share (%)"),
        xaxis_title=None, height=320,
        margin=dict(t=10, b=60, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font_size=11),
    )
    st.plotly_chart(fig_msh, use_container_width=True)

    st.divider()

    # ── Trip Profile: Distance & Driver Pay ───────────────────────────────────
    st.markdown("**Trip Profile Comparison: Uber vs Lyft**")

    tp = (
        df_f[df_f["Borough"].isin(BOROUGH_ORDER)]
        .groupby(["Borough","provider"])
        .agg(trips=("trip_count","sum"), miles_sum=("trip_miles_sum","sum"),
             driver_sum=("driver_pay_sum","sum"), fare_sum=("base_fare_sum","sum"),
             airport=("airport_trips","sum"))
        .reset_index()
        .assign(
            avg_miles    = lambda x: x.miles_sum / x.trips,
            driver_ratio = lambda x: x.driver_sum / x.fare_sum * 100,
            airport_pct  = lambda x: x.airport / x.trips * 100,
        )
    )

    tp1, tp2, tp3 = st.columns(3)
    with tp1:
        fig_mi = px.bar(
            tp, x="Borough", y="avg_miles", color="provider", barmode="group",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"avg_miles":"Avg Trip Miles","Borough":"","provider":""},
            category_orders={"Borough":BOROUGH_ORDER},
        )
        fig_mi.update_layout(
            title=dict(text="Avg Trip Distance", font_size=13, x=0, xanchor="left"),
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
            margin=dict(t=45, b=70, l=50, r=10),
        )
        st.plotly_chart(fig_mi, use_container_width=True)
        st.caption("Longer avg trips = higher-value bookings per ride.")

    with tp2:
        fig_dr = px.bar(
            tp, x="Borough", y="driver_ratio", color="provider", barmode="group",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"driver_ratio":"Driver Pay / Fare (%)","Borough":"","provider":""},
            category_orders={"Borough":BOROUGH_ORDER},
        )
        fig_dr.update_layout(
            title=dict(text="Driver Pay % of Fare", font_size=13, x=0, xanchor="left"),
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
            margin=dict(t=45, b=70, l=50, r=10),
        )
        st.plotly_chart(fig_dr, use_container_width=True)
        st.caption("Higher ratio → better driver economics. Tracks take-rate vs Lyft.")

    with tp3:
        fig_ap2 = px.bar(
            tp, x="Borough", y="airport_pct", color="provider", barmode="group",
            color_discrete_map={"Uber":UBER_COLOR,"Lyft":LYFT_COLOR},
            labels={"airport_pct":"Airport Trip % ","Borough":"","provider":""},
            category_orders={"Borough":BOROUGH_ORDER},
        )
        fig_ap2.update_layout(
            title=dict(text="Airport Trip Share", font_size=13, x=0, xanchor="left"),
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
            margin=dict(t=45, b=70, l=50, r=10),
        )
        st.plotly_chart(fig_ap2, use_container_width=True)
        st.caption("Airport trips carry airport fees and tend to have longer distances & higher fares.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEMAND DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("What Drives Uber Demand in NYC?")

    uber_all   = df_all[df_all["provider"] == "Uber"]
    daily_all  = to_daily(uber_all)
    daily_all["dow"]   = daily_all["request_date"].dt.dayofweek
    daily_all["month"] = daily_all["request_date"].dt.to_period("M")

    # ── Weather Correlation ───────────────────────────────────────────────────
    if weather is not None:
        wdf = weather.rename(columns={"date": "request_date"})
        joined = daily_all.merge(wdf, on="request_date", how="inner")
        joined = joined[joined["temp_c"].notna() & joined["precip_mm"].notna()]

        st.markdown("### Weather vs Daily Rides")
        wc1, wc2 = st.columns(2)

        with wc1:
            season = joined["request_date"].dt.month.map(
                {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                 6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
            )
            # Compute R² first so the title is set correctly in one pass
            _, _, r_t, _, _ = stats.linregress(
                joined["temp_c"].dropna(),
                joined["trips"][joined["temp_c"].notna()],
            )
            fig_temp, _ = scatter_with_regression(
                joined["temp_c"].values, joined["trips"].values,
                "Max Temperature (°C)", "Daily Rides",
                color=season,
                title=f"Temperature vs Demand  (R²={r_t**2:.2f})",
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            if r_t**2 > 0.1:
                direction = "warmer" if r_t > 0 else "colder"
                st.caption(f"**Insight:** {direction.capitalize()} days → {'more' if r_t > 0 else 'fewer'} rides (R²={r_t**2:.2f}).")
            else:
                st.caption(f"Temperature shows weak correlation with demand (R²={r_t**2:.2f}).")

        with wc2:
            rain_mask  = joined["precip_mm"] > 0
            color_rain = rain_mask.map({True: "Rainy", False: "Dry"})
            _, _, r_r, _, _ = stats.linregress(joined["precip_mm"], joined["trips"])
            fig_rain, _ = scatter_with_regression(
                joined["precip_mm"].values, joined["trips"].values,
                "Precipitation (mm)", "Daily Rides",
                color=color_rain,
                title=f"Precipitation vs Demand  (R²={r_r**2:.2f})",
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            if r_r > 0:
                st.caption(f"**Insight:** Rainy days boost ride demand (R²={r_r**2:.2f}) — monitor for driver supply shortfall.")
            else:
                st.caption(f"Precipitation shows weak negative correlation with demand (R²={r_r**2:.2f}).")

        st.divider()
    else:
        st.info("Weather data unavailable. Check internet connection.")
        st.divider()

    # ── Holiday Effect ────────────────────────────────────────────────────────
    st.markdown("### Holiday & Event Impact on Rides")
    st.caption("% difference vs the median for that day-of-week across all non-holiday dates.")

    # Build baseline: median rides per day-of-week from non-holiday dates
    holiday_dates = set(HOLIDAYS["date"].dt.date)
    non_hol = daily_all[~daily_all["request_date"].dt.date.isin(holiday_dates)].copy()
    dow_median = non_hol.groupby("dow")["trips"].median()

    hol_rows = []
    for _, row in HOLIDAYS.iterrows():
        match = daily_all[daily_all["request_date"].dt.date == row["date"].date()]
        if match.empty:
            continue
        actual  = float(match["trips"].iloc[0])
        dow     = int(match["dow"].iloc[0])
        baseline = float(dow_median.get(dow, np.nan))
        if np.isnan(baseline) or baseline == 0:
            continue
        hol_rows.append({
            "name":     row["name"],
            "date":     row["date"],
            "actual":   actual,
            "baseline": baseline,
            "pct":      (actual - baseline) / baseline * 100,
            "dow":      DOW_LABELS[dow],
        })

    if hol_rows:
        hdf = pd.DataFrame(hol_rows).sort_values("pct")
        hdf["color"]  = hdf["pct"].apply(lambda v: "#06C167" if v >= 0 else "#E53935")
        hdf["label"]  = hdf["pct"].apply(lambda v: f"{v:+.0f}%")
        hdf["hover"]  = hdf.apply(
            lambda r: f"{r['name']} ({r['dow']})<br>{r['actual']:,.0f} rides<br>{r['pct']:+.1f}% vs typical {r['dow']}",
            axis=1,
        )

        fig_hol = go.Figure(go.Bar(
            x=hdf["pct"], y=hdf["name"],
            orientation="h",
            marker_color=hdf["color"],
            text=hdf["label"], textposition="outside",
            hovertext=hdf["hover"], hoverinfo="text",
        ))
        fig_hol.add_vline(x=0, line_color="#444", line_width=1.5)
        fig_hol.update_layout(
            xaxis_title="% vs Typical Day-of-Week", yaxis_title=None,
            height=max(350, len(hdf) * 24),
            margin=dict(t=10, b=30, l=10, r=60),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_hol, use_container_width=True)
        st.caption("Green = demand above normal · Red = demand below normal. Use to pre-position drivers or run targeted promos.")
    else:
        st.info("No holiday data found in the current date range.")

    st.divider()

    # ── Seasonality + Day-of-Week Distribution ────────────────────────────────
    col_mon, col_dow = st.columns(2)

    with col_mon:
        st.markdown("**Monthly Ride Volume** *(seasonality)*")
        monthly = (
            daily_all.groupby("month")["trips"].sum().reset_index()
        )
        monthly["month_str"] = monthly["month"].dt.strftime("%b %Y")
        monthly["month_dt"]  = monthly["month"].dt.to_timestamp()
        monthly = monthly.sort_values("month_dt")

        fig_mon = px.bar(
            monthly, x="month_str", y="trips",
            color_discrete_sequence=["#06C167"],
            labels={"trips":"Total Rides","month_str":""},
        )
        # Highlight Dec (holiday surge) and summer dip
        fig_mon.update_layout(
            height=340, margin=dict(t=10,b=40,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    with col_dow:
        st.markdown("**Day-of-Week Distribution** *(daily ride count)*")
        st.caption("Box = middle 50% of days. Shows which days are most consistent vs variable.")

        fig_box = go.Figure()
        for i, label in enumerate(DOW_LABELS):
            vals = daily_all[daily_all["dow"] == i]["trips"].values
            fig_box.add_trace(go.Box(
                y=vals, name=label,
                marker_color=["#06C167" if i < 5 else "#276EF1"][0],
                boxmean=True,
                showlegend=False,
            ))
        fig_box.update_layout(
            yaxis_title="Daily Rides", height=340,
            margin=dict(t=10,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── Snow Impact ────────────────────────────────────────────────────────────
    if weather is not None and "snow_cm" in weather.columns:
        st.markdown("### Snow Days vs Ride Demand")
        snow_joined = daily_all.merge(
            weather.rename(columns={"date":"request_date"})[["request_date","snow_cm","precip_mm"]],
            on="request_date", how="inner",
        )
        snow_joined = snow_joined[snow_joined["snow_cm"].notna()]
        snow_joined["is_snow"] = snow_joined["snow_cm"] > 0.5

        avg_snow    = snow_joined[snow_joined["is_snow"]]["trips"].mean()
        avg_no_snow = snow_joined[~snow_joined["is_snow"]]["trips"].mean()

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Avg Rides — Snow Days",    f"{avg_snow:,.0f}" if not np.isnan(avg_snow) else "N/A")
        sc2.metric("Avg Rides — No Snow",       f"{avg_no_snow:,.0f}" if not np.isnan(avg_no_snow) else "N/A")
        if not np.isnan(avg_snow) and not np.isnan(avg_no_snow) and avg_no_snow > 0:
            sc3.metric("Snow Impact", f"{(avg_snow-avg_no_snow)/avg_no_snow*100:+.1f}%",
                       "vs non-snow days")

        snow_days = int(snow_joined["is_snow"].sum())
        st.caption(f"Based on {snow_days} snow days (>0.5cm) in the dataset.")

    st.divider()

    # ── Monthly Operational Metrics ───────────────────────────────────────────
    st.markdown("### Operational Metrics Over Time")
    st.caption("Monthly averages — use to spot pricing drift, supply issues, or seasonal quality shifts.")

    monthly_ops = (
        uber_all
        .assign(month=uber_all["request_date"].dt.to_period("M"))
        .groupby("month")
        .agg(
            trips        = ("trip_count",     "sum"),
            fare_sum     = ("base_fare_sum",  "sum"),
            wait_sum     = ("wait_time_sum",  "sum"),
            miles_sum    = ("trip_miles_sum", "sum"),
            driver_sum   = ("driver_pay_sum", "sum"),
            airport      = ("airport_trips",  "sum"),
            shared       = ("shared_trips",   "sum"),
        )
        .reset_index()
        .assign(
            month_dt     = lambda x: x["month"].dt.to_timestamp(),
            avg_fare     = lambda x: x.fare_sum  / x.trips,
            avg_wait_min = lambda x: x.wait_sum  / x.trips / 60,
            avg_miles    = lambda x: x.miles_sum / x.trips,
            driver_ratio = lambda x: x.driver_sum / x.fare_sum * 100,
            airport_pct  = lambda x: x.airport   / x.trips * 100,
            shared_pct   = lambda x: x.shared    / x.trips * 100,
            month_str    = lambda x: x["month"].dt.strftime("%b %Y"),
        )
    )

    om1, om2 = st.columns(2)
    with om1:
        fig_fare_trend = go.Figure()
        fig_fare_trend.add_trace(go.Scatter(
            x=monthly_ops["month_dt"], y=monthly_ops["avg_fare"],
            mode="lines+markers", name="Avg Fare", line=dict(color="#06C167", width=2.5),
            marker=dict(size=6),
        ))
        fig_fare_trend.update_layout(
            title="Avg Fare / Ride (Monthly)", yaxis_title="USD",
            height=280, margin=dict(t=40,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_fare_trend, use_container_width=True)

    with om2:
        fig_wait_trend = go.Figure()
        fig_wait_trend.add_trace(go.Scatter(
            x=monthly_ops["month_dt"], y=monthly_ops["avg_wait_min"],
            mode="lines+markers", name="Avg Wait", line=dict(color="#E53935", width=2.5),
            marker=dict(size=6),
        ))
        fig_wait_trend.update_layout(
            title="Avg Wait Time (Monthly)", yaxis_title="Minutes",
            height=280, margin=dict(t=40,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_wait_trend, use_container_width=True)

    om3, om4 = st.columns(2)
    with om3:
        fig_driver = go.Figure()
        fig_driver.add_trace(go.Bar(
            x=monthly_ops["month_str"], y=monthly_ops["driver_ratio"],
            name="Driver Pay %", marker_color="#276EF1",
            text=monthly_ops["driver_ratio"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
        ))
        fig_driver.update_layout(
            title="Driver Pay as % of Base Fare", yaxis_title="%",
            height=280, margin=dict(t=40,b=40,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30), showlegend=False,
        )
        st.plotly_chart(fig_driver, use_container_width=True)
        st.caption("Driver pay ratio reflects Uber's take rate. A rising ratio = more driver-friendly or competitive pressure.")

    with om4:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=monthly_ops["month_dt"], y=monthly_ops["avg_miles"],
            mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(6,193,103,0.12)", line=dict(color="#06C167", width=2),
            marker=dict(size=5),
        ))
        fig_dist.update_layout(
            title="Avg Trip Distance (Monthly)", yaxis_title="Miles",
            height=280, margin=dict(t=40,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── Airport & Shared Rides ────────────────────────────────────────────────
    st.markdown("### Airport Trips & Shared Rides")

    ap1, ap2 = st.columns(2)
    with ap1:
        fig_airport = go.Figure()
        fig_airport.add_trace(go.Scatter(
            x=monthly_ops["month_dt"], y=monthly_ops["airport_pct"],
            mode="lines+markers", name="Airport %",
            line=dict(color="#FF6B00", width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(255,107,0,0.1)",
        ))
        fig_airport.update_layout(
            title="Airport Trips as % of All Rides", yaxis_title="%",
            height=280, margin=dict(t=40,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_airport, use_container_width=True)
        st.caption("Includes JFK, LGA, and EWR. Spikes signal travel surges (holidays, events).")

    with ap2:
        # Airport trips by borough
        airport_boro = (
            uber_all[uber_all["Borough"].isin(BOROUGH_ORDER)]
            .groupby("Borough")
            .agg(airport=("airport_trips","sum"), trips=("trip_count","sum"))
            .reset_index()
            .assign(airport_pct=lambda x: x.airport / x.trips * 100)
            .sort_values("airport_pct")
        )
        fig_ap_boro = go.Figure(go.Bar(
            y=airport_boro["Borough"], x=airport_boro["airport_pct"],
            orientation="h",
            marker_color=[BOROUGH_COLORS.get(b,"#aaa") for b in airport_boro["Borough"]],
            text=airport_boro["airport_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_ap_boro.update_layout(
            title="Airport Trip % by Borough", xaxis_title="%",
            height=280, margin=dict(t=40,b=30,l=10,r=60),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_ap_boro, use_container_width=True)

    sh1, sh2 = st.columns(2)
    with sh1:
        fig_shared = go.Figure()
        fig_shared.add_trace(go.Scatter(
            x=monthly_ops["month_dt"], y=monthly_ops["shared_pct"],
            mode="lines+markers", name="Shared %",
            line=dict(color="#7356BF", width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(115,86,191,0.1)",
        ))
        fig_shared.update_layout(
            title="Shared (Pooled) Ride Rate — Monthly", yaxis_title="%",
            height=280, margin=dict(t=40,b=30,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_shared, use_container_width=True)
        st.caption("Shared ride rate reflects consumer price-sensitivity and product mix.")

    with sh2:
        shared_boro = (
            uber_all[uber_all["Borough"].isin(BOROUGH_ORDER)]
            .groupby("Borough")
            .agg(shared=("shared_trips","sum"), trips=("trip_count","sum"))
            .reset_index()
            .assign(shared_pct=lambda x: x.shared / x.trips * 100)
            .sort_values("shared_pct")
        )
        fig_sh_boro = go.Figure(go.Bar(
            y=shared_boro["Borough"], x=shared_boro["shared_pct"],
            orientation="h",
            marker_color=[BOROUGH_COLORS.get(b,"#aaa") for b in shared_boro["Borough"]],
            text=shared_boro["shared_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_sh_boro.update_layout(
            title="Shared Ride % by Borough", xaxis_title="%",
            height=280, margin=dict(t=40,b=30,l=10,r=60),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_sh_boro, use_container_width=True)

    st.divider()

    # ── Rush-Hour Segment Analysis ────────────────────────────────────────────
    st.markdown("### Rides by Time-of-Day Segment")
    st.caption("How each demand segment has trended over the year.")

    seg_data = (
        uber_all.assign(
            month   = uber_all["request_date"].dt.to_period("M"),
            segment = uber_all["request_hour"].apply(time_segment),
        )
        .groupby(["month", "segment"])["trip_count"].sum()
        .reset_index()
        .assign(month_dt=lambda x: x["month"].dt.to_timestamp())
    )
    # Compute segment share within each month
    seg_total = seg_data.groupby("month")["trip_count"].sum().reset_index().rename(columns={"trip_count":"total"})
    seg_data  = seg_data.merge(seg_total, on="month")
    seg_data["share_pct"] = seg_data["trip_count"] / seg_data["total"] * 100

    rs1, rs2 = st.columns(2)
    with rs1:
        fig_seg_abs = px.line(
            seg_data, x="month_dt", y="trip_count", color="segment",
            color_discrete_map=SEG_COLORS,
            labels={"trip_count":"Rides","month_dt":"","segment":"Segment"},
            category_orders={"segment": SEG_ORDER},
        )
        fig_seg_abs.update_traces(mode="lines+markers", marker_size=5)
        fig_seg_abs.update_layout(
            title="Monthly Rides by Time Segment", height=320,
            margin=dict(t=40,b=60,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font_size=10),
        )
        st.plotly_chart(fig_seg_abs, use_container_width=True)

    with rs2:
        fig_seg_share = px.area(
            seg_data, x="month_dt", y="share_pct", color="segment",
            color_discrete_map=SEG_COLORS,
            labels={"share_pct":"Share (%)","month_dt":"","segment":"Segment"},
            category_orders={"segment": SEG_ORDER},
        )
        fig_seg_share.update_layout(
            title="Time Segment Share of Total Rides (%)", height=320,
            margin=dict(t=40,b=60,l=50,r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font_size=10),
        )
        st.plotly_chart(fig_seg_share, use_container_width=True)
        st.caption("A growing Late Night share signals nightlife recovery; shrinking AM Rush may reflect WFH trends.")

    st.divider()

    # ── Google Trends: Search Interest ───────────────────────────────────────
    st.markdown("### Search Query Trends (Google Trends — NYC DMA)")
    st.caption("Weekly search interest index (0–100) for 'uber' and 'lyft' in the New York metro area.")

    with st.spinner("Fetching Google Trends data…"):
        gtrends = load_google_trends()

    if gtrends is not None and not gtrends.empty:
        gt1, gt2 = st.columns(2)
        with gt1:
            fig_gt = go.Figure()
            if "uber" in gtrends.columns:
                fig_gt.add_trace(go.Scatter(
                    x=gtrends["week"], y=gtrends["uber"],
                    name="'uber'", line=dict(color=UBER_COLOR, width=2),
                ))
            if "lyft" in gtrends.columns:
                fig_gt.add_trace(go.Scatter(
                    x=gtrends["week"], y=gtrends["lyft"],
                    name="'lyft'", line=dict(color=LYFT_COLOR, width=2),
                ))
            fig_gt.update_layout(
                title=dict(text="Search Interest: 'uber' vs 'lyft' (NYC)", font_size=13, x=0, xanchor="left"),
                yaxis_title="Search Index (0–100)", xaxis_title=None,
                height=300, margin=dict(t=45,b=70,l=50,r=20),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font_size=11),
            )
            st.plotly_chart(fig_gt, use_container_width=True)

        with gt2:
            # Correlate weekly rides with search interest
            if "uber" in gtrends.columns:
                uber_wkly = to_weekly(daily_all)
                uber_wkly["week"] = pd.to_datetime(uber_wkly["week"])
                merged_gt = gtrends.merge(
                    uber_wkly[["week","trips"]], on="week", how="inner"
                ).dropna(subset=["uber","trips"])
                if len(merged_gt) > 5:
                    _, _, r_gt, _, _ = stats.linregress(merged_gt["uber"], merged_gt["trips"])
                    fig_gt2, _ = scatter_with_regression(
                        merged_gt["uber"].values, merged_gt["trips"].values,
                        "'uber' Search Index", "Weekly Rides",
                        title=f"Search Interest vs Actual Rides  (R²={r_gt**2:.2f})",
                    )
                    st.plotly_chart(fig_gt2, use_container_width=True)
    else:
        st.info(
            "Google Trends data unavailable right now (rate-limited). "
            "The chart will load automatically when the quota resets (usually within an hour). "
            "Search interest correlates with both brand awareness and real-time intent to book a ride."
        )

