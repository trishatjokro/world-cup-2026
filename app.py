"""World Cup 2026 — Live ML Predictor & Squad Explorer (Streamlit dashboard).

Reads predictions.json (written by src/simulate.py) plus the data CSVs and
presents: title/knockout odds, group-winner probabilities, per-match W/D/L
forecasts, a country->squad->player drill-down, and a model-credibility tab.
Run:  streamlit run app.py
"""
import json
import os
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}
def fg(team):
    return f"{FLAGS.get(team, '')} {team}".strip()


@st.cache_data
def load():
    preds = json.load(open(os.path.join(HERE, "predictions.json")))
    groups = pd.read_csv(os.path.join(DATA, "groups.csv"))
    fixtures = pd.read_csv(os.path.join(DATA, "fixtures.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    return preds, groups, fixtures, players


def face_src(row):
    """Locally-served FIFA face if available (third-party CDNs get blocked by
    browser privacy filters), else a generated initials avatar."""
    f = row.get("face_file")
    if isinstance(f, str) and f:
        ap = os.path.join(HERE, f)
        if os.path.exists(ap):
            return ap
    return (f"https://ui-avatars.com/api/?name={quote(str(row['player']))}"
            f"&size=120&background=1f2937&color=fff&bold=true")


preds, groups, fixtures, players = load()
TEAMS = preds["teams"]
teams_df = pd.DataFrame(TEAMS).T.reset_index().rename(columns={"index": "team"})


# ----------------------------- header ----------------------------------------
@st.fragment(run_every="1s")
def live_clock(last_iso):
    """Ticking UTC clock + live 'time since last prediction' counter."""
    now = datetime.now(timezone.utc)
    st.metric("Live clock · UTC", now.strftime("%H:%M:%S"))
    if last_iso:
        try:
            s = int(max(0, (now - datetime.fromisoformat(last_iso)).total_seconds()))
            h, m, sec = s // 3600, (s % 3600) // 60, s % 60
            ago = (f"{h}h " if h else "") + (f"{m}m " if (m or h) else "") + f"{sec}s"
            st.caption(f"predictions updated **{ago}** ago")
        except ValueError:
            pass


st.title("🏆 World Cup 2026 — Live ML Predictor")
ts = preds.get("generated_ts")
when = datetime.fromisoformat(ts).strftime("%b %d, %Y %H:%M UTC") if ts else "—"
played = sum(1 for m in preds["matches"] if m.get("played"))
c1, c2, c3 = st.columns(3)
c1.metric("Simulations", f"{preds['n_sims']:,}")
c2.metric("Group matches played", f"{played} / 72")
with c3:
    live_clock(ts)
st.caption(f"Elo + Dixon–Coles match model → Monte-Carlo tournament simulation. "
           f"Last full simulation: **{when}**. "
           f"Probabilities, not predictions — favourites lose all the time.")

PAGES = ["🏆 Title & Knockouts", "📊 Groups", "⚽ Matches", "🌍 Squads & Players", "📈 Model credibility"]
PAGE_KEYS = ["title", "groups", "matches", "squads", "model"]
KEY2LABEL = dict(zip(PAGE_KEYS, PAGES))
LABEL2KEY = dict(zip(PAGES, PAGE_KEYS))


def _on_nav():
    st.query_params["page"] = LABEL2KEY[st.session_state.nav]
    st.query_params.pop("player", None)   # leaving deeper levels


# keep the radio in sync with the URL so browser back/forward works
_pk = st.query_params.get("page", "title")
if _pk not in PAGE_KEYS:
    _pk = "title"
if st.session_state.get("_pk") != _pk:
    st.session_state.nav = KEY2LABEL[_pk]
    st.session_state._pk = _pk

page = st.sidebar.radio("View", PAGES, key="nav", on_change=_on_nav)
st.sidebar.caption(f"Model ρ = {preds['model']['rho']} · host bonus +{preds['model']['home_bonus_hosts']:.0f} Elo")


def pct(x):
    return f"{x*100:.1f}%"


# ============================ TITLE & KNOCKOUTS ==============================
if page == PAGES[0]:
    st.subheader("Who wins the World Cup?")
    top = teams_df.sort_values("win_cup", ascending=False).head(24).iloc[::-1]
    fig = px.bar(top, x="win_cup", y=[fg(t) for t in top.team], orientation="h",
                 labels={"win_cup": "P(win cup)", "y": ""}, text=top["win_cup"].map(pct))
    fig.update_traces(marker_color="#e10600", textposition="outside")
    fig.update_layout(height=640, xaxis_tickformat=".0%", margin=dict(l=10, r=30, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Run to each stage")
    show = teams_df.sort_values("win_cup", ascending=False).copy()
    show["team"] = show["team"].map(fg)
    cols = ["team", "group", "elo", "qualify", "reach_r16", "reach_qf", "reach_sf",
            "reach_final", "win_cup"]
    disp = show[cols].rename(columns={
        "qualify": "Advance", "reach_r16": "R16", "reach_qf": "QF", "reach_sf": "SF",
        "reach_final": "Final", "win_cup": "Win Cup", "group": "Grp", "elo": "Elo"})
    for c in ["Advance", "R16", "QF", "SF", "Final", "Win Cup"]:
        disp[c] = disp[c].map(pct)
    st.dataframe(disp, width="stretch", hide_index=True, height=500)

# ================================ GROUPS =====================================
elif page == PAGES[1]:
    g = st.selectbox("Group", sorted(groups.group.unique()),
                     format_func=lambda x: f"Group {x}")
    gteams = teams_df[teams_df.group == g].sort_values("win_group", ascending=False)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### Group {g} — qualification odds")
        d = gteams.iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=[fg(t) for t in d.team], x=d.win_group, orientation="h",
                    name="Win group", marker_color="#e10600",
                    text=d.win_group.map(pct), textposition="outside")
        fig.add_bar(y=[fg(t) for t in d.team], x=d.qualify - d.win_group, orientation="h",
                    name="Advance (not 1st)", marker_color="#f5a3a0")
        fig.update_layout(barmode="stack", height=360, xaxis_tickformat=".0%",
                          margin=dict(l=10, r=20, t=10, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("### Stage odds")
        t = gteams[["team", "elo", "win_group", "qualify", "reach_qf", "win_cup"]].copy()
        t["team"] = t["team"].map(fg)
        for c in ["win_group", "qualify", "reach_qf", "win_cup"]:
            t[c] = t[c].map(pct)
        st.dataframe(t.rename(columns={"win_group": "Win grp", "qualify": "Advance",
                     "reach_qf": "QF", "win_cup": "Cup", "elo": "Elo"}),
                     width="stretch", hide_index=True)

    st.divider()
    st.markdown(f"### Group {g} fixtures")
    gm = [m for m in preds["matches"] if m["group"] == g]
    for m in gm:
        played = m.get("played")
        res = f"  ·  **{m['result'][0]}–{m['result'][1]}** ✅" if played else ""
        st.markdown(f"**{fg(m['home'])}** vs **{fg(m['away'])}** — {m['date']}{res}")
        b1, b2, b3 = st.columns(3)
        b1.progress(m["p_home"], text=f"{m['home']} {pct(m['p_home'])}")
        b2.progress(m["p_draw"], text=f"Draw {pct(m['p_draw'])}")
        b3.progress(m["p_away"], text=f"{m['away']} {pct(m['p_away'])}")

# ================================ MATCHES ====================================
elif page == PAGES[2]:
    st.subheader("Match forecasts (group stage)")
    ms = preds["matches"]
    gsel = st.selectbox("Filter by group", ["All"] + sorted({m["group"] for m in ms}),
                        format_func=lambda x: x if x == "All" else f"Group {x}")
    opts = [m for m in ms if gsel == "All" or m["group"] == gsel]
    labels = {m["match_id"]: f"{m['date']} · Grp {m['group']} · {fg(m['home'])} vs {fg(m['away'])}"
              for m in opts}
    mid = st.selectbox("Match", [m["match_id"] for m in opts],
                       format_func=lambda i: labels[i])
    m = next(x for x in ms if x["match_id"] == mid)

    st.markdown(f"## {fg(m['home'])} vs {fg(m['away'])}")
    st.caption(f"{m['date']} · {m['venue']} · Group {m['group']}")
    if m.get("played"):
        st.success(f"Final score: {m['home']} {m['result'][0]}–{m['result'][1]} {m['away']}")
    fig = go.Figure(go.Bar(
        x=[m["p_home"], m["p_draw"], m["p_away"]],
        y=[f"{m['home']} win", "Draw", f"{m['away']} win"],
        orientation="h", marker_color=["#e10600", "#9aa0a6", "#1a73e8"],
        text=[pct(m["p_home"]), pct(m["p_draw"]), pct(m["p_away"])], textposition="outside"))
    fig.update_layout(height=240, xaxis_tickformat=".0%", margin=dict(l=10, r=30, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    a, b, c = st.columns(3)
    a.metric("Expected goals", f"{m['exp_home']:.2f} – {m['exp_away']:.2f}")
    b.metric("Most likely score", f"{m['top_score'][0]}–{m['top_score'][1]}")
    c.metric("Draw probability", pct(m["p_draw"]))

# ============================ SQUADS & PLAYERS ===============================
elif page == PAGES[3]:
    countries = sorted(groups.team)
    # country lives in the URL (?country=) so browser back/forward works
    cqp = st.query_params.get("country", countries[0])
    if cqp not in countries:
        cqp = countries[0]
    if st.session_state.get("_cqp") != cqp:
        st.session_state.country_sel = cqp
        st.session_state._cqp = cqp

    def _on_country():
        st.query_params["country"] = st.session_state.country_sel
        st.query_params.pop("player", None)

    country = st.selectbox("Country", countries, key="country_sel",
                           format_func=fg, on_change=_on_country)

    info = TEAMS[country]
    st.markdown(f"## {fg(country)}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elo", f"{info['elo']:.0f}")
    m2.metric("Win group", pct(info["win_group"]))
    m3.metric("Advance", pct(info["qualify"]))
    m4.metric("Win cup", pct(info["win_cup"]))

    sq = players[players.country == country].copy()
    order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
    sq["po"] = sq.position.map(order).fillna(9)
    sq = sq.sort_values(["po", "shirt_number"]).reset_index(drop=True)
    # selected player lives in the URL (?player=) so back/forward works
    sel = st.query_params.get("player")

    # ---------------- player detail view ----------------
    if sel is not None and sel in set(sq.player):
        r = sq[sq.player == sel].iloc[0]
        if st.button("← Back to squad"):
            st.query_params.pop("player", None)
            st.rerun()
        L, R = st.columns([1, 2])
        with L:
            st.image(face_src(r), width=190)
            st.markdown(f"### {r.player}")
            st.caption(f"{'#'+str(int(r.shirt_number)) if pd.notna(r.shirt_number) else ''} · "
                       f"{r.position} · {r.club if isinstance(r.club, str) else '—'}")
        with R:
            a, b, c, d = st.columns(4)
            a.metric("Overall", int(r.overall) if pd.notna(r.overall) else "—")
            b.metric("Age", int(r.age) if pd.notna(r.age) else "—")
            c.metric("Caps", int(r.caps) if pd.notna(r.caps) else "—")
            d.metric("Position", r.position)
            if pd.notna(r.r1) and isinstance(r.radar_labels, str):
                labels = r.radar_labels.split("|")
                vals = [r.r1, r.r2, r.r3, r.r4, r.r5, r.r6]
                fig = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
                    line_color="#e10600"))
                fig.update_layout(height=330, margin=dict(l=30, r=30, t=20, b=20),
                                  polar=dict(radialaxis=dict(range=[0, 99], visible=True)))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No EA FC attributes for this player (not in the game / "
                        "post-2023 youngster). Bio shown above.")

    # ---------------- squad face grid ----------------
    else:
        matched = int(sq.has_photo.sum())
        st.markdown(f"### Squad ({len(sq)} players · {matched} with photos) — click a face")
        for pos, label in [("GK", "Goalkeepers"), ("DF", "Defenders"),
                           ("MF", "Midfielders"), ("FW", "Forwards")]:
            grp = sq[sq.position == pos]
            if grp.empty:
                continue
            st.markdown(f"**{label}**")
            cols = st.columns(6)
            for i, (_, r) in enumerate(grp.iterrows()):
                with cols[i % 6]:
                    st.image(face_src(r), width="stretch")
                    num = f"#{int(r.shirt_number)} " if pd.notna(r.shirt_number) else ""
                    if st.button(f"{num}{r.player}", key=f"p_{country}_{r.player}",
                                 width="stretch"):
                        st.query_params["player"] = r.player
                        st.rerun()
        st.caption("Real faces from EA Sports FC; players not in the game show an "
                   "initials avatar. Click any player for their attribute radar.")

# ============================ MODEL CREDIBILITY ==============================
elif page == PAGES[4]:
    st.subheader("How good is this model?")
    st.markdown(
        "Validated with a **leakage-free walk-forward backtest** (`src/backtest.py`): "
        "for each year, the model is fit only on prior matches, then scored on that "
        "year's held-out games. Metric of record is **RPS** (lower = better)."
    )
    bt = pd.DataFrame({
        "Test set": ["World Cup 2002–2022", "All competitive 2010–2022"],
        "n": [384, 8067], "Accuracy": ["54.7%", "61.2%"],
        "Log-loss": [0.978, 0.862], "RPS": [0.2010, 0.1699],
        "Base-rate RPS": [0.2375, 0.2307]})
    st.dataframe(bt, width="stretch", hide_index=True)

    st.markdown("#### Calibration (held-out World Cups) — predicted vs actual")
    cal = pd.DataFrame({
        "Model says": ["40%", "50%", "60%", "70%", "80%", "92%"],
        "Actually wins": ["41%", "50%", "61%", "71%", "79%", "91%"]})
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        st.dataframe(cal, width="stretch", hide_index=True)
    with cc2:
        fig = go.Figure()
        x = [40, 50, 60, 70, 80, 92]; y = [41, 50, 61, 71, 79, 91]
        fig.add_trace(go.Scatter(x=[30, 100], y=[30, 100], mode="lines",
                      line=dict(dash="dash", color="#9aa0a6"), name="perfect"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines",
                      marker=dict(size=10, color="#e10600"), name="model"))
        fig.update_layout(height=300, xaxis_title="predicted win %", yaxis_title="actual win %",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    st.success("When the model says 70%, the favourite wins ~70% of the time — it is "
               "**well-calibrated**, so the upsets you see are real, not noise.")
    st.caption("Realistic ceiling for international football is ~55–58% accuracy; "
               "draws (~22%) are the hardest class. The goal is calibration near the "
               "bookmakers', not high accuracy.")

st.sidebar.divider()
st.sidebar.caption("⚠️ Statistical model (Elo + Poisson), not betting advice.")
