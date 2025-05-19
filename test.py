import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import pprint
import streamlit as st

def render_df(df: pd.DataFrame) -> str:
    macro_cols = ["K", "N", "Ca", "S", "P", "Mg", "Na"]
    micro_cols = ["Fe", "Mn", "B", "Cu", "Zn", "Mo"]

    df_fmt = df.copy()

    for col in df_fmt.columns:
        if df_fmt[col].dtype == object:
            df_fmt[col] = df_fmt[col].map(lambda x: f'<div style="text-align:left">{x}</div>' if pd.notna(x) else "")
        elif col in macro_cols:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        elif col in micro_cols:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        else:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    return df_fmt.to_html(classes="streamlit-table", index=False, escape=False)

st.set_page_config(page_title="Fertilizer Optimizer", layout="wide")
st.title("Hydroponic Fertilizer Optimizer")

# Manual refresh button
if st.button("🔄 Refresh Google Sheets Data"):
    st.experimental_rerun()

# Add CSS for table styling
st.markdown(
    """
    <style>
      .streamlit-table {
        width: 100% !important;
        border-collapse: collapse;
      }
      .streamlit-table th, .streamlit-table td {
        padding: 8px;
        border: 1px solid #ddd;
      }
      /* Default: right-align everything */
      .streamlit-table td {
        text-align: right;
      }
      .streamlit-table th {
        text-align: left;
      }
      /* Left-align the first column in all rows */
      .streamlit-table td:first-child {
        text-align: left !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

active_ponds = range(1, 8)

pond_volumes = {
    1: 150_000,
    2: 394_000,
    3: 394_000,
    4: 394_000,
    5: 394_000,
    6: 394_000,
    7: 500_000
}
# ─── Sheet URL ───────────────────────────────────────────────────────────
LAB_RESULTS_CSV      = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR4F_2PNEw2UjfDd6P9Ht3vdp3443d0Y8DCK1_WBhqV86OWdJ8i4T7tag8ojLA4Wzl31JYeAmdxc-vo/pub?gid=1894878774&single=true&output=csv"
FERTILIZERS_CSV      = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR4F_2PNEw2UjfDd6P9Ht3vdp3443d0Y8DCK1_WBhqV86OWdJ8i4T7tag8ojLA4Wzl31JYeAmdxc-vo/pub?gid=1305563386&single=true&output=csv"
NUTRIENT_TARGETS_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR4F_2PNEw2UjfDd6P9Ht3vdp3443d0Y8DCK1_WBhqV86OWdJ8i4T7tag8ojLA4Wzl31JYeAmdxc-vo/pub?gid=711444710&single=true&output=csv"

# ─── Pull data from Google Sheets ──────────────────────────────────────────
lab_df         = pd.read_csv(LAB_RESULTS_CSV)
lab_df.columns = lab_df.columns.str.strip()
fertilizers_df = pd.read_csv(FERTILIZERS_CSV)
fertilizers_df.columns = fertilizers_df.columns.str.strip()
targets_df     = pd.read_csv(NUTRIENT_TARGETS_CSV)
targets_df.columns = targets_df.columns.str.strip()

lab_results_flat = lab_df.to_dict(orient="records")
nutrient_names = {row["nutrient"] for row in lab_results_flat}

# Create a nested dictionary: {pond: {nutrient: value}}
lab_results = {}
for row in lab_results_flat:
    p = row.get("pond")
    nutrient = row.get("nutrient")
    result = row.get("result")
    if p not in lab_results:
        lab_results[p] = {}
    if nutrient not in lab_results[p]:
        lab_results[p][nutrient] = result

# For fast access by nutrient name, use:
targets = {row["nutrient"]: row for row in targets_df.to_dict(orient="records")}
micro_ppms = {name: targets[name] for name in nutrient_names if float(targets[name]["target"]) < 5}
macro_ppms = {name: targets[name] for name in nutrient_names if float(targets[name]["target"]) >= 5}

fertilizers = fertilizers_df.to_dict(orient="records")
# Set the micro fert name equal to the primary nutrient (micro nutrient name)
micro_ferts = {
    fert["primary_nutrient"]: {
        "fert_name": fert["primary_nutrient"],
        "fraction": fert["primary_fraction"]
    }
    for fert in fertilizers
    if fert["primary_nutrient"] in micro_ppms
}

# create macro_ferts before using it
macro_ferts = {
    fert["fert_name"]: {
        "primary_nutrient": fert["primary_nutrient"],
        "primary_fraction": fert["primary_fraction"],
        "secondary_nutrient": fert["secondary_nutrient"],
        "secondary_fraction": fert["secondary_fraction"]
    }
    for fert in fertilizers
    if fert["primary_nutrient"] in macro_ppms
}

makeup_water = [{"nutrient": name, "target": 0} for name in nutrient_names]

rebal_steps = {}
for p in active_ponds:
    rebal_steps[p] = {}
    for nutrient in nutrient_names:
        rebal_steps[p][nutrient] = {
            "init_ratio": 0.0,
            "post_dilution": 0.0,
            "deficit": 0.0,
            "post_micros": 0.0,
            "final": 0.0,
            "final_deficit": 0.0
        }

# Create a nested dictionary: {pond: {fertilizer_name: addition}}
results = {}
for p in active_ponds:
    results[p] = {}
    for f in fertilizers:
        results[p][f["fert_name"]] = {"addition": 0.0}
    results[p]["dilution"] = 0.0

# do calcs
for p in active_ponds:
    for n in macro_ppms:
        rebal_steps[p][n]["init_ratio"] = lab_results[p][n] / float(targets[n]["target"]) - 1

    results[p]["dilution"] = min(max(rebal_steps[p][n]["init_ratio"] for n in macro_ppms),.2)

    for n in nutrient_names:
        rebal_steps[p][n]["post_dilution"] = lab_results[p][n] * (1 - results[p]["dilution"])
        rebal_steps[p][n]["final"] = rebal_steps[p][n]["post_dilution"]

    for n in micro_ppms:
        rebal_steps[p][n]["deficit"] = max(0, targets[n]["target"] - rebal_steps[p][n]["post_dilution"])
        frac = micro_ferts[n]["fraction"]
        g = rebal_steps[p][n]["deficit"] * pond_volumes[p] / (1000 * frac)
        results[p][micro_ferts[n]["fert_name"]]["addition"] = g
        rebal_steps[p][n]["final"] += g * frac * 1_000 / pond_volumes[p]

    A = []
    b = []
    for f in macro_ferts:
        row = []
        for n in macro_ppms:
            if macro_ferts[f]["primary_nutrient"] == n:
                row.append(macro_ferts[f]["primary_fraction"]*1_000/pond_volumes[p])
            elif macro_ferts[f]["secondary_nutrient"] == n:
                row.append(macro_ferts[f]["secondary_fraction"]*1_000/pond_volumes[p])
            else:
                row.append(0.0)
        A.append(row)
    A = np.array(A)
    
    for n in macro_ppms:
        rebal_steps[p][n]["deficit"] = max(0, targets[n]["target"] - rebal_steps[p][n]["post_dilution"])
        b.append(rebal_steps[p][n]["deficit"])
    b = np.array(b)
    
    res = lsq_linear(A.T, b, bounds=(0, 1000000))

    # Store the results from the optimization into results[p][f]["addition"]
    for i, fert_name in enumerate(macro_ferts):
        results[p][fert_name]["addition"] = res.x[i]

     # Add ppms of primary and secondary nutrients contributed by each fertilizer to rebal_steps[p]
    for i, fert_name in enumerate(macro_ferts):

        primary_nutrient = macro_ferts[fert_name]["primary_nutrient"]
        secondary_nutrient = macro_ferts[fert_name]["secondary_nutrient"]

        # Convert grams to ppm: ppm = (grams * 1_000_000) / volume (mg/L)
        ppm_primary = (res.x[i] * macro_ferts[fert_name]["primary_fraction"] * 1_000) / pond_volumes[p]
        ppm_secondary = (res.x[i] * macro_ferts[fert_name]["secondary_fraction"] * 1_000) / pond_volumes[p]

        # Add the ppm contribution from this fertilizer to the 'final' value
        rebal_steps[p][primary_nutrient]["final"] += ppm_primary
        rebal_steps[p][secondary_nutrient]["final"] += ppm_secondary
    
    for n in nutrient_names:
        rebal_steps[p][n]["final_deficit"] = float(targets[n]["target"]) - rebal_steps[p][n]["final"]

# ─── Create Streamlit Tabs ────────────────────────────────────────────────
tabs = st.tabs(["Results", "Log", "Statics"])

# Statics Tab – Show raw CSVs
with tabs[2]:
    st.header("Statics")

    st.subheader("Fertilizers CSV")
    st.markdown(render_df(fertilizers_df), unsafe_allow_html=True)

    st.subheader("Nutrient Targets CSV")
    st.markdown(render_df(targets_df), unsafe_allow_html=True)

# Display results in the Results tab
with tabs[0]:
    # Display Fertilizer Additions Table in Streamlit
    pond_list = list(active_ponds)
    macro_fert_names = list(macro_ferts.keys())
    micro_fert_names = [micro_ferts[n]["fert_name"] for n in micro_ppms]

    table_rows = []

    # First: pond volumes and dilution
    table_rows.append({"Fertilizer": "Liters", **{f"Pond {p}": pond_volumes[p] for p in pond_list}})
    table_rows.append({"Fertilizer": "Dilution (%)", **{f"Pond {p}": int(round(results[p]["dilution"] * 100)) for p in pond_list}})

    # Macro in kg
    for fert_name in macro_fert_names:
        row = {"Fertilizer": f"{fert_name} (kg)"}
        for p in pond_list:
            addition = results[p][fert_name]["addition"] / 1000
            row[f"Pond {p}"] = int(round(addition))
        table_rows.append(row)

    # Micro in g
    for fert_name in micro_fert_names:
        row = {"Fertilizer": f"{fert_name} (g)"}
        for p in pond_list:
            addition = results[p][fert_name]["addition"]
            row[f"Pond {p}"] = int(round(addition))
        table_rows.append(row)

    st.subheader("Fertilizer Additions")
    additions_df = pd.DataFrame(table_rows)
    st.markdown(render_df(additions_df), unsafe_allow_html=True)

# Display nutrient balancing steps in the Log tab
with tabs[1]:
    st.header("Nutrient Balancing Steps")
    for p in active_ponds:
        # Display Nutrient Step Table in Streamlit
        sorted_nutrients = sorted(nutrient_names, key=lambda n: float(targets[n]['target']), reverse=True)

        table_data = []
        rows = [
            ("Target", lambda n: targets[n]['target']),
            ("LabRes", lambda n: lab_results[p][n]),
            ("InitRat (%)", lambda n: rebal_steps[p][n]["init_ratio"] * 100),
            ("PostDil", lambda n: rebal_steps[p][n]["post_dilution"]),
            ("Deficit", lambda n: rebal_steps[p][n]["deficit"]),
            ("Final", lambda n: rebal_steps[p][n]["final"]),
            ("Final Deficit", lambda n: rebal_steps[p][n]["final_deficit"])
        ]

        for row_name, func in rows:
            row = {"Step": row_name}
            for n in sorted_nutrients:
                val = func(n)
                row[n] = round(val, 2) if isinstance(val, float) else val
            table_data.append(row)

        st.subheader(f"Pond {p}")
        steps_df = pd.DataFrame(table_data)
        st.markdown(render_df(steps_df), unsafe_allow_html=True) 