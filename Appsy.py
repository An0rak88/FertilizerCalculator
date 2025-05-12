import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import altair as alt

# --- Constants & Default Data ----------------------------------------------
NUTRIENTS = ["N", "S", "P", "Ca", "Mg", "K", "Na", "B", "Fe", "Mn", "Cu", "Zn", "Mo"]

default_target = [175, 95, 70, 130, 60, 210, 40, 0.5, 3.5, 0.5, 0.2, 0.2, 0.15]
default_actual = [133, 159, 45, 178, 106, 173, 27, 0.43, 0.98, 0.12, 0.14, 0.5, 0.19]

DEFAULT_FERTILIZERS = [
    {"name": "Calcium Nitrate",        "unit": "kg", "comp": {"Ca": 0.155, "N": 0.19}},
    {"name": "Potassium Sulfate",      "unit": "kg", "comp": {"K": 0.50, "S": 0.18}},
    {"name": "MonoPotassiumPhosphate", "unit": "kg", "comp": {"P": 0.2269, "K": 0.2822}},
    {"name": "Magnesium Sulfate",      "unit": "kg", "comp": {"Mg": 0.098, "S": 0.129}},
    {"name": "Ammonium Sulfate",       "unit": "kg", "comp": {"N": 0.21, "S": 0.24}},
    {"name": "Potassium Nitrate",      "unit": "kg", "comp": {"N": 0.137, "K": 0.384}},
    {"name": "Boron/Solubor",          "unit": "g",  "comp": {"B": 0.205}},
    {"name": "Iron Chelate",           "unit": "g",  "comp": {"Fe": 0.11}},
    {"name": "Manganese Sulfate",      "unit": "g",  "comp": {"Mn": 0.315, "S": 0.185}},
    {"name": "Sodium Molybdate",       "unit": "g",  "comp": {"Mo": 0.40, "Na": 0.14}},
    {"name": "Zinc Sulfate",           "unit": "g",  "comp": {"Zn": 0.355, "S": 0.175}},
    {"name": "Copper Sulfate",         "unit": "g",  "comp": {"Cu": 0.255, "S": 0.128}},
]

# --- Helper Functions ------------------------------------------------------
def format_number(v):
    return round(v) if abs(v) >= 10 else round(v, 2)

def format_percent(val):
    return f"{round(val * 100, 2)} %" if val > 0 else "0"

def smart_format(val):
    if isinstance(val, (int, float)):
        if abs(val) >= 1000:
            return f"{val:,.0f}"
        elif abs(val) >= 10:
            return f"{round(val):.0f}"
        else:
            return f"{val:.2f}"
    return val

def format_grams(val):
    try:
        val = float(val)
        if val > 1000:
            return f"{val / 1000:.2f} kg"
        else:
            return f"{round(val):,} g" if val >= 10 else f"{val:.2f} g"
    except:
        return val

@st.cache_data
def run_optimization(target_ppm, actual_ppm, fertilizers, vol_L, dilution):
    # 1) Apply fixed dilution
    target    = np.array(target_ppm, float)
    final_ppm = np.array(actual_ppm, float) * (1 - dilution)
    delta     = np.maximum(0, target - final_ppm)

    # 2) Process fertilizers by unit type
    fert_results = []
    
    # First handle all kg fertilizers (macros)
    for f in fertilizers:
        if f["unit"] == "kg":
            # Find primary nutrient (first non-S, non-Na nutrient)
            prim = next((n for n in f["comp"] if n not in {"S","Na"}), None)
            if prim:
                i = NUTRIENTS.index(prim)
                uf = 1000  # convert kg to g
                frac = f["comp"][prim]

                # grams needed to fill that one delta
                g_need = delta[i] * vol_L / (1000 * frac * uf)
                g_need = max(0, g_need)

                # record amount in grams
                fert_results.append({
                    "Fertilizer": f["name"],
                    "Amount (g)": round(g_need * uf, 2)
                })

                # subtract its contributions from all deltas
                for j, other in enumerate(NUTRIENTS):
                    other_frac = f["comp"].get(other, 0)
                    delta[j] -= other_frac * g_need * uf * 1000 / vol_L
                delta = np.maximum(0, delta)

                # update final_ppm for micros
                final_ppm = target - delta

    # 3) Then handle all g fertilizers (micros)
    S_idx = NUTRIENTS.index("S")
    for f in fertilizers:
        if f["unit"] == "g":
            prim = next((n for n in f["comp"] if n not in {"S","Na"}), None)
            if prim:
                i = NUTRIENTS.index(prim)
                uf = 1  # already in g
                frac = f["comp"][prim]

                req_ppm = max(0, target[i] - final_ppm[i])
                g_need = req_ppm * vol_L / (1000 * frac * uf)

                fert_results.append({
                    "Fertilizer": f["name"],
                    "Amount (g)": round(g_need * uf, 2)
                })

                # bump primary nutrient to target
                final_ppm[i] = target[i]

                # credit its small S knock-on
                if "S" in f["comp"]:
                    s_frac = f["comp"]["S"]
                    final_ppm[S_idx] += s_frac * g_need * uf * 1000 / vol_L

    # 4) Build summary and return
    tse = float(np.sum((target - final_ppm)**2))
    water_adj = {"Dilution fraction": dilution, "Total squared error": round(tse,4)}

    return pd.DataFrame(fert_results), water_adj, dict(zip(NUTRIENTS, np.round(final_ppm,2)))

# --- App Configuration -----------------------------------------------------
st.set_page_config(page_title="Hydroponic Fertilizer Balancer", layout="wide")

# Initialize ponds in session state
if 'ponds' not in st.session_state:
    st.session_state.ponds = [
        {
            'name': f'Pond {i}',
            'volume_L': 394099.0,      # (L)
            'target': default_target.copy(),
            'actual': default_actual.copy()
        }
        for i in range(1, 8)
    ]

# Define tabs
tab_main, tab_globals = st.tabs(["Dashboard", "Global Variables"])

# Sidebar: Basic Settings
with st.sidebar:
    st.header("Settings")

# Global Variables Tab
with tab_globals:
    st.header("Global Variables")

    st.subheader("Target Nutrient Levels (ppm)")
    target_df = pd.DataFrame(
        {p["name"]: p["target"] for p in st.session_state.ponds},
        index=NUTRIENTS,
    )
    edited_targets = st.data_editor(
        target_df,
        num_rows="fixed",
        key="all_pond_targets",
    ).reindex(NUTRIENTS)          # force canonical order

    for i, pond in enumerate(st.session_state.ponds):
        pond["target"] = edited_targets.loc[NUTRIENTS, pond["name"]].to_numpy()

    st.subheader("Fertilizer Composition %")
    comp_df = pd.DataFrame([
        {
            'Fertilizer': f['name'],
            **{nut: f['comp'].get(nut, 0) for nut in NUTRIENTS},
            'Unit': f['unit']
        }
        for f in DEFAULT_FERTILIZERS
    ])
    # Only show numeric columns for formatting
    numeric_cols = [nut for nut in NUTRIENTS]
    comp_df_numeric = comp_df[numeric_cols].astype(float) * 100  # show as percent
    comp_df_numeric = comp_df_numeric.round(2)
    styled_comp = (
        comp_df_numeric
        .style
        .format("{:.2f}")
        .set_properties(**{"text-align": "right"})
    )
    st.dataframe(styled_comp, use_container_width=True)

    fertilizers = []
    for _, row in comp_df.iterrows():
        comp = {
            nut: float(str(row[nut]).replace('%', '').strip()) / 100
            for nut in NUTRIENTS
            if str(row[nut]).replace('%', '').strip() and float(str(row[nut]).replace('%', '').strip()) > 0
        }
        fert = {'name': row['Fertilizer'], 'unit': row['Unit'], 'comp': comp}
        fertilizers.append(fert)
    st.session_state["fertilizers"] = fertilizers

    # --- Pond Management ---
    st.subheader("Manage Ponds")
    remove_indices = []
    for idx, pond in enumerate(st.session_state.ponds):
        cols = st.columns([2, 2, 1])
        pond['name'] = cols[0].text_input("Name", value=pond['name'], key=f"name_{idx}")
        pond["volume_L"] = cols[1].number_input(
            "Volume (L)", min_value=0.0, value=pond.get("volume_L",0.0),
            step=100.0, key=f"vol_{idx}"
        )
        if cols[2].button("🗑️", key=f"del_{idx}"):
            remove_indices.append(idx)
    for i in sorted(remove_indices, reverse=True):
        st.session_state.ponds.pop(i)

    st.markdown("Add a new pond")
    new_name = st.text_input("New Pond Name", key="new_pond_name")
    new_vol = st.number_input("New Pond Volume (L)", min_value=0.0, value=0.0, step=100.0, key="new_pond_vol")
    if st.button("Add Pond"):
        if new_name:
            st.session_state.ponds.append({
                'name': new_name,
                'volume_L': new_vol,
                'target': default_target.copy(),
                'actual': default_actual.copy()
            })
        else:
            st.warning("Please enter a pond name.")

# Main Dashboard Tab
with tab_main:
    st.title("Multi-Pond Hydroponic Fertilizer Balancer")

    dilution = st.slider(
        "Dilution (%)", min_value=0, max_value=15, value=10, step=1
    ) / 100
    st.caption(f"Note: Dilution set to {dilution*100:.0f}%")

    fertilizers = st.session_state.get("fertilizers", DEFAULT_FERTILIZERS)
    pond_names = [p["name"] for p in st.session_state.ponds]

    # 1) Nutrient input table
    st.subheader("Measured Nutrient Levels (ppm)")
    # 1) Build table with canonical nutrient order
    measured_df = pd.DataFrame(
        {pond["name"]: pond["actual"] for pond in st.session_state.ponds},
        index=NUTRIENTS
    )

    # 2) Display editor — no sorting or reordering allowed
    edited_measured = st.data_editor(
        measured_df,
        num_rows="fixed",
        key="measured_ppm",
        hide_index=False,
        column_order=[pond["name"] for pond in st.session_state.ponds]  # preserve pond order
    )

    # 3) Overwrite actuals using fixed row order
    for pond in st.session_state.ponds:
        pond["actual"] = edited_measured[pond["name"]].loc[NUTRIENTS].to_numpy()

    st.markdown("---")

    # 2) Fertilizer recommendations
    st.subheader("Calculated Fertilizer Additions")
    fert_names = [f["name"] for f in fertilizers]

    all_doses = []
    for pond in st.session_state.ponds:
        vol_L = pond["volume_L"]            # L
        fert_table, water_adj, _ = run_optimization(
            pond["target"], pond["actual"],
            fertilizers, vol_L, dilution
        )
        doses = fert_table.set_index("Fertilizer")["Amount (g)"].reindex(fert_names).fillna(0)
        all_doses.append(doses.values)

    # Store actual numbers for calculation, but format for display
    display_fert_df = pd.DataFrame(
        np.array(all_doses).T,
        index=fert_names,
        columns=pond_names,
    )
    display_fert_df = display_fert_df.applymap(format_grams)
    st.data_editor(
        display_fert_df,
        disabled=True,
        hide_index=False,
        num_rows="fixed",
        key="out_ferts"
    )

    # --- Final Nutrient Levels After Adjustment (ppm) ---
    st.subheader("Final Nutrient Levels (ppm) After Adjustment")

    final_ppm_dict = {}
    for pond in st.session_state.ponds:
        vol_L = pond["volume_L"]
        _, _, final_levels = run_optimization(
            pond["target"], pond["actual"],
            fertilizers, vol_L, dilution
        )
        final_ppm_dict[pond["name"]] = [final_levels[n] for n in NUTRIENTS]

    final_ppm_df = pd.DataFrame(final_ppm_dict, index=NUTRIENTS)
    final_ppm_df = final_ppm_df.astype(float).round(2)
    styled_final = (
        final_ppm_df
        .style
        .format("{:,}")
        .set_properties(**{"text-align": "right"})
    )
    st.dataframe(styled_final, use_container_width=True)

    st.markdown("---")
    st.subheader("Dilution Fractions & Errors")
    water_df = pd.DataFrame([
        run_optimization(p["target"], p["actual"], fertilizers, p["volume_L"], dilution)[1]
        for p in st.session_state.ponds
    ], index=pond_names)
    water_df = water_df.applymap(smart_format)
    st.dataframe(water_df)
