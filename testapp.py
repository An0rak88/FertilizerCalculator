import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import streamlit as st

st.set_page_config(page_title="Hydroponic Fertilizer Balancer", layout="wide")
st.write("--- Script Rerun ---")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers for Streamlit rerun
# ────────────────────────────────────────────────────────────────────────────────
def _rerun():
    st.rerun()

# Helper function for universal number formatting for tables
def pretty_num(val):
    try:
        v = float(val)
        if abs(v) >= 10:
            return f"{v:,.0f}"
        else:
            return f"{v:.2f}"
    except:
        return val

# ────────────────────────────────────────────────────────────────────────────────
# Default pond settings
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_VOL_L       = 394099      # litres
DEFAULT_DILUTION    = 0.50         # fraction (50%)

# ---------------------------------------------------------------------------
# constants & hard‑coded data
# ---------------------------------------------------------------------------
NUTRIENT_ORDER = ["N","S","P","Ca","Mg","K","Na","B","Fe","Mn","Cu","Zn","Mo"]

# Initialize fertilizers in session state if not present
if "fertilizers" not in st.session_state:
    st.session_state.fertilizers = [
        {"name":"Calcium Nitrate",        "unit":"kg","comp":{"Ca":0.155,"N":0.19}},
        {"name":"Potassium Sulfate",      "unit":"kg","comp":{"K":0.50 ,"S":0.18}},
        {"name":"MonoPotassiumPhosphate", "unit":"kg","comp":{"P":0.2269,"K":0.2822}},
        {"name":"Magnesium Sulfate",      "unit":"kg","comp":{"Mg":0.098,"S":0.129}},
        {"name":"Potassium Nitrate",      "unit":"kg","comp":{"N":0.137,"K":0.384}},
        {"name":"Ammonium Sulfate",       "unit":"kg","comp":{"N":0.21 ,"S":0.24}},
        {"name":"Boron/Solubor",          "unit":"g", "comp":{"B":0.205}},
        {"name":"Iron Chelate",           "unit":"g", "comp":{"Fe":0.11}},
        {"name":"Manganese Sulfate",      "unit":"g", "comp":{"Mn":0.315,"S":0.185}},
        {"name":"Sodium Molybdate",       "unit":"g", "comp":{"Mo":0.40,"Na":0.14}},
        {"name":"Zinc Sulfate",           "unit":"g", "comp":{"Zn":0.355,"S":0.175}},
        {"name":"Copper Sulfate",         "unit":"g", "comp":{"Cu":0.255,"S":0.128}},
    ]

vol_L    = 394099          # default pond volume in litres
dilution = 0.50            # default dilution factor (50%)

# --- Session Initialization for Multiple Ponds ---
if "ponds" not in st.session_state:
    st.session_state.ponds = [
        {
            "name": f"Pond {i+1}",
            "volume_L": DEFAULT_VOL_L,
            "dilution": DEFAULT_DILUTION,
            "NUTRIENTS": {
                name: {
                    "name":             name,
                    "target_ppm":       tgt,
                    "initial_pond_ppm": act,
                    "makeup_water":     mw,
                    "diluted_ppm":      act * (1 - dilution) + mw * dilution,
                    "post_micro_ppm":   act * (1 - dilution) + mw * dilution,
                    "final_ppm":        act * (1 - dilution) + mw * dilution,
                }
                for name, tgt, act, mw in zip(
                    NUTRIENT_ORDER,
                    [175,  95, 70, 130,  60, 210,  40, 0.5,  3.5,  0.5,  0.2,  0.2,  0.15],  # target values
                    [133, 159, 45, 178, 106, 173,  27, 0.43,  0.98,  0.12,  0.14,  0.50,  0.19],  # initial values
                    [  0,   0,  0,   0,   0,   0,  40,    0,    0,    0,    0,    0,    0]   # makeup water
                )
            }
        }
        for i in range(7)
    ]

# ---------------------------------------------------------------------------
# Solver – integrates micro and macro routines
# ---------------------------------------------------------------------------
def run_solver(pond):
    try:
        NUTRIENTS = pond["NUTRIENTS"]
        vol_L     = pond["volume_L"]
        dilution  = pond["dilution"]

        # refresh dilution‑based values
        for n in NUTRIENT_ORDER:
            d = NUTRIENTS[n]
            d["diluted_ppm"]  = d["initial_pond_ppm"] * (1 - dilution) + d["makeup_water"] * dilution
            d["post_micro_ppm"] = d["diluted_ppm"]     # reset pipeline
            d["final_ppm"]      = d["diluted_ppm"]

        results = []

        # --- Process micros (emit on "kg" so downstream sees them) ---
        for fert in st.session_state.fertilizers:
            if fert["unit"] == "g":
                primary = next(iter(fert["comp"]))
                frac    = fert["comp"][primary]
                deficit = max(0, NUTRIENTS[primary]["target_ppm"] - NUTRIENTS[primary]["diluted_ppm"])

                # compute gram dose, then convert to kg
                g_dose  = deficit * vol_L / (1000 * frac)
                kg_dose = g_dose / 1000.0

                # append on "kg" so your existing all_doses pipeline picks it up
                results.append({
                    "Fertiliser": fert["name"],
                    "kg":         kg_dose
                })

                # update NUTRIENTS just as before
                NUTRIENTS[primary]["post_micro_ppm"] = NUTRIENTS[primary]["target_ppm"]
                NUTRIENTS[primary]["final_ppm"]      = NUTRIENTS[primary]["target_ppm"]
                
                # Fix: Calculate secondary nutrient contributions correctly
                if "S" in fert["comp"]:
                    s_frac = fert["comp"]["S"]
                    s_contribution = kg_dose * s_frac * 1_000_000 / vol_L
                    NUTRIENTS["S"]["post_micro_ppm"] += s_contribution
                    NUTRIENTS["S"]["final_ppm"] += s_contribution
                
                if "Na" in fert["comp"]:
                    na_frac = fert["comp"]["Na"]
                    na_contribution = kg_dose * na_frac * 1_000_000 / vol_L
                    NUTRIENTS["Na"]["post_micro_ppm"] += na_contribution
                    NUTRIENTS["Na"]["final_ppm"] += na_contribution

        # --- Process macros ---
        macro_ferts   = [f for f in st.session_state.fertilizers if f["unit"] == "kg"]
        macro_targets = [n for n, d in NUTRIENTS.items() if d["target_ppm"] > 5]

        A_rows, b = [], []
        for n in macro_targets:
            deficit = max(0, NUTRIENTS[n]["target_ppm"] - NUTRIENTS[n]["post_micro_ppm"])
            b.append(deficit)
            A_rows.append([
                f["comp"].get(n, 0.0) * 1_000_000 / vol_L
                for f in macro_ferts
            ])

        A = np.array(A_rows)
        b = np.array(b)
        res = lsq_linear(A, b, bounds=(0, 1000))
        doses = [(f["name"], kg) for kg, f in zip(res.x, macro_ferts) if kg > 1e-6]

        for name, kg in doses:
            fert = next(f for f in st.session_state.fertilizers if f["name"] == name)
            for nutr, frac in fert["comp"].items():
                NUTRIENTS[nutr]["final_ppm"] += kg * frac * 1_000_000 / vol_L
            results.append({"Fertiliser": name, "kg": kg})

        # --- Build nutrient DataFrame ---
        df_nutrients = pd.DataFrame(
            {
                "Target":     [d["target_ppm"]      for d in NUTRIENTS.values()],
                "Initial":    [d["initial_pond_ppm"] for d in NUTRIENTS.values()],
                "Diluted":    [d["diluted_ppm"]     for d in NUTRIENTS.values()],
                "Post Micro": [d["post_micro_ppm"]  for d in NUTRIENTS.values()],
                "Final":      [d["final_ppm"]       for d in NUTRIENTS.values()],
            },
            index=[d["name"] for d in NUTRIENTS.values()]
        )

        return results, df_nutrients

    except Exception as e:
        st.error(f"Error in solver for pond {pond.get('name','Unknown')}: {str(e)}")
        return [], pd.DataFrame()

# Create tabs
tab_main, tab_globals = st.tabs(["Dashboard", "Global Variables"])

# Main Dashboard Tab
with tab_main:
    st.title("Multi-Pond Fertilizer Balancer")

    # Debug logging for data flow
    st.write("Debug: Current fertilizer compositions:")
    st.json({f["name"]: f["comp"] for f in st.session_state.fertilizers})

    # Dilution slider at the top of the dashboard (now up to 100%)
    user_dilution = st.slider(
        "Dilution (%)",
        min_value=0,
        max_value=100,
        value=int(DEFAULT_DILUTION * 100),
        step=1
    ) / 100
    st.caption(f"Note: Dilution set to {user_dilution*100:.0f}%")

    # Update ponds' dilution
    for pond in st.session_state.ponds:
        pond["dilution"] = user_dilution

    if not st.session_state.ponds:
        st.error("No ponds available. Please check the Global Variables tab.")
    else:
        fert_names = [f["name"] for f in st.session_state.fertilizers]
        pond_names = [p["name"] for p in st.session_state.ponds]
        pond_results = []

        # === Measured Nutrient Levels (editable) ===
        measured_data = np.array([
            [p["NUTRIENTS"][n]["initial_pond_ppm"] for n in NUTRIENT_ORDER]
            for p in st.session_state.ponds
        ]).T

        measured_df = pd.DataFrame(measured_data,
                                   index=NUTRIENT_ORDER,
                                   columns=[p["name"] for p in st.session_state.ponds]).astype(float)

        edited_measured = st.data_editor(
            measured_df,
            disabled=False,                 # ← allow editing
            hide_index=False,
            num_rows="fixed",
            key="measured_nutrients_dashboard",
            height=35 * (len(NUTRIENT_ORDER) + 1)
        )

        # Debug logging for measured values
        st.write("Debug: Edited measured values:")
        st.dataframe(edited_measured)

        # push edits back into session‑state BEFORE running the solver
        for pond in st.session_state.ponds:
            for n in NUTRIENT_ORDER:
                pond["NUTRIENTS"][n]["initial_pond_ppm"] = float(
                    edited_measured.at[n, pond["name"]]
                )

        # Debug logging for session state after update
        st.write("Debug: Session state after update:")
        st.json({p["name"]: {n: p["NUTRIENTS"][n]["initial_pond_ppm"] for n in NUTRIENT_ORDER} for p in st.session_state.ponds})

        # Collect results for all ponds
        for pond in st.session_state.ponds:
            plan, df_nutrients = run_solver(pond)
            pond_results.append((plan, df_nutrients))

        # Debug logging for solver results
        st.write("Debug: Solver results for first pond:")
        if pond_results:
            st.json(pond_results[0][0])  # Show plan
            st.dataframe(pond_results[0][1])  # Show nutrient levels

        # === Consolidated Fertilizer Recommendations ===
        all_doses = []
        for plan, _ in pond_results:
            df = pd.DataFrame(plan).set_index("Fertiliser")
            all_doses.append(df.reindex(fert_names).fillna(0)["kg"].values)

        # Build unit map for each fertilizer
        unit_map = {f["name"]: f["unit"] for f in st.session_state.fertilizers}

        # raw doses in kg
        raw_df = pd.DataFrame(
            np.array(all_doses).T,
            index=fert_names,
            columns=pond_names,
        )

        # now format micros in grams, macros in kg
        formatted_df = pd.DataFrame(index=fert_names, columns=pond_names)
        for fert in fert_names:
            for pond in pond_names:
                dose_kg = raw_df.at[fert, pond]
                if unit_map[fert] == "g":
                    g_amt = dose_kg * 1000
                    formatted_df.at[fert, pond] = f"{pretty_num(g_amt)} g"
                else:
                    formatted_df.at[fert, pond] = f"{pretty_num(dose_kg)} kg"

        fert_table_height = 35 * (len(fert_names) + 1)

        st.subheader("Calculated Fertilizer Additions (micros in g, macros in kg)")
        st.data_editor(
            formatted_df,
            disabled=True,
            hide_index=False,
            num_rows="fixed",
            key="consolidated_ferts",
            height=fert_table_height
        )

        # === Consolidated Final Nutrient Levels After Adjustment ===
        nutrient_tables = []
        for _, df_nutrients in pond_results:
            nutrient_tables.append(df_nutrients["Final"].values)
        final_nutrients_df = pd.DataFrame(
            np.array(nutrient_tables).T,
            index=NUTRIENT_ORDER,
            columns=pond_names,
        ).astype(float)
        final_nutrients_df = final_nutrients_df.round(2)
        final_nutrients_table_height = 35 * (len(NUTRIENT_ORDER) + 1)

        st.subheader("Final Nutrient Levels (ppm) After Adjustment")
        st.data_editor(
            final_nutrients_df,
            disabled=True,
            hide_index=False,
            num_rows="fixed",
            key="consolidated_nutrients",
            height=final_nutrients_table_height
        )

        # === Total Initial and Final Difference Table ===
        total_initial_diff = [
            abs(df["Initial"] - df["Target"]).sum()
            for _, df in pond_results
        ]
        total_final_diff = [
            abs(df["Final"] - df["Target"]).sum()
            for _, df in pond_results
        ]
        diff_df = pd.DataFrame(
            [total_initial_diff, total_final_diff],
            index=["Initial Difference", "Final Difference"],
            columns=pond_names
        )
        diff_df = diff_df.round(1)
        diff_table_height = 35 * (len(diff_df) + 1)
        st.subheader("Total Difference from Target (ppm)")
        st.data_editor(
            diff_df,
            disabled=True,
            hide_index=False,
            num_rows="fixed",
            key="total_diff_dashboard",
            height=diff_table_height
        )

# Global Variables Tab
with tab_globals:
    st.header("Global Variables")

    # === Target Nutrient Levels Table (Editable) ===
    target_data = np.array([
        [p["NUTRIENTS"][n]["target_ppm"] for p in st.session_state.ponds]
        for n in NUTRIENT_ORDER
    ])
    target_df = pd.DataFrame(
        target_data,
        index=NUTRIENT_ORDER,
        columns=[p["name"] for p in st.session_state.ponds]
    ).astype(float)
    target_df = target_df.round(0).astype(int)
    target_table_height = 35 * (len(NUTRIENT_ORDER) + 1)
    st.subheader("Target Nutrient Levels (ppm)")
    edited_targets = st.data_editor(
        target_df,
        num_rows="fixed",
        key="target_editor",
        height=target_table_height
    ).reindex(NUTRIENT_ORDER)
    for pond in st.session_state.ponds:
        for n in NUTRIENT_ORDER:
            pond["NUTRIENTS"][n]["target_ppm"] = edited_targets.at[n, pond["name"]]

    # --- Fertilizer Composition Table ---
    st.subheader("Fertilizer Composition (% by nutrient)")
    fert_names = [f["name"] for f in st.session_state.fertilizers]
    fert_comp_data = []
    for fert in st.session_state.fertilizers:
        row = {nutr: (100*fert["comp"].get(nutr, 0)) for nutr in NUTRIENT_ORDER}
        fert_comp_data.append(row)
    fert_comp_df = pd.DataFrame(fert_comp_data, index=fert_names)
    fert_comp_df = fert_comp_df[[n for n in NUTRIENT_ORDER]]  # ensure column order
    fert_comp_df = fert_comp_df.replace(0, "")  # blank for zero
    fert_comp_df = fert_comp_df.round(1)
    
    # Make the table editable
    edited_fert_comp = st.data_editor(
        fert_comp_df,
        disabled=False,  # Allow editing
        hide_index=False,
        num_rows="fixed",
        key="fertilizer_composition_table",
        height=35 * (len(fert_names) + 1)
    )
    
    # Update session state with edited values
    for i, fert in enumerate(st.session_state.fertilizers):
        for nutr in NUTRIENT_ORDER:
            val = edited_fert_comp.at[fert["name"], nutr]
            if val != "":
                # Convert percentage back to fraction
                st.session_state.fertilizers[i]["comp"][nutr] = float(val) / 100
            else:
                # Remove nutrient if value is empty
                st.session_state.fertilizers[i]["comp"].pop(nutr, None)

    st.subheader("Manage Ponds")
    for i, pond in enumerate(st.session_state.ponds):
        cols = st.columns([2, 2])
        pond["name"] = cols[0].text_input(
            "Pond Name", pond["name"], key=f"name_{i}"
        )
        pond["volume_L"] = cols[1].number_input(
            "Volume (L)",
            value=float(pond["volume_L"]),
            step=100.0,
            key=f"vol_{i}"
        )