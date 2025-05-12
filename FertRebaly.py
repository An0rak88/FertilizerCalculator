import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title=" Fertilizer Calculator", layout="wide")

# ─── Editable Defaults ─────────────────────────────────────────────────────
nutrient_df = pd.DataFrame({
    "name":   ["N","S","P","Ca","Mg","K","Na","B","Fe","Mn","Cu","Zn","Mo"],
    "target": [175, 95, 70, 130, 60, 210, 40, 0.5, 3.5, 0.5, 0.2, 0.2, 0.15],
    "initial":[192, 159, 45, 178, 106, 173, 27, 0.43, 0.98, 0.12, 0.14, 0.50, 0.19],
    "makeup": [  0,   0,   0,   0,   0,   0,  40,   0,    0,    0,    0,    0,    0]
}).set_index("name")

# ─── Constants ─────────────────────────────────────────────────────────────
POND_NAMES = [f"Pond {i}" for i in range(1, 8)]
DEFAULT_VOL = 394_099
DEFAULT_DILUTION = 0.50
MICRO_PRECISION  = 2
MACRO_PRECISION  = 2
FINAL_PRECISION  = 2  # New constant for final nutrient levels

FERTILISERS = [
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

MICRO_FERTS = [f for f in FERTILISERS if f["unit"] == "g"]
MACRO_FERTS = [f for f in FERTILISERS if f["unit"] == "kg"]

# ─── Helpers ───────────────────────────────────────────────────────────────
def show_table(title, df, precision):
    st.subheader(title)
    st.dataframe(df.round(precision), use_container_width=True)

def process_pond(df: pd.DataFrame, vol_L: float, dilution: float):
    micro_df, post = apply_micros(df, vol_L, dilution)
    macro_df, final_df = solve_macros(post, vol_L)

    # transpose + label
    micro = micro_df.T;  micro.index = ["g_added"]
    macro = macro_df.T;  macro.index = ["kg_added"]
    final = final_df[["final"]].T; final.index = ["Final (ppm)"]
    return micro, macro, final

# ─── Core Logic ────────────────────────────────────────────────────────────
@st.cache_data
def apply_micros(nutrients, vol_L, dilution, pond_name):  # Added pond_name for cache separation
    nutrients = nutrients.copy()
    nutrients["diluted"]    = nutrients["initial"]*(1-dilution) + nutrients["makeup"]*dilution
    nutrients["post_micro"] = nutrients["diluted"]

    records = []
    for fert in MICRO_FERTS:
        nut     = next(iter(fert["comp"]))
        deficit = max(0, nutrients.at[nut,"target"] - nutrients.at[nut,"diluted"])
        grams   = deficit * vol_L / (1000 * fert["comp"][nut])
        records.append({"Fertiliser": fert["name"], "g_added": grams})
        nutrients.at[nut,"post_micro"] = nutrients.at[nut,"target"]
        for side in ("S","Na"):
            if side in fert["comp"]:
                nutrients.at[side,"post_micro"] += grams * fert["comp"][side] * 1000 / vol_L

    return pd.DataFrame(records).set_index("Fertiliser"), nutrients

@st.cache_data
def solve_macros(nutrients, vol_L, pond_name):  # Added pond_name for cache separation
    nutrients = nutrients.copy()
    nutrients["final"] = nutrients["post_micro"]

    # build A & b
    ppm_per_kg = 1_000_000 / vol_L
    targets = [n for n in nutrients.index if nutrients.at[n,"target"] > 5]
    A, b = [], []
    for n in targets:
        deficit = max(0, nutrients.at[n,"target"] - nutrients.at[n,"post_micro"])
        b.append(deficit)
        A.append([fert["comp"].get(n,0) * ppm_per_kg for fert in MACRO_FERTS])

    res = lsq_linear(np.array(A), np.array(b), bounds=(0,1000))

    records = []
    for amt, fert in zip(res.x, MACRO_FERTS):
        if amt > 1e-6:
            records.append({"Fertiliser": fert["name"], "kg_added": amt})
            for nut, frac in fert["comp"].items():
                nutrients.at[nut,"final"] += amt * frac * ppm_per_kg

    return pd.DataFrame(records).set_index("Fertiliser"), nutrients

# ─── App Layout ────────────────────────────────────────────────────────────
st.title("🍃 Hydroponic Fertilizer Balancer")

# Initialize pond inputs and volumes
pond_volumes = {pond: DEFAULT_VOL for pond in POND_NAMES}

# Create main tabs
tab_main, tab_orange = st.tabs(["Main", "Orange"])

with tab_orange:
    st.subheader("Edit Nutrient Targets per Pond (ppm)")
    # build a DataFrame with one row per pond and one column per nutrient
    target_df = pd.DataFrame(
        {nutrient: [nutrient_df.at[nutrient, "target"]] * len(POND_NAMES)
         for nutrient in nutrient_df.index},
        index=POND_NAMES,
    )
    target_df.index.name = "Pond"

    # let the user edit target ppm for each pond
    edited_targets = st.data_editor(
        target_df.astype(float),
        use_container_width=True,
        key="edit_target_per_pond"
    )

    st.subheader("Pond Volumes (L)")
    # build a DataFrame with one row per pond
    volumes_df = pd.DataFrame(
        {"Volume (L)": [pond_volumes[p] for p in POND_NAMES]},
        index=POND_NAMES
    )
    volumes_df.index.name = "Pond"
    col_cfg = {"Volume (L)": st.column_config.Column("Volume (L)", disabled=True)}
    edited_volumes = st.data_editor(
        volumes_df,
        use_container_width=True,
        num_rows="fixed",
        key="edit_volumes",
        column_config=col_cfg
    )
    # write back to your dict
    for pond in POND_NAMES:
        pond_volumes[pond] = float(edited_volumes.loc[pond, "Volume (L)"])

with tab_main:
    # 1) dilution slider (keep at top)
    dilution = st.slider(
        "Dilution factor",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_DILUTION,
        help="Fraction of pond replaced by makeup water"
    )

    # 2) Initial ppm for all ponds
    st.subheader("Edit Initial Nutrient Levels (ppm)")
    init_df = pd.DataFrame(
        {nutrient: [nutrient_df.at[nutrient,"initial"]] * len(POND_NAMES)
         for nutrient in nutrient_df.index},
        index=POND_NAMES
    )
    init_df.index.name = "Pond"
    col_cfg = {c: st.column_config.Column(c, disabled=True) for c in init_df.columns}
    edited_init = st.data_editor(
        init_df,
        use_container_width=True,
        key="edit_init_multi",
        column_config=col_cfg
    )

    # 3) Makeup ppm for all ponds
    st.subheader("Edit Makeup Water Levels (ppm)")
    makeup_df = pd.DataFrame(
        {nutrient: [nutrient_df.at[nutrient,"makeup"]] * len(POND_NAMES)
         for nutrient in nutrient_df.index},
        index=POND_NAMES
    )
    makeup_df.index.name = "Pond"
    col_cfg = {c: st.column_config.Column(c, disabled=True) for c in makeup_df.columns}
    edited_makeup = st.data_editor(
        makeup_df,
        use_container_width=True,
        key="edit_makeup_multi",
        column_config=col_cfg
    )

    # Compute results for each pond
    results = {}
    for pond in POND_NAMES:
        # assemble a small df for this pond
        df = pd.DataFrame({
            "initial": edited_init.loc[pond],
            "makeup":  edited_makeup.loc[pond],
            "target":  edited_targets.loc[pond]      # Use pond-specific targets
        })
        df = df.clip(lower=0)
        vol = pond_volumes[pond]

        micro_df, post = apply_micros(df, vol, dilution, pond)
        macro_df, final_df = solve_macros(post, vol, pond)

        results[pond] = {
            "micro":  micro_df["g_added"],
            "macro":  macro_df["kg_added"],
            "final":  final_df["final"]
        }

    # Build summary tables
    micro_summary = pd.DataFrame.from_dict(
        {p: r["micro"] for p,r in results.items()}, orient="index"
    )
    macro_summary = pd.DataFrame.from_dict(
        {p: r["macro"] for p,r in results.items()}, orient="index"
    )
    final_summary = pd.DataFrame.from_dict(
        {p: r["final"] for p,r in results.items()}, orient="index"
    )

    # Label the row‐index column
    for df in (micro_summary, macro_summary, final_summary):
        df.index.name = "Pond"

    # Display results
    show_table("Micro-Fertilizer Additions", micro_summary, MICRO_PRECISION)
    show_table("Macro-Fertilizer Additions", macro_summary, MACRO_PRECISION)
    show_table("Final Nutrient Levels", final_summary, FINAL_PRECISION)
