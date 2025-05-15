import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title=" Fertilizer Calculator", layout="wide")

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


# lab_df ---> lab_data
# pivot to get one row per pond, one column per nutrient
_lab_pivot = lab_df.pivot(index="pond", columns="nutrient", values="value")

# rename ponds "1"→"Pond 1", etc., then convert to dict-of-dicts
lab_data = {
    f"Pond {int(pond)}": {nut: float(val) for nut, val in nuts.items()}
    for pond, nuts in _lab_pivot.to_dict(orient="index").items()
}

# fertilizers_df ---> fertilizers
FERTILISERS = []
for _, r in fertilizers_df.iterrows():
    comp = { r["primary_element"]: float(r["primary_fraction"]) }
    if pd.notna(r["secondary_elem"]):
        comp[r["secondary_elem"]] = float(r["secondary_fraction"])

    FERTILISERS.append({
        "name": r["fert_name"],
        "comp": comp
    })

# targets_df ---> targets_lookup
targets_lookup = (
    targets_df
    .set_index("nutrient_name")["nutrient_target"]
    .astype(float)
    .to_dict()
)
# → targets_lookup["N"] == 192.0

# Default makeup water levels (ppm)
DEFAULT_MAKEUP = {
    "N": 0, "S": 0, "P": 0, "Ca": 0, "Mg": 0, "K": 0, "Na": 40,
    "B": 0, "Fe": 0, "Mn": 0, "Cu": 0, "Zn": 0, "Mo": 0
}

POND_NAMES = [f"Pond {i}" for i in range(1, 8)]
DEFAULT_VOL = 394_099
DEFAULT_DILUTION = 0.50
MICRO_PRECISION = 2
MACRO_PRECISION = 2
FINAL_PRECISION = 2





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
    for fert in FERTILISERS:
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
        A.append([fert["comp"].get(n,0) * ppm_per_kg for fert in FERTILISERS])

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
    # build a list of row-dicts using targets_lookup
    target_records = []
    for pond in POND_NAMES:
        row = {"Pond": pond}
        for nut, targ in targets_lookup.items():
            row[nut] = targ
        target_records.append(row)

    # let the user edit target ppm for each pond
    edited_targets = st.data_editor(
        target_records,
        use_container_width=True,
        key="edit_target_per_pond"
    )
    edited_targets = edited_targets.set_index("Pond")

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
    init_records = []
    for pond in POND_NAMES:
        row = {"Pond": pond}
        for nut, val in lab_data[pond].items():
            row[nut] = val
        init_records.append(row)

    edited_init = st.data_editor(
        init_records,
        use_container_width=True,
        key="edit_init_multi",
        column_config={
            c: st.column_config.Column(c, disabled=True)
            for c in init_records[0] if c != "Pond"
        }
    )
    edited_init = edited_init.set_index("Pond")

    # 3) Makeup ppm for all ponds
    st.subheader("Edit Makeup Water Levels (ppm)")

    makeup_records = []
    for pond in POND_NAMES:
        row = {"Pond": pond}
        for nut, val in DEFAULT_MAKEUP.items():
            row[nut] = val
        makeup_records.append(row)

    edited_makeup = st.data_editor(
        makeup_records,
        use_container_width=True,
        key="edit_makeup_multi",
        column_config={
            c: st.column_config.Column(c, disabled=False)
            for c in makeup_records[0] if c != "Pond"
        }
    )
    edited_makeup = edited_makeup.set_index("Pond")

    # Compute results for each pond
    results = {}
    for pond in POND_NAMES:
        # assemble a small df for this pond
        initial = edited_init.loc[pond].to_dict()
        makeup  = edited_makeup.loc[pond].to_dict()
        target  = edited_targets.loc[pond].to_dict()

        df = pd.DataFrame({
            "initial": initial,
            "makeup":  makeup,
            "target":  target
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
    # After you compute results[pond]["micro"], etc.
    for pond in POND_NAMES:
        results[pond]["micro"] = results[pond]["micro"].apply(
            lambda g: f"{g/1000:.2f} kg" if g > 1000 else f"{g:.1f} g"
    )
    # Build summary tables
    micro_summary = pd.DataFrame.from_dict(
        {p: r["micro"] for p, r in results.items()}, orient="index"
    )
    macro_summary = pd.DataFrame.from_dict(
        {p: r["macro"] for p, r in results.items()}, orient="index"
    )
    final_summary = pd.DataFrame.from_dict(
        {p: r["final"] for p, r in results.items()}, orient="index"
    )

    # Label the row‐index column
    for df in (micro_summary, macro_summary, final_summary):
        df.index.name = "Pond"

    # Display results
    show_table("Micro-Fertilizer Additions", micro_summary, MICRO_PRECISION)
    show_table("Macro-Fertilizer Additions", macro_summary, MACRO_PRECISION)
    show_table("Final Nutrient Levels", final_summary, FINAL_PRECISION)