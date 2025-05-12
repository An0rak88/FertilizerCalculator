import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

# ─── editable defaults ────────────────────────────────────────────────────────
nutrient_df = pd.DataFrame({
    "name":   ["N","S","P","Ca","Mg","K","Na","B","Fe","Mn","Cu","Zn","Mo"],
    "target": [300,  95,  70, 130,  60, 210,  40, 0.5,  3.5,  0.5,  0.2,  0.2,  0.15],
    "initial":[300,159,  45, 178, 106, 173,  27,0.43, 0.98, 0.12, 0.14, 0.50,  0.19],
    "makeup": [  0,   0,   0,   0,   0,   0,  40,   0,    0,    0,    0,    0,     0]
}).set_index("name")

VOL_L            = 394_099
DEFAULT_DILUTION = 0.50

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
# ───────────────────────────────────────────────────────────────────────────────

def apply_micros(df, ferts, vol_L, dilution):
    df = df.copy()
    df["diluted"]   = df["initial"]*(1-dilution) + df["makeup"]*dilution
    df["post_micro"] = df["diluted"].copy()
    micros = []
    for f in ferts:
        if f["unit"]=="g":
            nut = next(iter(f["comp"]))
            deficit = max(0, df.at[nut,"target"] - df.at[nut,"diluted"])
            g = deficit * vol_L / (1000*f["comp"][nut])
            micros.append({"Fertiliser":f["name"],"g_added":g})
            df.at[nut,"post_micro"] = df.at[nut,"target"]
            for side in ("S","Na"):
                if side in f["comp"]:
                    df.at[side,"post_micro"] += g*f["comp"][side]*1000/vol_L
    return pd.DataFrame(micros).set_index("Fertiliser"), df

def solve_macros(df, ferts, vol_L):
    df = df.copy()
    df["final"] = df["post_micro"].copy()
    macros = []
    macro_ferts = [f for f in ferts if f["unit"]=="kg"]
    targets = [n for n in df.index if df.at[n,"target"]>5]
    A, b = [], []
    for n in targets:
        deficit = max(0, df.at[n,"target"] - df.at[n,"post_micro"])
        b.append(deficit)
        A.append([f["comp"].get(n,0)*1_000_000/vol_L for f in macro_ferts])
    A, b = np.array(A), np.array(b)
    res = lsq_linear(A,b,bounds=(0,1000))
    for amt, f in zip(res.x, macro_ferts):
        if amt>1e-6:
            macros.append({"Fertiliser":f["name"],"kg_added":amt})
            for nut, frac in f["comp"].items():
                df.at[nut,"final"] += amt*frac*1_000_000/vol_L
    return pd.DataFrame(macros).set_index("Fertiliser"), df

st.set_page_config(page_title="Fertilizer Balancer", layout="wide")
st.title("🍃 Hydroponic Fertilizer Balancer")

# ─── sidebar inputs ────────────────────────────────────────────────────────────
dilution = st.sidebar.slider("Dilution factor", 0.0, 1.0, DEFAULT_DILUTION)
vol_L     = st.sidebar.number_input("Pond volume (L)", value=VOL_L, step=1000)
# ───────────────────────────────────────────────────────────────────────────────

inputs = nutrient_df.copy()

# ─── main page nutrient editors ──────────────────────────────────────────────
st.subheader("🔧 Edit Nutrient Targets")
target_df = inputs[["target"]].rename(columns={"target":"Target (ppm)"})
edited_targets = st.experimental_data_editor(target_df, num_rows="fixed")
inputs["target"] = edited_targets["Target (ppm)"]

st.subheader("🔧 Edit Nutrient Initial Values")
initial_df = inputs[["initial"]].rename(columns={"initial":"Initial (ppm)"})
edited_initial = st.experimental_data_editor(initial_df, num_rows="fixed")
inputs["initial"] = edited_initial["Initial (ppm)"]

st.subheader("🔧 Edit Nutrient Makeup Water")
makeup_df = inputs[["makeup"]].rename(columns={"makeup":"Makeup (ppm)"})
edited_makeup = st.experimental_data_editor(makeup_df, num_rows="fixed")
inputs["makeup"] = edited_makeup["Makeup (ppm)"]
# ───────────────────────────────────────────────────────────────────────────────

# ─── compute ───────────────────────────────────────────────────────────────────
micro_df, mid_df  = apply_micros(inputs, FERTILISERS, vol_L, dilution)
macro_df, final_df = solve_macros(mid_df, FERTILISERS, vol_L)

# ─── display ───────────────────────────────────────────────────────────────────
st.subheader("1️⃣ Micro-nutrient Additions")
st.dataframe(micro_df.round(2), use_container_width=True)

st.subheader("2️⃣ Macro-nutrient Additions")
st.dataframe(macro_df.round(3), use_container_width=True)

st.subheader("3️⃣ Nutrient Levels Through Each Stage")
st.dataframe(
    final_df[["initial","diluted","post_micro","final"]]
      .rename(columns={
          "initial":"Starting ppm",
          "diluted":"After dilution",
          "post_micro":"After micros",
          "final":"After macros"
      })
    .round(2),
    use_container_width=True,
)
