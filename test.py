# ─── Imports ───────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
from st_aggrid import AgGrid, GridOptionsBuilder

# ─── Constants & Data ─────────────────────────────────
# Page Config
st.set_page_config(page_title=" Fertilizer Calculator", layout="wide")

# CSS Styling
combined_css = """
<style>
  /* HTML table styling */
  .styled-table { width:100% !important; border-collapse: separate; border-spacing:0; }
  .styled-table thead tr:first-child th:first-child { border-top-left-radius:6px; }
  .styled-table thead tr:first-child th:last-child  { border-top-right-radius:6px; }
  .styled-table tbody tr:last-child td:first-child { border-bottom-left-radius:6px; }
  .styled-table tbody tr:last-child td:last-child  { border-bottom-right-radius:6px; }
  .styled-table thead { background:#F5F5F5; font-weight:600; }
  .styled-table tbody tr:nth-child(even) { background:#FAFAFA; }
  .styled-table th, .styled-table td { padding:6px 12px; text-align:right; }
  .styled-table th:first-child, .styled-table td:first-child { text-align:left; }
</style>
"""
st.markdown(combined_css, unsafe_allow_html=True)

# ─── Pull initial & target from published Google Sheet ───────────────────────
sheet_url = (
    "https://docs.google.com/spreadsheets/d/"
    "e/2PACX-1vR4F_2PNEw2UjfDd6P9Ht3vdp3443d0Y8DCK1_WBhqV86OWdJ8i4T7tag8ojLA4Wzl31JYeAmdxc-vo/"
    "pub?gid=1894878774&output=csv"
)

@st.cache_data(ttl=3600)       # keep sheet in cache for 1 hour
def load_sheet(url):
    return pd.read_csv(url)

# ─── Manual refresh button ────────────────────────────────────────────────
if st.button("🔄 Refresh sheet data"):
    load_sheet.clear()

# actually load (cached unless just cleared)
df_sheet = load_sheet(sheet_url)

# capture the exact nutrient order as it appears in your sheet
nutrient_order = df_sheet["nutrient"].drop_duplicates().tolist()

# pivot targets: index=nutrient, columns=Pond N, values=value
target_df_sheet = (
    df_sheet[df_sheet["type"] == "target"]
    .pivot(index="nutrient", columns="pond", values="value")
    .reindex(nutrient_order)      # ← enforce your sheet's row order
)

# pivot initials: same shape, for lab readings
initial_df_sheet = (
    df_sheet[df_sheet["type"] == "initial"]
    .pivot(index="nutrient", columns="pond", values="value")
    .reindex(nutrient_order)
)

# Default Parameters
NUM_PONDS     = 7
DEFAULT_VOL   = 394_099
DEFAULT_DILUTE= 0.50
FINAL_PREC    = 2

# Default Data
fertiliser_data = [
    {"name":"Ca Nitrate",        "unit":"kg","Ca":0.155,  "N":0.19},
    {"name":"K Sulfate",      "unit":"kg","K":0.50,   "S":0.18},
    {"name":"K Phosphate", "unit":"kg","P":0.2269, "K":0.2822},
    {"name":"Mg Sulfate",      "unit":"kg","Mg":0.098,  "S":0.129},
    {"name":"K Nitrate",      "unit":"kg","N":0.137,  "K":0.384},
    {"name":"Ammonium Sulfate",       "unit":"kg","N":0.21,   "S":0.24},
    {"name":"Boron",          "unit":"g", "B":0.205},
    {"name":"Iron",           "unit":"g", "Fe":0.11},
    {"name":"Mn Sulfate",      "unit":"g", "Mn":0.315, "S":0.185},
    {"name":"Mo Molybdate",       "unit":"g", "Mo":0.40,  "Na":0.14},
    {"name":"Zn Sulfate",           "unit":"g", "Zn":0.355, "S":0.175},
    {"name":"Cu Sulfate",         "unit":"g", "Cu":0.255, "S":0.128},
]
fertiliser_df = (
    pd.DataFrame(fertiliser_data)
      .set_index("name")
      .fillna(0)     # missing comps → 0
)

# derive micro & macro lists directly from the DataFrame
MICRO_FERTS = fertiliser_df[fertiliser_df["unit"] == "g"]
MACRO_FERTS = fertiliser_df[fertiliser_df["unit"] == "kg"]

# Fertilizer name mapping
fertiliser_name_with_unit = {
    name: f"{name} ({row['unit']})"
    for name, row in fertiliser_df.iterrows()
}

# ─── Helpers & Core Logic ─────────────────────────────
def show_table(title: str, df: pd.DataFrame, precision: int = 0) -> None:
    """
    Render a read‑only table that looks like Streamlit's data_editor but
    lets us control number formatting.
    """
    st.subheader(title)

    # Move the index into a normal column called "Pond"
    df_to_show = df.reset_index().rename(columns={df.index.name or "index": "Pond"})

    # numeric formatting helper
    def fmt(x):
        try:
            x = float(x)
            if abs(x) >= 1_000:
                return f"{x:,.0f}" if precision == 0 else f"{x:,.{precision}f}"
            return f"{x:.0f}" if precision == 0 else f"{x:,.{precision}f}"
        except Exception:
            return x

    html = (
        df_to_show.style
        .format(fmt)
        .hide(axis="index")                              # kill the auto integer index
        .set_table_attributes('class="styled-table"')    # hook in our CSS class
        .to_html()
    )
    st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def apply_micros(nutrients, vol_L, dilution, pond_name):  # Added pond_name for cache separation
    nutrients = nutrients.copy()
    nutrients["diluted"]    = nutrients["initial"]*(1-dilution) + nutrients["makeup"]*dilution
    nutrients["post_micro"] = nutrients["diluted"]

    records = []
    for fert_name, fert in MICRO_FERTS.iterrows():
        # Find the first non-zero nutrient in this fertilizer
        nut = next((col for col in fert.index if col != "unit" and fert[col] > 0), None)
        if nut is None:
            continue
            
        deficit = max(0, nutrients.at[nut,"target"] - nutrients.at[nut,"diluted"])
        grams   = deficit * vol_L / (1000 * fert[nut])
        records.append({"Fertiliser": fert_name, "g_added": grams})
        nutrients.at[nut,"post_micro"] = nutrients.at[nut,"target"]
        
        # Handle side effects (S, Na)
        for side in ("S","Na"):
            if side in fert.index and fert[side] > 0:
                nutrients.at[side,"post_micro"] += grams * fert[side] * 1000 / vol_L

    out = pd.DataFrame(index=MICRO_FERTS.index)
    out["g_added"] = 0.0
    for r in records:
        out.at[r["Fertiliser"], "g_added"] = r["g_added"]
    return out, nutrients

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
        A.append([fert.get(n, 0) * ppm_per_kg for _, fert in MACRO_FERTS.iterrows()])

    res = lsq_linear(np.array(A), np.array(b), bounds=(0,1000))

    records = []
    for amt, (fert_name, fert) in zip(res.x, MACRO_FERTS.iterrows()):
        if amt > 1e-6:
            records.append({"Fertiliser": fert_name, "kg_added": amt})
            for nut in fert.index:
                if nut != "unit" and fert[nut] > 0:
                    nutrients.at[nut,"final"] += amt * fert[nut] * ppm_per_kg

    out = pd.DataFrame(index=MACRO_FERTS.index)
    out["kg_added"] = 0.0
    for r in records:
        out.at[r["Fertiliser"], "kg_added"] = r["kg_added"]
    return out, nutrients

def edit_grid(df: pd.DataFrame, key: str) -> pd.DataFrame:
    gb = GridOptionsBuilder.from_dataframe(df)

    # 1) Default column config (all but first col): editable, right-aligned, gray header
    gb.configure_default_column(
        editable=True,
        cellStyle={'textAlign': 'right'},
        headerStyle={
            'backgroundColor': '#F5F5F5',
            'fontWeight': '600'
        }
    )

    # 2) First column: left-aligned, gray background, non-editable
    first_col = df.columns[0]
    gb.configure_column(
        field=first_col,
        editable=False,
        cellStyle={
            'textAlign': 'left',
            'backgroundColor': '#F5F5F5'
        },
        headerStyle={
            'backgroundColor': '#F5F5F5',
            'fontWeight': '600'
        }
    )

    # 3) Layout and rendering
    # figure out how tall the grid needs to be:
    #  • header row ≈ 35px
    #  • each data row ≈ 30px
    n_rows = df.shape[0] + 1
    grid_height = 35 + n_rows * 30

    resp = AgGrid(
        df.reset_index(),  # make index a normal column
        gridOptions=gb.build(),
        theme="balham",
        enable_enterprise_modules=False,
        update_mode="MODEL_CHANGED",
        fit_columns_on_grid_load=True,
        height=grid_height,           # <- force the grid to be tall enough
        width="100%",
        key=key,
    )

    return pd.DataFrame(resp["data"]).set_index(df.index.name or "index")

# ─── Streamlit App Layout ─────────────────────────────
st.title("Fertilizer Calculator")

# Define POND_NAMES before tabs since it's used in both
num_ponds = NUM_PONDS
POND_NAMES = [f"Pond {i}" for i in range(1, num_ponds+1)]

# Initialize edited_targets from sheet data
target_df = target_df_sheet.copy()
target_df.columns = [c.split()[-1] for c in target_df.columns]
edited_targets = (
    target_df
    .reset_index()
    .set_index("nutrient")
    .astype(float)
    .T
    .rename(index=lambda c: f"Pond {c}")
)

# Initialize pond volumes and other default parameters
default_vol = DEFAULT_VOL
default_dilute = DEFAULT_DILUTE
final_prec = FINAL_PREC
pond_volumes = {pond: default_vol for pond in POND_NAMES}

# Initialize edited_volumes with default values
volumes_df = pd.DataFrame(
    {"Volume (L)": [pond_volumes[p] for p in POND_NAMES],
     "Dilution %": [default_dilute * 100] * len(POND_NAMES)},
    index=POND_NAMES,
)
volumes_t = volumes_df.T
volumes_t.index.name = "Parameter"
volumes_t.columns = [c.replace("Pond ", "") for c in volumes_t.columns]
edited_volumes = (
    volumes_t
    .reset_index()
    .set_index("Parameter")
    .astype(float)
    .T
    .rename(index=lambda c: f"Pond {c}")
)

# Create tabs, Calculator first
tab_calculator, tab_settings = st.tabs(["Calculator", "Settings"])

with tab_calculator:
    st.subheader("Lab Nutrient Levels (ppm)")

    # build lab_df off the sheet's "initial" pivots
    pond_rows = initial_df_sheet.T.copy()
    pond_rows.index.name = "Pond"

    # if you still need a single "Makeup Water" row from previous code,
    # you can leave it as-is—or remove entirely if you're not using "makeup":
    makeup_row = pd.DataFrame(
        {nut: [0] for nut in initial_df_sheet.index},
        index=["Makeup Water"]
    )

    lab_df = pd.concat([pond_rows, makeup_row])
    lab_df.index.name = "Pond"

    # 1) transpose
    lab_t = lab_df.T

    # 2) strip "Pond " prefix from column names
    lab_t.columns = [
      c.split()[-1] if c.startswith("Pond ") else c
      for c in lab_t.columns
    ]
    lab_t.index.name = "Nutrient"

    # 3) editable transposed table via Ag-Grid
    grid_lab_df      = lab_t.reset_index()               # "Nutrient" → column
    grid_lab_resp_df = edit_grid(grid_lab_df, key="lab_grid")
    edited_lab       = grid_lab_resp_df.set_index("Nutrient").astype(float)

    # Compute results for each pond
    results = {}
    for pond in POND_NAMES:
        col = pond.split()[-1]   # "1", "2", … "7"
        df = pd.DataFrame({
          "initial": edited_lab[col],
          "makeup":  edited_lab["Makeup Water"],
          "target":  edited_targets.loc[pond]
        })
        df = df.clip(lower=0)
        vol = pond_volumes[pond]
        dilution = np.clip(edited_volumes.loc[pond, "Dilution %"], 0, 100) / 100
        dilution_percent = dilution * 100
        liters_removed = vol * dilution

        micro_df, post = apply_micros(df, vol, dilution, pond)
        macro_df, final_df = solve_macros(post, vol, pond)

        results[pond] = {
            "dilution": dilution_percent,
            "removed_L": liters_removed,
            "micro":    micro_df["g_added"],
            "macro":    macro_df["kg_added"],
            "final":    final_df["final"]
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

    # Rename columns to include units
    macro_summary = macro_summary.rename(columns=fertiliser_name_with_unit)
    micro_summary = micro_summary.rename(columns=fertiliser_name_with_unit)

    # Display results in new order
    # Final Nutrient Levels (transposed)
    final_t = final_summary.T
    final_t.columns = [c.split()[-1] for c in final_t.columns]
    final_t.index.name = "Nutrient"
    show_table("Final Nutrient Levels (ppm)", final_t, final_prec)
    
    # Combine macro + micro (macros first), then transpose
    combined_fert = pd.concat([macro_summary, micro_summary], axis=1)
    dilution_series = pd.Series({p: r["dilution"] for p, r in results.items()}, name="Dilution (%)").round(1)
    combined_fert.insert(0, "Dilution (%)", dilution_series)
    removed_series = pd.Series({p: r["removed_L"] for p, r in results.items()}, name="Water Removed (l)").round(1)
    combined_fert.insert(1, "Water Removed (l)", removed_series)
    fert_t = combined_fert.T
    fert_t.columns = [c.split()[-1] for c in fert_t.columns]
    fert_t.index.name = "Fertiliser"
    show_table("Fertilizer Additions", fert_t, 0)

with tab_settings:
    st.subheader("Calculator Parameters")
    
    st.subheader("Nutrient Targets per Pond (ppm)")

    # feed it through AgGrid as before
    grid_df       = target_df.reset_index()
    grid_resp_df  = edit_grid(grid_df, key="nutrient_grid")
    edited_targets = (
        grid_resp_df
        .set_index("nutrient")
        .astype(float)
        .T
        .rename(index=lambda c: f"Pond {c}")
    )

    st.subheader("Pond Volumes and Dilution")
    grid_vol_df = volumes_t.reset_index()
    grid_vol_resp_df = edit_grid(grid_vol_df, key="volume_grid")
    edited_volumes = (
        grid_vol_resp_df
        .set_index("Parameter")
        .astype(float)
        .T
        .rename(index=lambda c: f"Pond {c}")
    )

    # Update pond volumes
    for pond in POND_NAMES:
        pond_volumes[pond] = edited_volumes.loc[pond, "Volume (L)"]

