import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# --------------------------------------------------------------------------- 
# constants & hard‑coded data
# --------------------------------------------------------------------------- 
vol_L    = 394099          # pond volume in litres
dilution = 0.10             # 10 % water change

NUTRIENT_NAMES = ["N","S","P","Ca","Mg","K","Na","B","Fe","Mn","Cu","Zn","Mo"]

# Create NUTRIENTS dictionary with all PPM values
NUTRIENTS = {
    name: {
        "name": name,
        "target_ppm": target,
        "initial_pond_ppm": init,
        "diluted_ppm": init * (1 - dilution),  # 10% dilution
        "post_micro_ppm": init * (1 - dilution),  # initialize with diluted value
        "final_ppm": init * (1 - dilution)
    }
    for name, target, init in zip(
        NUTRIENT_NAMES,
        [175, 95, 70, 130, 60, 210, 40, 0.5, 3.5, 0.5, 0.2, 0.2, 0.15],  # target values
        [133, 159, 45, 178, 106, 173, 27, 0.43, 0.98, 0.12, 0.14, 0.50, 0.19]  # initial values
    )
}


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

results = []

# ---------------------------------------------------------------------------
# helper – add the contribution of *all* nutrients in a fertiliser dose
# ---------------------------------------------------------------------------
def apply_fertiliser_ppm(fert, mass_kg):
    """
    Increment NUTRIENTS[n]['final_ppm'] for every nutrient present in fert['comp'].
    mass_kg : dose already expressed in kilograms.
    """
    for nutr, frac in fert["comp"].items():
        added_ppm = mass_kg * frac * 1000000 / vol_L   # g × (g‑nutr/g‑fert) × mg/g ÷ L
        NUTRIENTS[nutr]["final_ppm"] += added_ppm

# --------------------------------------------------------------------------- 
# Process micros
# --------------------------------------------------------------------------- 
def run_micros():
    """
    Execute the micro‑nutrient routine exactly as it exists now.
    Keeps NUTRIENTS[*]['post_micro_ppm'] and 'final_ppm' in sync.
    """
    for fert in FERTILISERS:
        if fert["unit"] == "g":
            primary_nutrient = next(iter(fert["comp"]))
            fertilizer_fraction = fert["comp"][primary_nutrient]
            deficit = max(0, NUTRIENTS[primary_nutrient]["target_ppm"]
                             - NUTRIENTS[primary_nutrient]["diluted_ppm"])
            results.append({
                "Fertiliser": fert["name"],
                "g": deficit * vol_L / (1000 * fertilizer_fraction)
            })
            NUTRIENTS[primary_nutrient]["post_micro_ppm"] = NUTRIENTS[primary_nutrient]["target_ppm"]
            NUTRIENTS[primary_nutrient]["final_ppm"]      = NUTRIENTS[primary_nutrient]["target_ppm"]

            # credit small S / Na contributions
            if "S" in fert["comp"]:
                NUTRIENTS["S"]["post_micro_ppm"] += deficit * fertilizer_fraction / fert["comp"]["S"]
            if "Na" in fert["comp"]:
                NUTRIENTS["Na"]["post_micro_ppm"] += deficit * fertilizer_fraction / fert["comp"]["Na"]

# --------------------------------------------------------------------------- 
# Process macros
# --------------------------------------------------------------------------- 
# >>> comment‑out the original "largest_deficit" macro loop
# ---------------------------------------------------------------------------
# simultaneous macro optimisation (NNLS) with guard‑rails
# ---------------------------------------------------------------------------

# realistic one‑pond upper bounds (kg) – adjust to suit your system
MAX_KG = {
    "Calcium Nitrate":        500.0,
    "Potassium Sulfate":      500.0,
    "MonoPotassiumPhosphate": 500.0,
    "Magnesium Sulfate":      500.0,
    "Potassium Nitrate":      500.0,
}

macro_ferts = [f for f in FERTILISERS if f["unit"] == "kg"]
ub          = np.array([MAX_KG.get(f["name"], np.inf) for f in macro_ferts])  # upper bounds

def solve_macros_nnls():
    """
    Build A·x ≈ deficit and solve non‑negative least squares with upper bounds ub.
    Returns list of (fert name, kg dose).
    """
    macro_targets = [n for n in NUTRIENT_NAMES if NUTRIENTS[n]["target_ppm"] > 5]
    A_rows, b = [], []

    for n in macro_targets:
        deficit = max(0, NUTRIENTS[n]["target_ppm"] - NUTRIENTS[n]["post_micro_ppm"])
        b.append(deficit)
        row = []
        for fert in macro_ferts:
            frac = fert["comp"].get(n, 0.0)
            row.append(frac * 1_000 * 1_000 / vol_L)  # ppm per kg
        A_rows.append(row)

    A = np.array(A_rows)
    b = np.array(b)

    res = lsq_linear(A, b, bounds=(0, ub))
    return [(f["name"], kg) for kg, f in zip(res.x, macro_ferts) if kg > 1e-6]

# ---------------------------------------------------------------------------
# evaluate dilutions 10–15 % and pick the plan with lowest total error
# ---------------------------------------------------------------------------
best_err, best_plan, best_dil = 1e9, None, None
print("\nOptimization steps:")
print("Dilution%  Total Error  Individual Differences (ppm)")
print("-" * 60)

for dil in np.linspace(0, dilution, 10):          # 10,11,…15 %
    # reset ppm tables for this candidate
    for n in NUTRIENT_NAMES:
        base = NUTRIENTS[n]["initial_pond_ppm"] * (1 - dil)
        NUTRIENTS[n]["diluted_ppm"]    = base
        NUTRIENTS[n]["post_micro_ppm"] = base
        NUTRIENTS[n]["final_ppm"]      = base

    results.clear()               # fresh list for this candidate

    run_micros()                  # micros routine (unchanged)

    macro_doses = solve_macros_nnls()      # NNLS macros
    for fert_name, kg in macro_doses:
        results.append({"Fertiliser": fert_name, "kg": kg})
        apply_fertiliser_ppm(next(f for f in FERTILISERS if f["name"] == fert_name), kg)

    # score absolute target error
    err = sum(abs(NUTRIENTS[n]["final_ppm"] - NUTRIENTS[n]["target_ppm"]) for n in NUTRIENT_NAMES)
    
    # Print differences for this step
    diffs = [abs(NUTRIENTS[n]["final_ppm"] - NUTRIENTS[n]["target_ppm"]) for n in NUTRIENT_NAMES]
    print(f"{dil*100:6.1f}%  {err:10.2f}  {' '.join(f'{d:6.2f}' for d in diffs)}")

    if err < best_err:
        best_err, best_plan, best_dil = err, list(results), dil

# keep the best plan for display
results = best_plan

# --------------------------------------------------------------------------- 
# display
# --------------------------------------------------------------------------- 
print(f"\nChosen dilution: {best_dil*100:.1f}%")
print("\nRecommended additions for one pond (grams & kg):")
df = pd.DataFrame(results).set_index("Fertiliser")
print(df.round(2))

print("\nNutrient levels (ppm):")
nutrient_data = []
for nutrient in NUTRIENT_NAMES:
    nutrient_data.append({
        "Nutrient": nutrient,
        "Target": NUTRIENTS[nutrient]["target_ppm"],
        "Initial": NUTRIENTS[nutrient]["initial_pond_ppm"],
        "Initial Diff": NUTRIENTS[nutrient]["initial_pond_ppm"] - NUTRIENTS[nutrient]["target_ppm"],
        "Diluted": NUTRIENTS[nutrient]["diluted_ppm"],
        "Post Micro": NUTRIENTS[nutrient]["post_micro_ppm"],
        "Final": NUTRIENTS[nutrient]["final_ppm"],
        "Difference": NUTRIENTS[nutrient]["final_ppm"] - NUTRIENTS[nutrient]["target_ppm"]
    })
df_nutrients = pd.DataFrame(nutrient_data).set_index("Nutrient")
print(df_nutrients.round(2))
print(f"\nInitial difference from target: {abs(df_nutrients['Initial'] - df_nutrients['Target']).sum():.2f} ppm")
print(f"Final difference from target: {df_nutrients['Difference'].abs().sum():.2f} ppm")