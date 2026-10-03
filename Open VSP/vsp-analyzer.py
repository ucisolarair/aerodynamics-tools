"""
Be warned: you must install the OpenVSP Python package to run this script. See the README in your OpenVSP installation for instructions.
You must also run this script in the "openvsp" virtual environment, which is created during the OpenVSP python installation.

Only runs on the python version with your OpenVSP installation (most likely 3.13). Good Luck!!
"""

import csv
import numpy as np
from pathlib import Path

import openvsp as vsp


MODEL_FILE = Path(__file__).resolve().parent / "vsp-files" / "twist-wing.vsp3"
AIRFOIL_DIR = Path(__file__).resolve().parent / "airfoil dats"
RESULTS_CSV = Path(__file__).resolve().parent / "airfoil_sweep_results.csv"

# Every listed root airfoil is paired with every listed tip airfoil.
ROOT_AIRFOILS = ["fx63137sm.dat", "s7055.dat", "mh32.dat", "ch10sm.dat"]
TIP_AIRFOILS = ["s7055.dat", "mh32.dat", "sd8020.dat"]

AOA_START = 4.0
AOA_END = 10.0
AOA_STEP = 0.25


def set_wing_airfoil(xsec_surf_id, xsec_index, airfoil_path):
    vsp.ChangeXSecShape(xsec_surf_id, xsec_index, vsp.XS_FILE_AIRFOIL)
    xsec_id = vsp.GetXSec(xsec_surf_id, xsec_index)
    vsp.ReadFileAirfoil(xsec_id, str(airfoil_path))


airfoil_pairs = [
    (root_airfoil, tip_airfoil)
    for root_airfoil in ROOT_AIRFOILS
    for tip_airfoil in TIP_AIRFOILS
]

if not airfoil_pairs:
    raise ValueError("Add at least one root and one tip airfoil filename.")

for airfoil_name in ROOT_AIRFOILS + TIP_AIRFOILS:
    airfoil_path = AIRFOIL_DIR / airfoil_name
    if not airfoil_path.is_file():
        raise FileNotFoundError(f"Airfoil file not found: {airfoil_path}")

num_points = round((AOA_END - AOA_START) / AOA_STEP) + 1
aoa_array = np.linspace(AOA_START, AOA_END, num_points)
csv_rows = []

for root_airfoil, tip_airfoil in airfoil_pairs:
    print(f"\n=== Root: {root_airfoil} | Tip: {tip_airfoil} ===")
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(MODEL_FILE))

    wing_ids = [
        geom_id
        for geom_id in vsp.FindGeoms()
        if vsp.GetGeomTypeName(geom_id) == "Wing"
    ]
    if len(wing_ids) != 1:
        raise RuntimeError(f"Expected one Wing geometry, found {len(wing_ids)}.")

    wing_id = wing_ids[0]
    xsec_surf_id = vsp.GetXSecSurf(wing_id, 0)
    xsec_count = vsp.GetNumXSec(xsec_surf_id)
    if xsec_count < 2:
        raise RuntimeError(f"Wing needs root and tip sections; found {xsec_count}.")

    set_wing_airfoil(xsec_surf_id, 0, AIRFOIL_DIR / root_airfoil)
    set_wing_airfoil(xsec_surf_id, xsec_count - 1, AIRFOIL_DIR / tip_airfoil)
    vsp.Update()

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaStart", [AOA_START])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaEnd", [AOA_END])
    vsp.SetIntAnalysisInput("VSPAEROSweep", "AlphaNpts", [num_points])
    vsp.SetIntAnalysisInput("VSPAEROSweep", "ThinGeomSet", [0])

    vsp.SetAnalysisInputDefaults("ParasiteDrag")
    vsp.SetIntAnalysisInput("ParasiteDrag", "GeomSet", [0])
    vsp.SetDoubleAnalysisInput(
        "ParasiteDrag",
        "Sref",
        [vsp.GetDoubleAnalysisInput("VSPAEROSweep", "Sref")[0]],
    )
    vsp.SetDoubleAnalysisInput(
        "ParasiteDrag",
        "Vinf",
        [vsp.GetDoubleAnalysisInput("VSPAEROSweep", "Vinf")[0]],
    )
    vsp.SetDoubleAnalysisInput(
        "ParasiteDrag",
        "Mach",
        [vsp.GetDoubleAnalysisInput("VSPAEROSweep", "MachStart")[0]],
    )
    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Altitude", [0.0])
    parasite_results = vsp.ExecAnalysis("ParasiteDrag")
    cd_parasitic = vsp.GetDoubleResults(parasite_results, "Total_CD_Total")[0]

    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [0])
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    log_file = MODEL_FILE.with_name(
        f"{MODEL_FILE.stem}_{Path(root_airfoil).stem}_root_"
        f"{Path(tip_airfoil).stem}_tip.vspaero.log"
    )
    vsp.SetStringAnalysisInput("VSPAEROSweep", "RedirectFile", [str(log_file)])
    vsp.ExecAnalysis("VSPAEROSweep")

    polar_results = vsp.FindLatestResultsID("VSPAERO_Polar")
    cl_values = vsp.GetDoubleResults(polar_results, "CLtot")
    cd_induced_values = vsp.GetDoubleResults(polar_results, "CDiw")

    best_aoa = None
    best_endurance = -1.0
    best_cl = 0.0
    best_cd_induced = 0.0
    best_cd_total = 0.0

    print(f"Parasite CDp: {cd_parasitic:.5f}")
    print(f"Solver log: {log_file}")
    print(
        f"{'AoA':<6} | {'CL':<8} | {'CDi':<8} | {'CDp':<8} | "
        f"{'CDtotal':<9} | {'CL^1.5/CDtotal':<16}"
    )
    print("-" * 76)

    for index, alpha in enumerate(aoa_array):
        cl = cl_values[index]
        cd_induced = cd_induced_values[index]
        cd_total = cd_induced + cd_parasitic
        endurance_factor = (cl ** 1.5) / cd_total if cl > 0 and cd_total > 0 else 0.0

        print(
            f"{alpha:6.2f} | {cl:8.4f} | {cd_induced:8.5f} | "
            f"{cd_parasitic:8.5f} | {cd_total:9.5f} | {endurance_factor:16.4f}"
        )

        if endurance_factor > best_endurance:
            best_aoa = alpha
            best_endurance = endurance_factor
            best_cl = cl
            best_cd_induced = cd_induced
            best_cd_total = cd_total

    if best_aoa is None:
        print("No positive-lift sweep points were found for this combination.")
    else:
        print(
            f"Optimal Metrics at {best_aoa:.2f} deg -> CL: {best_cl:.4f} | "
            f"CDi: {best_cd_induced:.5f} | CDp: {cd_parasitic:.5f} | "
            f"CDtotal: {best_cd_total:.5f} | Score: {best_endurance:.4f}"
        )
        csv_rows.append(
            [
                Path(root_airfoil).stem,
                Path(tip_airfoil).stem,
                float(best_aoa),
                float(best_cl),
                float(best_cd_induced),
                float(cd_parasitic),
                float(best_cd_total),
            ]
        )

write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
with RESULTS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(
            ["Root Airfoil", "Tip Airfoil", "AOA (deg)", "CL", "CDi", "CDp", "CD"]
        )
    writer.writerows(csv_rows)

print(f"Sweep results appended to {RESULTS_CSV}")