# af_2d.py
import numpy as np
import pandas as pd


def _load_polar(file_path):
    df = pd.read_csv(file_path, delimiter=",", skiprows=10)
    df = df[["Alpha", "Cl", "Cd"]].copy()
    df["Alpha"] = pd.to_numeric(df["Alpha"], errors="coerce")
    df["Cl"] = pd.to_numeric(df["Cl"], errors="coerce")
    df["Cd"] = pd.to_numeric(df["Cd"], errors="coerce")
    return df.dropna(subset=["Alpha", "Cl", "Cd"]).sort_values("Alpha").reset_index(drop=True)


def get_C_l_0(file_path):
    df = _load_polar(file_path)

    zero_alpha = df.loc[np.isclose(df["Alpha"], 0.0), "Cl"]
    if not zero_alpha.empty:
        return float(zero_alpha.iloc[0])

    return float(np.interp(0.0, df["Alpha"].to_numpy(), df["Cl"].to_numpy()))


def get_C_d_0(file_path):
    df = _load_polar(file_path)

    zero_alpha = df.loc[np.isclose(df["Alpha"], 0.0), "Cd"]
    if not zero_alpha.empty:
        return float(zero_alpha.iloc[0])

    return float(np.interp(0.0, df["Alpha"].to_numpy(), df["Cd"].to_numpy()))


def get_C_l_vs_alpha(file_path, alpha_min=0.0, alpha_max=5.0):
    df = _load_polar(file_path)
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)

    subset = df[df["Alpha"].between(alpha_min, alpha_max)].sort_values("Alpha")
    if subset.empty:
        raise ValueError(f"No data found for alpha range {alpha_min} to {alpha_max} deg")

    c_l_1 = float(subset.iloc[0]["Cl"])
    c_l_2 = float(subset.iloc[-1]["Cl"])
    alpha_span_rad = np.deg2rad(alpha_max - alpha_min)
    if np.isclose(alpha_span_rad, 0.0):
        return 0.0
    return float((c_l_2 - c_l_1) / alpha_span_rad)


def get_C_d_vs_alpha(file_path, alpha_min=0.0, alpha_max=5.0):
    df = _load_polar(file_path)
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)

    subset = df[df["Alpha"].between(alpha_min, alpha_max)].sort_values("Alpha")
    if subset.empty:
        raise ValueError(f"No data found for alpha range {alpha_min} to {alpha_max} deg")

    c_d_1 = float(subset.iloc[0]["Cd"])
    c_d_2 = float(subset.iloc[-1]["Cd"])
    alpha_span_rad = np.deg2rad(alpha_max - alpha_min)
    if np.isclose(alpha_span_rad, 0.0):
        return 0.0
    return float((c_d_2 - c_d_1) / alpha_span_rad)


def get_alpha_l_0(file_path):
    df = _load_polar(file_path)
    cl = df["Cl"].to_numpy()
    alpha = df["Alpha"].to_numpy()

    for i in range(len(cl) - 1):
        if (cl[i] >= 0 and cl[i + 1] <= 0) or (cl[i] <= 0 and cl[i + 1] >= 0):
            return float(np.interp(0.0, [cl[i], cl[i + 1]], [alpha[i], alpha[i + 1]])) * np.pi / 180.0

    return float(np.interp(0.0, alpha, cl) * np.pi / 180.0) 


get_cl_0 = get_C_l_0
get_cd_0 = get_C_d_0
get_cl_vs_alpha = get_C_l_vs_alpha
get_cd_vs_alpha = get_C_d_vs_alpha
get_alpha_cl_0 = get_alpha_l_0
get_alpha_l0 = get_alpha_l_0