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


def get_C_l_vs_alpha(file_path, alpha_min=0, alpha_max=5.0):
    df = _load_polar(file_path)
    mask = df["Alpha"].between(alpha_min, alpha_max)
    return df.loc[mask, ["Alpha", "Cl"]].reset_index(drop=True)


def get_C_d_vs_alpha(file_path, alpha_min=0, alpha_max=5.0):
    df = _load_polar(file_path)
    mask = df["Alpha"].between(alpha_min, alpha_max)
    return df.loc[mask, ["Alpha", "Cd"]].reset_index(drop=True)


get_cl_0 = get_C_l_0
get_cd_0 = get_C_d_0
get_cl_vs_alpha = get_C_l_vs_alpha
get_cd_vs_alpha = get_C_d_vs_alpha