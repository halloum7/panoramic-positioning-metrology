"""
pano_calibrated.py — Simulateur d'erreurs de positionnement pilote en
grandeurs physiques (Article 7). Surcouche de pano_artifacts.py (Article 6) :
le simulateur d'origine n'est pas modifie, on convertit seulement les
grandeurs physiques en ses parametres relatifs.

Grandeurs physiques et conversion
---------------------------------
* delta_mm  (deplacement antero-posterieur, + = trop en avant)
      s = -delta / (x0 + delta)                 [h = 1 + s = Mh(delta)/M0]
      + correction verticale Mv(delta)/M0 dans la zone incisive
      + flou horizontal sigma = sqrt(s0^2 + (kappa*delta)^2) (mm objet -> px)
  -> NECESSITE x0 (et kappa, s0) issus de l'essai sur fantome.
* chin_deg  (bascule de la tete autour d'un axe transverse, + = menton bas)
      k = d_ap * tan(theta) / FIELD_H_MM
* tilt_deg  (inclinaison autour de l'axe antero-posterieur, + = cote droit de
  l'image plus BAS)   t = L * tan(phi) / FIELD_H_MM
  chin et tilt ne dependent PAS de x0 : ils sont utilisables tout de suite.

FIELD_H_MM = 1536 lignes x 0,078156 mm = 120,0 mm (hauteur du champ dans le
plan de la couche focale, lue dans les DICOM Owandy). Pour une image
redimensionnee, la conversion reste valable (elle est relative a H).

Hypotheses anatomiques (a citer et a faire varier en analyse de sensibilite) :
  d_ap = 40 mm : distance sagittale incisives -> region molaire (|x-xc| = 0,3 W)
  L    = 100 mm : distance transversale entre les bords gauche et droit de
                  l'image (rami), sur laquelle porte le denivele t
"""
from __future__ import annotations
import math
import numpy as np
import cv2

import pano_artifacts as pa
from pano_geometry import SID_MM, M0

FIELD_H_MM = 1536 * 0.078156      # 120.05 mm
OBJ_PIX_FULL = 0.078156           # mm/pixel objet a pleine resolution (H = 1536)

# Valeurs PROVISOIRES, a remplacer par le resultat de phantom_analysis.py
CALIB = dict(x0_mm=None, blur_sigma0_mm=0.0, blur_kappa_mm_per_mm=0.05)


def set_calibration(x0_mm, kappa=None, sigma0=None):
    CALIB["x0_mm"] = float(x0_mm)
    if kappa is not None:
        CALIB["blur_kappa_mm_per_mm"] = float(kappa)
    if sigma0 is not None:
        CALIB["blur_sigma0_mm"] = float(sigma0)


def _obj_pix_mm(h):
    return FIELD_H_MM / h


# ------------------------------------------------------------ conversions
def s_from_delta(delta_mm, x0_mm):
    return -delta_mm / (x0_mm + delta_mm)


def delta_from_s(s, x0_mm):
    return -s * x0_mm / (1.0 + s)


def k_from_chin(theta_deg, d_ap_mm=40.0):
    return d_ap_mm * math.tan(math.radians(abs(theta_deg))) / FIELD_H_MM


def chin_from_k(k, d_ap_mm=40.0):
    return math.degrees(math.atan(k * FIELD_H_MM / d_ap_mm))


def t_from_tilt(phi_deg, L_mm=100.0):
    return L_mm * math.tan(math.radians(phi_deg)) / FIELD_H_MM


def tilt_from_t(t, L_mm=100.0):
    return math.degrees(math.atan(t * FIELD_H_MM / L_mm))


# ------------------------------------------------------------ artefacts
def anterior_mm(img, mask=None, delta_mm=5.0, x0_mm=None, xc=0.5, yo=0.55, half=0.11,
                vertical=True):
    """Patient trop en avant (delta > 0) ou trop en arriere (delta < 0), en mm."""
    x0 = x0_mm if x0_mm is not None else CALIB["x0_mm"]
    if x0 is None:
        raise ValueError("x0 inconnu : lancer phantom_analysis.py puis set_calibration(x0)")
    h, w = img.shape[:2]
    s = s_from_delta(delta_mm, x0)
    # flou hors couche focale, en px a la resolution courante
    sig_mm = math.hypot(CALIB["blur_sigma0_mm"], CALIB["blur_kappa_mm_per_mm"] * delta_mm)
    blur_px = sig_mm / _obj_pix_mm(h) if abs(delta_mm) > 0 else 0
    out, m = pa.anterior_position(img, mask, s=s, blur=blur_px, xc=xc, yo=yo, half=half)
    if vertical and delta_mm != 0:
        sod0 = SID_MM / M0
        sv = sod0 / (sod0 + delta_mm)                       # Mv(delta)/M0
        x, y = pa._grid(h, w)
        wx = np.exp(-((x - xc * w) / (half * w)) ** 4)
        ysrc = yo * h + (y - yo * h) / (1 + (sv - 1) * wx)
        out, m = pa._remap(out, m, x, ysrc.astype(np.float32))
    return out, m, dict(delta_mm=delta_mm, x0_mm=x0, s=s, blur_px=blur_px)


def chin_deg(img, mask=None, theta_deg=8.0, d_ap_mm=40.0, palate=0.2, **kw):
    """theta > 0 : menton trop bas (sourire) ; theta < 0 : menton trop haut."""
    k = k_from_chin(theta_deg, d_ap_mm)
    if theta_deg >= 0:
        out, m = pa.chin_low(img, mask, k=k, **kw)
    else:
        out, m = pa.chin_high(img, mask, k=k, palate=palate, **kw)
    return out, m, dict(theta_deg=theta_deg, k=k, d_ap_mm=d_ap_mm)


def tilt_deg(img, mask=None, phi_deg=3.0, L_mm=100.0, mag=0.0, **kw):
    t = t_from_tilt(phi_deg, L_mm)
    out, m = pa.tilt(img, mask, t=t, mag=mag, **kw)
    return out, m, dict(phi_deg=phi_deg, t=t, L_mm=L_mm)


# plages physiques (equivalentes aux plages relatives de l'Article 6 si x0 ~ 30 mm)
PHYS_RANGES = {
    "trop_en_avant": ("delta_mm", (2.0, 12.0)),
    "trop_en_arriere": ("delta_mm", (-12.0, -2.0)),
    "menton_bas": ("theta_deg", (3.0, 20.0)),
    "menton_haut": ("theta_deg", (-20.0, -3.0)),
    "inclinaison": ("phi_deg", (-4.0, 4.0)),
}


def sample(name, img, mask=None, rng=None):
    """Tire une erreur au hasard dans sa plage physique ; renvoie aussi la verite."""
    rng = rng or np.random.default_rng()
    key, (lo, hi) = PHYS_RANGES[name]
    v = float(rng.uniform(lo, hi))
    if name.startswith("trop"):
        return anterior_mm(img, mask, delta_mm=v)
    if name.startswith("menton"):
        return chin_deg(img, mask, theta_deg=v)
    return tilt_deg(img, mask, phi_deg=v)


def relative_ranges_to_physical(x0_mm, d_ap_mm=40.0, L_mm=100.0):
    """Traduit les plages relatives de l'Article 6 en grandeurs physiques."""
    r = pa.ARTIFACTS
    return {
        "trop_en_avant (mm)": tuple(round(delta_from_s(s, x0_mm), 1) for s in r["trop_en_avant"][1]["s"]),
        "trop_en_arriere (mm)": tuple(round(delta_from_s(s, x0_mm), 1) for s in r["trop_en_arriere"][1]["s"]),
        "menton_bas (deg)": tuple(round(chin_from_k(k, d_ap_mm), 1) for k in r["menton_bas"][1]["k"]),
        "menton_haut (deg)": tuple(round(-chin_from_k(k, d_ap_mm), 1) for k in r["menton_haut"][1]["k"]),
        "inclinaison (deg)": tuple(round(tilt_from_t(t, L_mm), 1) for t in r["inclinaison"][1]["t"]),
    }


if __name__ == "__main__":
    for x0 in (20, 30, 40):
        print(f"x0 = {x0} mm :", relative_ranges_to_physical(x0))
