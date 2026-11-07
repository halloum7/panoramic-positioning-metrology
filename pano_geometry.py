"""
pano_geometry.py — Modele geometrique de la panoramique (Article 7)
====================================================================
Relie les parametres relatifs du simulateur de l'Article 6 (s, k, t)
a des grandeurs physiques (deplacement en mm, angles en degres).

Modele : faisceau etroit en rotation autour d'un centre instantane C.
  - SID  : distance source-detecteur (Owandy I-Max : 500 mm)
  - M0   : grandissement nominal dans la couche focale (I-Max : 1.23)
  - SOD0 = SID / M0 : distance source - couche focale (~406.5 mm)
  - x0   : distance centre de rotation -> couche focale dans la zone
           incisive (INCONNUE : a etalonner sur fantome)

Convention de signe : delta > 0 = objet deplace vers le detecteur
(patient TROP EN AVANT) ; delta < 0 = patient TROP EN ARRIERE.

Grandissements (derivation dans le plan de l'article, section Methodes) :
  Mv(delta) = SID / (SOD0 + delta)                (projection conique)
  Mh(delta) = M0 * x0 / (x0 + delta)              (vitesse de lecture du
                                                   capteur accordee sur x0)
  -> en delta = 0 : Mh = Mv = M0 (definition de la couche focale)
  -> trop en avant : Mh < Mv -> incisives etroites
  -> trop en arriere : Mh > Mv -> incisives larges
"""
from __future__ import annotations
import numpy as np

# --- Constantes constructeur Owandy I-Max (fiche technique) -----------
SID_MM = 500.0          # source-to-image distance
M0 = 1.23               # grandissement panoramique standard adulte
SENSOR_H_MM = 146.0     # hauteur sensible du capteur PAN
SENSOR_PIX_MM = 0.096   # ImagerPixelSpacing lu dans les DICOM (pixel effectif, 1536 lignes)
OBJECT_PIX_MM = 0.078156  # PixelSpacing DICOM = 0.096 / 1.228 (plan de la couche focale)
SOD0_MM = SID_MM / M0   # ~406.5 mm


# --- Grandissements ----------------------------------------------------
def mag_vertical(delta_mm, sid=SID_MM, m0=M0):
    sod0 = sid / m0
    return sid / (sod0 + np.asarray(delta_mm, float))


def mag_horizontal(delta_mm, x0_mm, m0=M0):
    return m0 * x0_mm / (x0_mm + np.asarray(delta_mm, float))


def aspect_distortion(delta_mm, x0_mm, sid=SID_MM, m0=M0):
    """q = (largeur/hauteur image) / (largeur/hauteur reelle) = Mh/Mv."""
    return mag_horizontal(delta_mm, x0_mm, m0) / mag_vertical(delta_mm, sid, m0)


# --- Inversions : parametre relatif -> grandeur physique -------------
def delta_from_hscale(h_scale, x0_mm):
    """Simulateur 'anterior_position' : facteur d'echelle horizontal
    h = Mh(delta)/M0 applique a la zone incisive (h = 1 - s ou 1 + s
    selon la convention du code). Retourne delta en mm."""
    h = np.asarray(h_scale, float)
    return x0_mm * (1.0 / h - 1.0)


def delta_from_aspect(q, x0_mm, sid=SID_MM, m0=M0):
    """Estimateur sur image REELLE : q = rapport largeur/hauteur mesure
    d'une couronne incisive divise par le rapport anatomique de reference.
    Solution exacte de q = Mh/Mv."""
    sod0 = sid / m0
    q = np.asarray(q, float)
    return x0_mm * sod0 * (1.0 - q) / (q * sod0 - x0_mm)


def detector_pixel_mm(image_height_px=1536):
    """Pixel au niveau du detecteur : ImagerPixelSpacing = 0.096 mm (verifie
    sur les 198 DICOM, 1536 lignes = toute la hauteur du capteur)."""
    return SENSOR_PIX_MM * 1536 / image_height_px


def chin_angle_deg(k_frac_h, image_height_px, d_ap_mm=40.0, delta_mm=0.0):
    """Simulateur 'chin_low/chin_high' : fleche k (fraction de H) mesuree
    aux molaires. d_ap_mm : distance sagittale incisives-molaires (~25-40 mm).
    Retourne l'angle de bascule du plan d'occlusion (degres)."""
    dz_det = np.asarray(k_frac_h, float) * image_height_px * detector_pixel_mm(image_height_px)
    dz_obj = dz_det / mag_vertical(delta_mm)
    return np.degrees(np.arctan(dz_obj / d_ap_mm))


def tilt_angle_deg(t_frac_h, image_height_px, inter_molar_mm=100.0, delta_mm=0.0):
    """Simulateur 'tilt' : denivele t (fraction de H) entre les bords gauche et
    droit de l'image (rami, ~100 mm). Voir aussi pano_calibrated.py."""
    dz_det = np.asarray(t_frac_h, float) * image_height_px * detector_pixel_mm(image_height_px)
    dz_obj = dz_det / mag_vertical(delta_mm)
    return np.degrees(np.arctan(dz_obj / inter_molar_mm))


def rescale_sigma(sigma_px, new_width, orig_width):
    """Remise a l'echelle des flous definis en pixels."""
    return sigma_px * new_width / orig_width


# --- Etalonnage sur fantome : ajustement de x0 -------------------------
def fit_x0(delta_known_mm, measured_hscale):
    """Moindres carres de h = x0/(x0+delta) sur des acquisitions fantome
    a deplacements connus. Retourne x0 (mm) et son ecart-type."""
    from scipy.optimize import curve_fit
    f = lambda d, x0: x0 / (x0 + d)
    popt, pcov = curve_fit(f, np.asarray(delta_known_mm, float),
                           np.asarray(measured_hscale, float), p0=[40.0])
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


if __name__ == "__main__":
    print(f"SOD0 = {SOD0_MM:.1f} mm ; detecteur-couche focale = {SID_MM - SOD0_MM:.1f} mm")
    print("\nTable : facteur d'echelle simulateur -> deplacement (mm) selon x0")
    for x0 in (20, 40, 60):
        row = [f"h={h:.2f}: {delta_from_hscale(h, x0):+6.1f}" for h in (0.65, 0.85, 1.15, 1.35)]
        print(f"  x0={x0:>2} mm | " + " | ".join(row))
