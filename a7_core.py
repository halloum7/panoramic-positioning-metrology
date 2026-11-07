"""
a7_core.py — Coeur de l'Article 7 (sans torch) : simulation des erreurs de
positionnement a verite connue, et metriques de validation sur images reelles.

Cibles apprises par le reseau (toutes independantes de x0) :
  s     : facteur d'echelle horizontal de la zone incisive (h = 1 + s)
          s < 0 -> trop en avant ; s > 0 -> trop en arriere
          conversion en mm APRES l'essai sur fantome : delta = -s * x0 / (1 + s)
  theta : bascule du menton, degres (+ = menton bas, - = menton haut)
  phi   : inclinaison de la tete, degres (+ = cote droit de l'image plus bas)
"""
from __future__ import annotations
import numpy as np
import cv2

import pano_artifacts as PA
import pano_calibrated as PC

S_RANGE = 0.35          # plage de l'Article 6
CHIN_RANGE = 20.0       # degres (equivalent de k <= 0,12 avec d_ap = 40 mm)
TILT_RANGE = 4.0        # degres (equivalent de |t| <= 0,06 avec L = 100 mm)
P_ZERO = 0.30           # proportion d'images sans l'erreur consideree
BLUR_PER_S = 6.0 / 0.235  # couplage flou/echelle de l'Article 6 (px a W = 1024) ;
                          # sera remplace par kappa mesure sur le fantome
NUISANCE = ["fantome_rachis", "mouvement", "fantome_metallique", "dose_inadaptee", "rotation"]
P_NUISANCE = 0.25
TARGET_SCALE = np.array([S_RANGE, CHIN_RANGE, TILT_RANGE], np.float32)
TARGET_NAMES = ["s", "theta_deg", "phi_deg"]


# ------------------------------------------------------------ simulation
def draw_targets(rng):
    s = 0.0 if rng.random() < P_ZERO else float(rng.uniform(-S_RANGE, S_RANGE))
    th = 0.0 if rng.random() < P_ZERO else float(rng.uniform(-CHIN_RANGE, CHIN_RANGE))
    ph = 0.0 if rng.random() < P_ZERO else float(rng.uniform(-TILT_RANGE, TILT_RANGE))
    return s, th, ph


def text_band(im, rng, frac=56 / 1536):
    """Imite le masquage d'anonymisation des images Owandy (bande en bas)."""
    h = im.shape[0]
    out = im.copy()
    out[int(h * (1 - frac * rng.uniform(0.8, 1.2))):] = 0
    return out


def photometric(im, rng):
    f = im.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    if rng.random() < 0.3:
        f = f + rng.normal(0, rng.uniform(2, 6), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def base_aug(im, rng):
    """Augmentation de la base AVANT simulation, sans effet sur les cibles :
    miroir gauche-droite, petite translation/echelle isotrope (pas de rotation :
    une rotation dans le plan imiterait une inclinaison)."""
    if rng.random() < 0.5:
        im = im[:, ::-1].copy()
    h, w = im.shape[:2]
    sc = rng.uniform(0.95, 1.05)
    M = np.float32([[sc, 0, (1 - sc) * w / 2 + rng.uniform(-0.02, 0.02) * w],
                    [0, sc, (1 - sc) * h / 2 + rng.uniform(-0.02, 0.02) * h]])
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def simulate(im, s, th, ph, rng, nuisance=True, band=True):
    """Applique les trois erreurs (verite connue) puis des perturbations sans
    rapport avec les cibles (randomisation de domaine)."""
    w = im.shape[1]
    if s != 0:
        im, _ = PA.anterior_position(im, None, s=s, blur=BLUR_PER_S * abs(s) * w / 1024)
    if th != 0:
        im, _, _ = PC.chin_deg(im, None, theta_deg=th, palate=float(rng.uniform(0.1, 0.35)))
    if ph != 0:
        im, _, _ = PC.tilt_deg(im, None, phi_deg=ph, mag=0.0)
    if nuisance:
        for k in rng.permutation(NUISANCE):
            if rng.random() < P_NUISANCE:
                im, _, _ = PA.apply(k, im, None, rng)
    im = photometric(im, rng)
    if band and rng.random() < 0.7:
        im = text_band(im, rng)
    return im


def make_sample(base, rng, nuisance=True):
    im = base_aug(base, rng)
    s, th, ph = draw_targets(rng)
    return simulate(im, s, th, ph, rng, nuisance), np.array([s, th, ph], np.float32)


# ------------------------------------------------------------ metriques
def auc(y, score):
    """AUC de Mann-Whitney (gere les ex aequo)."""
    y = np.asarray(y).astype(bool); sc = np.asarray(score, float)
    ok = np.isfinite(sc); y, sc = y[ok], sc[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(sc)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, score, n_boot=2000, seed=0):
    y = np.asarray(y).astype(bool); sc = np.asarray(score, float)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a = auc(y[i], sc[i])
        if np.isfinite(a):
            vals.append(a)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return auc(y, sc), float(lo), float(hi)


def sign_test(pos_label, neg_label, value):
    """Images portant UNE seule des deux etiquettes opposees : la valeur
    predite doit etre positive pour pos_label, negative pour neg_label.
    Retourne (n, exactitude, p binomial unilateral vs 0,5)."""
    from scipy.stats import binomtest
    pos = np.asarray(pos_label).astype(bool); neg = np.asarray(neg_label).astype(bool)
    v = np.asarray(value, float)
    keep = pos ^ neg
    if keep.sum() == 0:
        return 0, np.nan, np.nan
    correct = np.where(pos[keep], v[keep] > 0, v[keep] < 0)
    k, n = int(correct.sum()), int(keep.sum())
    return n, k / n, float(binomtest(k, n, 0.5, alternative="greater").pvalue)


# correspondance etiquettes du dentiste -> score oriente du modele
LABEL_SCORES = {
    "Trop en avant": ("s", -1),
    "Trop en arrière": ("s", +1),
    "Menton trop bas": ("theta_deg", +1),
    "Menton trop haut": ("theta_deg", -1),
    "Inclinaison tête": ("phi_deg", "abs"),
}


def score_for(pred_df, label):
    col, how = LABEL_SCORES[label]
    v = pred_df[col].to_numpy(float)
    return np.abs(v) if how == "abs" else how * v
