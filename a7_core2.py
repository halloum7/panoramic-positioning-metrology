"""
a7_core2.py — Version 2 (Article 7) : simulation sans "raccourci" et
estimateur geometrique explicable.

Pourquoi : en v1, le reseau retrouvait parfaitement les erreurs simulees
(r = 0,998) mais predisait ~0 sur les images reelles. Hypotheses :
  H1 le reseau reconnait la signature du simulateur (bande floue, bords de
     la fenetre de deformation) plutot que l'anatomie ;
  H2 les vraies erreurs ne ressemblent pas aux erreurs simulees ;
  H3 les annotations reelles sont bruitees.
La v2 attaque H1 (cibles inchangees : s, theta, phi) et ajoute une mesure
geometrique directe, sans apprentissage des erreurs.
"""
from __future__ import annotations
import numpy as np
import cv2

import pano_artifacts as PA
import pano_calibrated as PC
import a7_core as C

S_RANGE, CHIN_RANGE, TILT_RANGE = C.S_RANGE, C.CHIN_RANGE, C.TILT_RANGE
TARGET_SCALE = C.TARGET_SCALE
TARGET_NAMES = C.TARGET_NAMES
NUISANCE = C.NUISANCE


# ================================================================ simulation v2
def _grid(h, w):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def anterior_v2(im, s, xc, yo, half, p, jitter):
    """Comme pano_artifacts.anterior_position, mais fenetre de forme variable
    (exposant p = 2 ou 4, demi-largeur 'half' tiree au hasard) et SANS flou :
    le flou est gere a part (decorrele de s). Toujours appliquee, meme si s = 0
    (deformation nulle + decalage sous-pixel) : l'interpolation ne trahit plus
    la presence d'une erreur."""
    h, w = im.shape[:2]
    x, y = _grid(h, w)
    wx = np.exp(-np.abs((x - xc * w) / (half * w)) ** p)
    xsrc = xc * w + (x - xc * w) / (1 + s * wx) + jitter[0]
    return cv2.remap(im.astype(np.float32), xsrc.astype(np.float32), (y + jitter[1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def incisor_blur(f, sigma, xc, yo, half):
    if sigma <= 0.05:
        return f
    h, w = f.shape[:2]
    x, y = _grid(h, w)
    wgt = (np.exp(-((x - xc * w) / (half * w)) ** 2) * np.exp(-((y - yo * h) / (0.32 * h)) ** 4)).astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=0.7)
    return f * (1 - wgt) + b * wgt


def simulate_v2(im, s, th, ph, rng, nuisance=True, band=True, use_blur=True):
    h, w = im.shape[:2]
    xc = 0.5 + rng.uniform(-0.02, 0.02)
    yo = 0.55 + rng.uniform(-0.03, 0.03)
    half = rng.uniform(0.08, 0.15)
    p = rng.choice([2, 4])
    f = anterior_v2(im, s, xc, yo, half, p, rng.uniform(-0.5, 0.5, 2))
    if use_blur:
        # flou = part liee a |s| (physique) + part aleatoire independante de s
        sig = (C.BLUR_PER_S * abs(s) * rng.uniform(0.5, 1.5) + rng.uniform(0, 4) * (rng.random() < 0.5)) * w / 1024
        f = incisor_blur(f, sig, xc, yo, half)
    im = np.clip(f, 0, 255).astype(np.uint8)
    if th != 0:
        im, _, _ = PC.chin_deg(im, None, theta_deg=th, palate=float(rng.uniform(0.0, 0.35)), xc=xc, yo=yo)
    if ph != 0:
        im, _, _ = PC.tilt_deg(im, None, phi_deg=ph, mag=0.0, xc=xc, yo=yo)
    if nuisance:
        for k in rng.permutation(NUISANCE):
            if rng.random() < C.P_NUISANCE:
                im, _, _ = PA.apply(k, im, None, rng)
    im = C.photometric(im, rng)
    if band and rng.random() < 0.7:
        im = C.text_band(im, rng)
    return im


def make_sample_v2(base, rng, nuisance=True, use_blur=True):
    im = C.base_aug(base, rng)
    s, th, ph = C.draw_targets(rng)
    return simulate_v2(im, s, th, ph, rng, nuisance, use_blur=use_blur), np.array([s, th, ph], np.float32)


# ================================================================ histogrammes
def mean_cdf(images, bins=256):
    hs = [np.bincount(i.ravel(), minlength=bins)[:bins] / i.size for i in images]
    return np.cumsum(np.mean(hs, axis=0))


def match_to_cdf(im, target_cdf):
    """Adapte les niveaux de gris d'une image STS a la distribution moyenne
    des images Owandy (aucune etiquette utilisee)."""
    c = np.cumsum(np.bincount(im.ravel(), minlength=256)[:256] / im.size)
    lut = np.searchsorted(target_cdf, c).clip(0, 255).astype(np.uint8)
    return lut[im]


# ================================================================ estimateur geometrique
def occlusal_line(mask, lo=0.30, hi=0.80, span=0.30, min_teeth=6):
    """Ligne d'occlusion y(x) : milieu du plus grand vide entre dents
    superieures et inferieures, colonne par colonne, sur |x - xc| < span*W."""
    h, w = mask.shape
    xs, ys = [], []
    for x in range(int(w * (0.5 - span)), int(w * (0.5 + span))):
        col = np.nonzero(mask[int(lo * h):int(hi * h), x])[0] + int(lo * h)
        if len(col) < min_teeth:
            continue
        gaps = np.diff(col)
        j = int(np.argmax(gaps))
        if gaps[j] < 2:
            continue
        xs.append(x); ys.append((col[j] + col[j + 1]) / 2)
    return np.array(xs, float), np.array(ys, float)


def robust_quadratic(xs, ys, w, iters=3):
    """y = a + b*u + c*u^2, u = (x - W/2)/(0.3 W) ; elimination iterative des
    points aberrants (> 2,5 ecarts robustes)."""
    if len(xs) < 20:
        return np.nan, np.nan, np.nan
    u = (xs - w / 2) / (0.3 * w)
    keep = np.ones(len(u), bool)
    for _ in range(iters):
        c2, b, a = np.polyfit(u[keep], ys[keep], 2)
        r = ys - (a + b * u + c2 * u ** 2)
        mad = 1.4826 * np.median(np.abs(r[keep])) + 1e-6
        keep = np.abs(r) < 2.5 * mad
        if keep.sum() < 20:
            break
    return a, b, c2


def dominant_period(profile, pmin, pmax):
    p = profile - profile.mean()
    if np.allclose(p, 0):
        return np.nan
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(pmin), min(int(pmax), len(ac) - 2)
    if hi <= lo + 2:
        return np.nan
    seg = ac[lo:hi]
    k = int(np.argmax(seg)) + lo
    if ac[k] <= 0:
        return np.nan
    # raffinement parabolique
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-9)
    return float(k + np.clip(d, -0.5, 0.5))


def zone_metrics(img, mask, x_lo, x_hi, yocc, h):
    """Periode inter-dentaire (px) et hauteur dentaire (px) dans une zone."""
    x_lo, x_hi = int(x_lo), int(x_hi)
    rows = slice(max(0, int(yocc - 0.12 * h)), min(h, int(yocc + 0.12 * h)))
    sub = img[rows, x_lo:x_hi].astype(np.float32)
    msub = mask[rows, x_lo:x_hi]
    prof = (sub * msub).sum(0) / (msub.sum(0) + 1)
    w = img.shape[1]
    period = dominant_period(prof, 0.012 * w, 0.09 * w)
    height = float(np.median(mask[:, x_lo:x_hi].sum(0)))
    return period, height


def geometric_features(img, mask):
    """Mesures physiques sur une panoramique (img uint8, mask dents 0/1) :
      k_hat, theta_hat : courbure de la ligne d'occlusion -> bascule du menton
      t_hat, phi_hat   : pente de la ligne d'occlusion  -> inclinaison
      R_ant            : (periode/hauteur) incisives / (periode/hauteur) premolaires
                         -> grandissement horizontal relatif de la zone incisive"""
    h, w = mask.shape
    xs, ys = occlusal_line(mask)
    a, b, c2 = robust_quadratic(xs, ys, w)
    out = dict(n_occl=len(xs), occl_y=a)
    if np.isfinite(c2):
        k_hat = -c2 / h
        t_hat = b / (0.3 * h)
        out.update(k_hat=k_hat, theta_hat=np.sign(k_hat) * PC.chin_from_k(abs(k_hat)),
                   t_hat=t_hat, phi_hat=PC.tilt_from_t(t_hat))
    else:
        out.update(k_hat=np.nan, theta_hat=np.nan, t_hat=np.nan, phi_hat=np.nan)
    yocc = a if np.isfinite(a) else 0.55 * h
    pc, hc = zone_metrics(img, mask, w * 0.43, w * 0.57, yocc, h)
    pl, hl = zone_metrics(img, mask, w * 0.25, w * 0.33, yocc, h)
    pr, hr = zone_metrics(img, mask, w * 0.67, w * 0.75, yocc, h)
    pm = np.nanmean([pl, pr]); hm = np.nanmean([hl, hr])
    out.update(period_ant=pc, height_ant=hc, period_pm=pm, height_pm=hm)
    out["R_ant"] = (pc / hc) / (pm / hm) if all(np.isfinite([pc, hc, pm, hm])) and hc > 0 and hm > 0 and pm > 0 else np.nan
    out["P_ratio"] = pc / pm if np.isfinite(pc) and np.isfinite(pm) and pm > 0 else np.nan
    return out


# score oriente des mesures geometriques pour chaque etiquette du dentiste
GEO_SCORES = {
    "Trop en avant": ("P_ratio", -1),       # incisives plus etroites
    "Trop en arrière": ("P_ratio", +1),
    "Menton trop bas": ("theta_hat", +1),
    "Menton trop haut": ("theta_hat", -1),
    "Inclinaison tête": ("phi_hat", "abs"),
}


def geo_score(df, label):
    col, how = GEO_SCORES[label]
    v = df[col].to_numpy(float)
    return np.abs(v) if how == "abs" else how * v


def sign_test(pos_label, neg_label, value):
    """Test du signe en ignorant les valeurs manquantes (mesure impossible)."""
    v = np.asarray(value, float); ok = np.isfinite(v)
    return C.sign_test(np.asarray(pos_label)[ok], np.asarray(neg_label)[ok], v[ok])
