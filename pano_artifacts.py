"""
pano_artifacts.py — Simulateur physique d'artefacts de radiographie panoramique dentaire.

Chaque fonction prend une image 2D (uint8 ou float, H x W) et, optionnellement, un masque
(même taille) qui subit EXACTEMENT la même transformation géométrique. Elles renvoient (image, masque).
Toutes les grandeurs sont relatives à la taille de l'image (indépendantes de la résolution).

Repère panoramique utilisé (valeurs par défaut, ajustables) :
  xc = 0.5 W   : plan sagittal médian (incisives)
  yo = 0.55 H  : hauteur du plan d'occlusion
  Teeth band   : |x - xc| < 0.27 W

Modèles physiques (géométrie du canal focal) :
  - Menton bas / haut  : rotation de la tête autour d'un axe transverse -> le plan d'occlusion se
                          courbe en "sourire" (bas) ou en "grimace" (haut) : dy(x) = ±k (x-xc)^2.
                          Menton haut : en plus, superposition du palais dur (bande radio-opaque
                          horizontale au-dessus des apex maxillaires).
  - Inclinaison         : rotation autour de l'axe antéro-postérieur -> plan d'occlusion penché :
                          dy(x) = t (x-xc), plus grandissement vertical différent G/D.
  - Rotation (lacet)    : un côté plus proche du détecteur -> grandissement horizontal
                          asymétrique (dents plus larges d'un côté, plus étroites de l'autre).
  - Trop en avant/arrière : les incisives sortent du canal focal -> grandissement horizontal
                          < 1 (avant, dents étroites) ou > 1 (arrière, dents larges) dans la zone
                          antérieure + flou horizontal (hors du plan de coupe).
  - Fantôme du rachis   : colonne cervicale projetée au centre -> opacité verticale gaussienne
                          qui réduit le contraste de la région antérieure.
  - Mouvement           : le patient bouge pendant le balayage (le temps correspond à x) ->
                          discontinuité verticale à une colonne x0 + flou de bougé local.
  - Fantôme métallique  : objet dense d'un côté reprojeté du côté opposé, agrandi
                          horizontalement (x1.5-2.5), plus haut, très flou et de faible amplitude.
  - Dose inadaptée      : calibré sur la base Owandy : le bruit final ne dépend pas du DAP
                          (normalisation logicielle) ; les images jugées "dose inadaptée" sont
                          plus claires/contrastées (pénétration insuffisante) -> LUT gamma + saturation.
"""
import numpy as np
import cv2

# ----------------------------------------------------------------------------- outils
def _f(img):
    return img.astype(np.float32)

def _out(img, like):
    if like.dtype == np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img.astype(like.dtype)

def _remap(img, mask, mapx, mapy):
    out = cv2.remap(_f(img), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = None
    if mask is not None:
        m = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return _out(out, img), m

def _grid(h, w):
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return x, y

def _band(x, xc, half, soft):
    """poids 1 dans |x-xc|<half, décroissance douce sur 'soft' pixels."""
    d = np.abs(x - xc) - half
    return np.clip(1 - d / max(soft, 1), 0, 1).astype(np.float32)

# ----------------------------------------------------------------------------- artefacts
def chin_low(img, mask=None, k=0.10, xc=0.5, yo=0.55):
    """Menton trop bas : plan d'occlusion en 'sourire'. k = flèche relative (fraction de H)
    au niveau des molaires (|x-xc| = 0.3 W). Valeurs réalistes 0.03-0.12."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    u = (x - xc * w) / (0.3 * w)
    vert = np.exp(-((y - yo * h) / (0.35 * h)) ** 2)          # surtout près du plan d'occlusion
    dy = -k * h * (u ** 2 - 0.35) * vert                         # centre descend, côtés montent
    return _remap(img, mask, x, (y - dy).astype(np.float32))

def chin_high(img, mask=None, k=0.08, palate=0.25, xc=0.5, yo=0.55):
    """Menton trop haut : plan d'occlusion plat/inversé ('grimace') + palais dur projeté
    sur les apex maxillaires (bande radio-opaque). palate = amplitude relative (0-0.4)."""
    out, m = chin_low(img, mask, k=-k, xc=xc, yo=yo)
    h, w = img.shape[:2]; x, y = _grid(h, w)
    yp = (yo - 0.17) * h                                          # niveau des apex maxillaires
    band = np.exp(-((y - yp) / (0.025 * h)) ** 2) * _band(x, xc * w, 0.25 * w, 0.08 * w)
    o = _f(out); o = o + palate * (255 - o) * band
    return _out(o, img), m

def tilt(img, mask=None, t=0.04, mag=0.04, xc=0.5, yo=0.55):
    """Inclinaison de la tête : plan d'occlusion penché (t = dénivelé relatif entre les deux
    côtés, en fraction de H) + grandissement vertical différent à gauche/droite (mag)."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    u = (x - xc * w) / (0.5 * w)                                   # -1 .. 1
    dy = t * h * u / 2
    s = 1 + mag * u                                                # grandissement vertical local
    ysrc = yo * h + (y - yo * h - dy) / s
    return _remap(img, mask, x, ysrc.astype(np.float32))

def rotation(img, mask=None, r=0.08, xc=0.5):
    """Rotation (lacet) : grandissement horizontal 1+r d'un côté, 1-r de l'autre."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    u = (x - xc * w) / (0.5 * w)
    # dérivée dx_src/dx = 1/(1 + r*sign) intégrée -> carte lisse
    scale = 1 + r * np.tanh(3 * u)
    xs = np.cumsum(1 / scale[0:1, :], axis=1).astype(np.float32)
    xs = xs * (w / xs[0, -1])
    xsrc = np.repeat(xs, h, axis=0)
    return _remap(img, mask, xsrc, y)

def anterior_position(img, mask=None, s=-0.25, blur=6, xc=0.5, yo=0.55, half=0.11):
    """Patient trop en avant (s<0 : incisives étroites) ou trop en arrière (s>0 : élargies),
    avec flou horizontal des incisives sorties du canal focal. s réaliste : ±0.15-0.35.
    Pondération super-gaussienne (pas de bords francs), centrée sur l'arcade antérieure."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    wx = np.exp(-((x - xc * w) / (half * w)) ** 4)
    wy = np.exp(-((y - yo * h) / (0.32 * h)) ** 4)
    xsrc = xc * w + (x - xc * w) / (1 + s * wx)
    out, m = _remap(img, mask, xsrc.astype(np.float32), y)
    if blur > 0:
        wgt = (wx * wy).astype(np.float32)
        b = cv2.GaussianBlur(_f(out), (0, 0), sigmaX=blur, sigmaY=0.7)
        out = _out(_f(out) * (1 - wgt) + b * wgt, img)
    return out, m

def spine_ghost(img, mask=None, a=0.35, width=0.10, xc=0.5, y0=0.45):
    """Fantôme du rachis cervical : opacité médiane qui s'élargit vers le bas (sous le plan
    d'occlusion, sur les incisives inférieures et la symphyse) ; voile radio-opaque qui
    réduit le contraste. a : amplitude (0.15-0.5) ; width : demi-largeur relative en haut."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    t = np.clip((y - y0 * h) / ((1 - y0) * h), 0, 1)               # 0 en haut -> 1 en bas
    wid = width * w * (0.6 + 0.8 * t)
    prof = np.exp(-((x - xc * w) / wid) ** 2)
    vert = 1 / (1 + np.exp(-(y - y0 * h) / (0.03 * h)))            # début progressif
    g = (a * prof * vert).astype(np.float32)
    o = _f(img)
    loc = cv2.GaussianBlur(o, (0, 0), 25)                          # perte de contraste local
    o = o * (1 - g) + (0.3 * loc + 0.7 * 230) * g
    return _out(o, img), mask

def motion(img, mask=None, x0=0.6, shift=0.015, width=0.01):
    """Mouvement pendant le balayage : à partir de la colonne x0 (fraction de W) le reste de
    l'image est décalé verticalement de 'shift' (fraction de H) ; transition sur 'width'."""
    h, w = img.shape[:2]; x, y = _grid(h, w)
    ramp = 1 / (1 + np.exp(np.clip(-(x - x0 * w) / max(width * w / 4, 1), -50, 50)))
    out, m = _remap(img, mask, x, (y - shift * h * ramp).astype(np.float32))
    zone = np.exp(-((x - x0 * w) / (width * w)) ** 2)
    b = cv2.GaussianBlur(_f(out), (0, 0), sigmaX=0.5, sigmaY=4)
    return _out(_f(out) * (1 - zone) + b * zone, img), m

def metal_ghost(img, mask=None, side=None, amp=0.35, hmag=2.0, up=0.08, blur=6,
                thr=245, min_area=400, rng=None, source='auto'):
    """Fantôme métallique : l'objet dense le plus grand d'un côté (couronne, implant, bijou)
    est reprojeté du côté opposé : miroir, agrandi horizontalement (hmag), décalé vers le
    haut (up, fraction de H), très flou, d'amplitude amp. Sans objet dense (ou source='boucle') :
    boucle d'oreille synthétique."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    a = _f(img)
    thr = min(thr, max(200, float(np.quantile(img, 0.998))))      # seuil adaptatif
    n, lab, st, _ = cv2.connectedComponentsWithStats((img >= thr).astype(np.uint8), 8)
    cand = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area
            and abs(st[i, 0] + st[i, 2] / 2 - w / 2) > 0.08 * w]
    obj = np.zeros((h, w), np.float32)
    if source == 'auto' and cand:
        i = max(cand, key=lambda j: st[j, cv2.CC_STAT_AREA])
        obj[lab == i] = 1
    else:                                                          # boucle d'oreille synthétique
        sd = side or rng.choice([-1, 1])
        cx = int(w * (0.07 if sd < 0 else 0.93)); cy = int(0.66 * h)
        r = int(0.035 * h)
        cv2.ellipse(obj, (cx, cy), (int(r * 0.8), r), 0, 0, 360, 1, thickness=max(3, int(0.009 * h)))
        cv2.line(obj, (cx, cy - r), (cx, cy - int(1.6 * r)), 1, max(2, int(0.005 * h)))
        a = np.maximum(a, 250 * obj)                                # l'objet réel, net, de son côté
    ghost = cv2.flip(obj, 1)                                       # côté opposé
    ys, xs = np.nonzero(ghost)
    gx = float(xs.mean()) if len(xs) else w / 2                   # centre de l'objet reprojeté
    # agrandissement horizontal autour de son propre centre + décalage vers le haut
    M = np.float32([[hmag, 0, gx * (1 - hmag)], [0, 1, -up * h]])
    ghost = cv2.warpAffine(ghost, M, (w, h))
    ghost = cv2.GaussianBlur(ghost, (0, 0), sigmaX=blur * hmag, sigmaY=blur)
    ghost = ghost / (ghost.max() + 1e-6)
    o = a + amp * (255 - a) * ghost
    return _out(o, img), mask

def dose_lut(img, mask=None, gamma=0.85, clip_hi=0.995):
    """Dose inadaptée (calibrée Owandy) : gamma<1 -> image plus claire et plus contrastée dans
    les hauts niveaux (pénétration insuffisante) ; gamma>1 -> image plus sombre (surexposition).
    clip_hi : quantile au-delà duquel les niveaux sont saturés à 255."""
    a = _f(img) / 255.0
    o = np.power(np.clip(a, 0, 1), gamma)
    q = np.quantile(o, clip_hi)
    o = np.clip(o / max(q, 1e-6), 0, 1) * 255
    return _out(o, img), mask

# ----------------------------------------------------------------------------- registre
ARTIFACTS = {
    'menton_bas':        (chin_low,          dict(k=(0.03, 0.12))),
    'menton_haut':       (chin_high,         dict(k=(0.03, 0.10), palate=(0.10, 0.35))),
    'inclinaison':       (tilt,              dict(t=(-0.06, 0.06), mag=(-0.05, 0.05))),
    'rotation':          (rotation,          dict(r=(-0.20, 0.20))),
    'trop_en_avant':     (anterior_position, dict(s=(-0.35, -0.12), blur=(3, 9))),
    'trop_en_arriere':   (anterior_position, dict(s=(0.12, 0.35), blur=(3, 9))),
    'fantome_rachis':    (spine_ghost,       dict(a=(0.25, 0.65), width=(0.08, 0.14))),
    'mouvement':         (motion,            dict(x0=(0.25, 0.75), shift=(-0.04, 0.04))),
    'fantome_metallique':(metal_ghost,       dict(amp=(0.15, 0.45), hmag=(1.5, 2.5), up=(0.04, 0.12))),
    'dose_inadaptee':    (dose_lut,          dict(gamma=(0.70, 0.92))),
}

def apply(name, img, mask=None, rng=None, **override):
    """Applique l'artefact 'name' avec des paramètres tirés au hasard dans les plages réalistes."""
    rng = rng or np.random.default_rng()
    fn, ranges = ARTIFACTS[name]
    p = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
    p.update(override)
    if fn is metal_ghost:
        p['rng'] = rng
    out, m = fn(img, mask, **p)
    return out, m, p

def random_augment(img, mask=None, n=(1, 3), rng=None, exclude=()):
    """Augmentation 'physique' : combine 1 à 3 artefacts compatibles (jamais menton haut+bas,
    jamais avant+arrière)."""
    rng = rng or np.random.default_rng()
    names = [k for k in ARTIFACTS if k not in exclude]
    chosen = []
    for k in rng.permutation(names):
        if len(chosen) >= rng.integers(n[0], n[1] + 1):
            break
        if {k, *chosen} >= {'menton_bas', 'menton_haut'} or {k, *chosen} >= {'trop_en_avant', 'trop_en_arriere'}:
            continue
        chosen.append(k)
    params = {}
    for k in chosen:
        img, mask, p = apply(k, img, mask, rng)
        params[k] = {a: round(b, 4) for a, b in p.items() if a != 'rng'}
    return img, mask, params
