# === Article 7 — v3 : métrologie du mesureur (aucun réentraînement, ~20-30 min) ===
# Input : article6-pano, article7-code, ET la sortie (Output) du notebook v2 (poids *.best.pt)
import os, sys, glob, time, json, math
import numpy as np, cv2, pandas as pd, torch
os.system('pip -q install timm')
import timm
import matplotlib.pyplot as plt
T0 = time.time()
def _find(pattern, required=True):
    r = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    assert r or not required, f'Introuvable dans /kaggle/input : {pattern}'
    return r[0] if r else None
sys.path.insert(0, os.path.dirname(_find('a7_core.py'))); sys.path.insert(0, '/kaggle/working/code7')
import pano_artifacts as PA, pano_calibrated as PC, a7_core as C, a7_core2 as G
META_CSV = _find('metadata_157.csv'); ANNOT_XLSX = _find('grille_annotation_artefacts-remplis.xlsx')
IMG_DIR = os.path.dirname(_find('IMG_0001.png'))
assert not glob.glob('/kaggle/input/**/CONFIDENTIEL*', recursive=True)
W_V2 = _find('convnext_tiny__v2.best.pt')
W_NB = _find('convnext_tiny__v2_sans_flou.best.pt', required=False)
OUT = '/kaggle/working/resultats7v3'; os.makedirs(f'{OUT}/figures', exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Appareil :', DEV)
H, W = 512, 1024

meta = pd.read_csv(META_CSV)
meta = meta[meta.duplicate_of.isna() & meta.flag.isna()] if {'duplicate_of', 'flag'} <= set(meta.columns) else meta
ann = pd.read_excel(ANNOT_XLSX, sheet_name='Annotation'); ann = ann[ann['Qualité globale (1-3)'].notna()]
df = meta.merge(ann, on=['image_id', 'patient_id'], how='inner').reset_index(drop=True); print('Images Owandy :', len(df))
REAL = np.stack([cv2.resize(cv2.imread(f'{IMG_DIR}/{i}.png', 0), (W, H), interpolation=cv2.INTER_AREA) for i in df.image_id])

MEAN, STD = 0.45, 0.225
def to_tensor(im):
    x = (im.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(np.repeat(x[None], 3, 0))
def load(path):
    m = timm.create_model('convnext_tiny', pretrained=False, num_classes=3).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV)); m.eval(); return m
MODELS = {'v2': load(W_V2)}
if W_NB: MODELS['v2_sans_flou'] = load(W_NB)
@torch.no_grad()
def predict(model, ims, bs=32):
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[i:i + bs]]).to(DEV)
        with torch.autocast(device_type='cuda', enabled=DEV == 'cuda'):
            p = model(x).float(); pf = model(x.flip(-1)).float() * torch.tensor([1., 1., -1.], device=DEV)
        out.append(((p + pf) / 2).cpu().numpy())
    return np.concatenate(out) * G.TARGET_SCALE
NAMES = G.TARGET_NAMES   # s, theta_deg, phi_deg

# ---------- 1. Répétabilité : même image, déformation nulle (s = θ = φ = 0) tirée 5 fois ----------
R = 5
NULL = np.stack([G.simulate_v2(REAL[j], 0., 0., 0., np.random.default_rng(70_000 + 1000 * r + j), nuisance=False, band=False)
                 for j in range(len(REAL)) for r in range(R)])
NULL_BASE = np.repeat(np.arange(len(REAL)), R)

# ---------- 2. Courbes dose-réponse : UNE erreur ajoutée à la fois, sur les 151 images ----------
GRID = {'s': [-0.2, -0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1, 0.2],
        'theta_deg': [-10, -6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6, 10],
        'phi_deg': [-4, -3, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 3, 4]}
DR_IM, DR_META = [], []
for k, name in enumerate(NAMES):
    for v in GRID[name]:
        for j in range(len(REAL)):
            rng = np.random.default_rng(90_000 + 7919 * k + 104729 * GRID[name].index(v) + j)
            t = [0., 0., 0.]; t[k] = v
            DR_IM.append(G.simulate_v2(REAL[j], *t, rng, nuisance=False, band=False)); DR_META.append((name, v, j))
DR_IM = np.stack(DR_IM); DR = pd.DataFrame(DR_META, columns=['parametre', 'ajoute', 'image'])
print('Images simulées : null', len(NULL), '| dose-réponse', len(DR_IM), '| %.1f min' % ((time.time() - T0) / 60))

rows_rep, rows_lin, rows_lod, curves = [], [], [], []
for mname, mod in MODELS.items():
    P0 = predict(mod, REAL); PN = predict(mod, NULL); PD = predict(mod, DR_IM)
    pd.DataFrame(P0, columns=NAMES).assign(image_id=df.image_id).to_csv(f'{OUT}/mesures_reelles_{mname}.csv', index=False)
    for k, name in enumerate(NAMES):
        # répétabilité (écart-type intra-image) et biais du zéro
        resid = PN[:, k] - np.repeat(PN[:, k].reshape(-1, R).mean(1), R)
        sd_r = float(np.sqrt((resid ** 2).sum() / (len(resid) - len(REAL))))
        base = PN[:, k].reshape(-1, R).mean(1)              # mesure de référence de chaque image
        rows_rep.append(dict(modele=mname, parametre=name, SD_repetabilite=sd_r,
                             SD_entre_images=float(base.std(ddof=1)), mediane_reelle=float(np.median(P0[:, k]))))
        # dose-réponse (différence appariée par rapport à la référence de la même image)
        sel = (DR.parametre == name).to_numpy()
        d = DR[sel].copy(); d['mesure'] = PD[sel, k] - base[d.image.to_numpy()]
        g = d.groupby('ajoute').mesure.agg(['mean', 'std', 'count']).reset_index(); g['modele'] = mname; g['parametre'] = name
        curves.append(g)
        x = np.r_[np.zeros(len(REAL)), d.ajoute.to_numpy()]; y = np.r_[np.zeros(len(REAL)), d.mesure.to_numpy()]
        slope, icpt = np.polyfit(x, y, 1); r = float(np.corrcoef(x, y)[0, 1])
        res = y - (slope * x + icpt)
        rows_lin.append(dict(modele=mname, parametre=name, pente=slope, ordonnee=icpt, r=r, SD_residus=float(res.std(ddof=2))))
        # seuil de détection : (a) formule 3,29·SD0/pente ; (b) plus petite erreur détectée avec AUC >= 0,95
        sd0 = float(np.std(np.concatenate([PN[:, k] - np.repeat(base, R)]), ddof=1))
        lod_formula = 3.29 * sd0 / abs(slope)
        null_scores = np.abs(PN[:, k] - np.repeat(base, R))
        lod_auc = np.nan
        for v in sorted(set(abs(t) for t in GRID[name])):
            pos = np.abs(d.mesure[np.isclose(np.abs(d.ajoute), v)].to_numpy())
            a = C.auc(np.r_[np.ones(len(pos)), np.zeros(len(null_scores))], np.r_[pos, null_scores])
            if a >= 0.95: lod_auc = v; break
        rows_lod.append(dict(modele=mname, parametre=name, LoD_formule=lod_formula, LoD_AUC95=lod_auc))
REP = pd.DataFrame(rows_rep); LIN = pd.DataFrame(rows_lin); LOD = pd.DataFrame(rows_lod); CUR = pd.concat(curves)
for x0 in (20, 30, 40):   # conversion provisoire de s en mm (x0 exact : essai sur fantôme)
    m = LOD.parametre == 's'
    LOD.loc[m, f'LoD_mm_x0_{x0}'] = LOD.loc[m, 'LoD_formule'] * x0 / (1 - LOD.loc[m, 'LoD_formule'])
REP.to_csv(f'{OUT}/M1_repetabilite.csv', index=False); LIN.to_csv(f'{OUT}/M2_linearite.csv', index=False)
LOD.to_csv(f'{OUT}/M3_seuils_detection.csv', index=False); CUR.to_csv(f'{OUT}/M4_courbes_dose_reponse.csv', index=False)
print(REP.round(4).to_string()); print(LIN.round(4).to_string()); print(LOD.round(4).to_string())

# ---------- 3. Part des patients réels au-dessus du seuil de détection ----------
rows = []
P0 = pd.read_csv(f'{OUT}/mesures_reelles_v2.csv')
for name in NAMES:
    lod = float(LOD[(LOD.modele == 'v2') & (LOD.parametre == name)].LoD_formule.iloc[0])
    dev = P0[name] - P0[name].median()
    rows.append(dict(parametre=name, LoD=lod, n=len(dev), n_au_dessus=int((dev.abs() > lod).sum()),
                     pourcent=float(100 * (dev.abs() > lod).mean()), ecart_max=float(dev.abs().max())))
ABOVE = pd.DataFrame(rows); ABOVE.to_csv(f'{OUT}/M5_patients_au_dessus_seuil.csv', index=False); print(ABOVE.round(3).to_string())

# ---------- 4. Seuils selon les facteurs de confusion (métal, dentition) ----------
rows = []
strata = {'metal': (df['Métal dentaire'].fillna(0) >= 1).to_numpy(),
          'dentition_D': (df['Dentition (D/P/E/M)'].astype(str).str.upper().str[0] == 'D').to_numpy()}
P0n = predict(MODELS['v2'], NULL); PDv = predict(MODELS['v2'], DR_IM)
for k, name in enumerate(NAMES):
    base = P0n[:, k].reshape(-1, R).mean(1); sel = (DR.parametre == name).to_numpy(); d = DR[sel].copy()
    d['mesure'] = PDv[sel, k] - base[d.image.to_numpy()]
    for sname, mask in strata.items():
        for val in (True, False):
            imgs = np.nonzero(mask == val)[0]
            dd = d[d.image.isin(imgs)]
            slope = np.polyfit(dd.ajoute, dd.mesure, 1)[0]
            nn = (P0n[:, k] - np.repeat(base, R)).reshape(-1, R)[imgs].ravel()
            rows.append(dict(parametre=name, strate=sname, valeur=val, n_images=len(imgs), pente=slope,
                             LoD_formule=3.29 * nn.std(ddof=1) / abs(slope)))
STR = pd.DataFrame(rows); STR.to_csv(f'{OUT}/M6_strates.csv', index=False); print(STR.round(4).to_string())

# ---------- 5. Figures ----------
lab = {'s': 's ajouté (sans unité)', 'theta_deg': 'menton ajouté (°)', 'phi_deg': 'inclinaison ajoutée (°)'}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a_, name in zip(ax, NAMES):
    for mname, st in [('v2', '-o'), ('v2_sans_flou', '--s')]:
        g = CUR[(CUR.modele == mname) & (CUR.parametre == name)].sort_values('ajoute')
        if len(g): a_.errorbar(g.ajoute, g['mean'], yerr=g['std'], fmt=st, ms=4, capsize=2, label=mname)
    lim = max(GRID[name]); a_.plot([-lim, lim], [-lim, lim], 'k:', lw=1)
    lod = float(LOD[(LOD.modele == 'v2') & (LOD.parametre == name)].LoD_formule.iloc[0])
    a_.axvspan(-lod, lod, color='grey', alpha=.15, label='sous le seuil de détection')
    a_.set_xlabel(lab[name]); a_.set_ylabel('mesuré (différence appariée)'); a_.legend(fontsize=8)
plt.suptitle('Réponse du mesureur à des erreurs connues ajoutées aux 151 images Owandy'); plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig5_dose_reponse.png', dpi=150); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a_, name in zip(ax, NAMES):
    dev = P0[name] - P0[name].median(); lod = float(ABOVE[ABOVE.parametre == name].LoD.iloc[0])
    a_.hist(dev, bins=30, color='tab:blue', alpha=.7); a_.axvline(-lod, c='r', ls='--'); a_.axvline(lod, c='r', ls='--', label='seuil de détection')
    a_.set_xlabel(f'{name} mesuré sur les patients (écart à la médiane)'); a_.legend(fontsize=8)
plt.suptitle('Écarts de positionnement mesurés sur 151 patients vs seuil de détection'); plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig6_patients_vs_seuil.png', dpi=150); plt.show()

import shutil
shutil.make_archive('/kaggle/working/resultats_article7_v3', 'zip', OUT)
print('Terminé en %.1f min → Output : resultats_article7_v3.zip' % ((time.time() - T0) / 60))
