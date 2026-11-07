# === Article 7 — v2 (script unique) ===
import os, sys
# === 0. Configuration ===
import os, sys, glob, time, json, random, math
import numpy as np, cv2, pandas as pd, torch, torch.nn as nn
os.system('pip -q install segmentation-models-pytorch timm')
import timm, segmentation_models_pytorch as smp
T_START = time.time(); TIME_BUDGET_H = 10.5
def _find(pattern, required=True):
    r = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    assert r or not required, f'Introuvable dans /kaggle/input : {pattern}'
    return r[0] if r else None
CODE_DIR = os.path.dirname(_find('a7_core.py'))
sys.path.insert(0, CODE_DIR); sys.path.insert(0, '/kaggle/working/code7')
import pano_artifacts as PA, pano_calibrated as PC, a7_core as C, a7_core2 as G
META_CSV = _find('metadata_157.csv'); ANNOT_XLSX = _find('grille_annotation_artefacts-remplis.xlsx')
IMG_DIR = os.path.dirname(_find('IMG_0001.png'))
assert not glob.glob('/kaggle/input/**/CONFIDENTIEL*', recursive=True), 'Retirez CONFIDENTIEL_correspondance.csv !'
V1_CKPT = _find('convnext_tiny__geo+perturbations.best.pt', required=False)
print('Modèle v1 trouvé :', V1_CKPT)
OUT = '/kaggle/working/resultats7v2'
for d in ['ckpt', 'figures', 'controle_masques']:
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Appareil :', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '(PAS DE GPU)')
SEED = 42; H, W = 512, 1024
EPOCHS = 20; REPEAT = 4; BATCH = 12; LR = 3e-4; NUM_WORKERS = 4
SEG_EPOCHS = 15
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
import matplotlib.pyplot as plt

# === 1. Données : Owandy (test seulement) et STS (entraînement) ===
meta = pd.read_csv(META_CSV)
meta = meta[meta.duplicate_of.isna() & meta.flag.isna()] if {'duplicate_of', 'flag'} <= set(meta.columns) else meta
ann = pd.read_excel(ANNOT_XLSX, sheet_name='Annotation'); ann = ann[ann['Qualité globale (1-3)'].notna()]
df = meta.merge(ann, on=['image_id', 'patient_id'], how='inner'); print('Images Owandy :', len(df))
REAL = np.stack([cv2.resize(cv2.imread(f'{IMG_DIR}/{i}.png', 0), (W, H), interpolation=cv2.INTER_AREA) for i in df.image_id])
TARGET_CDF = G.mean_cdf(REAL)   # niveaux de gris moyens Owandy (aucune étiquette utilisée)

from datasets import load_dataset
def load_sts():
    for rid in ['Angelou0516/STS-2D-Tooth', 'MedOtter/STS-2D-Tooth']:
        try:
            return load_dataset(rid, split='a_pxi_labeled'), load_dataset(rid, split='c_pxi_labeled')
        except Exception as e:
            print(rid, '->', repr(e)[:200])
    raise RuntimeError('STS-2D-Tooth indisponible (Internet On ?)')
a_ds, c_ds = load_sts()
X, Y = [], []
for ds in (a_ds, c_ds):
    for ex in ds:
        im = cv2.resize(np.array(ex['image'].convert('L')), (W, H), interpolation=cv2.INTER_AREA)
        mk = cv2.resize((np.array(ex['mask'].convert('L')) > 127).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        X.append(G.match_to_cdf(im, TARGET_CDF)); Y.append(mk)
X = np.stack(X); Y = np.stack(Y)
if Y.mean() > 0.5: Y = 1 - Y
rs = np.random.RandomState(SEED); idx = rs.permutation(len(X)); n = len(X)
TE, VA, TR = idx[:int(.1 * n)], idx[int(.1 * n):int(.2 * n)], idx[int(.2 * n):]
print('STS train/val/test :', len(TR), len(VA), len(TE))

fig, ax = plt.subplots(2, 3, figsize=(18, 6)); b = X[TR[0]]
demo = [('aucune', 0, 0, 0), ('s=-0.3', -.3, 0, 0), ('s=+0.3', .3, 0, 0), ('menton +15°', 0, 15, 0), ('menton -15°', 0, -15, 0), ('inclinaison +4°', 0, 0, 4)]
for a_, (t, s, th, ph) in zip(ax.ravel(), demo):
    a_.imshow(G.simulate_v2(b, s, th, ph, np.random.default_rng(1), nuisance=False, band=False), cmap='gray'); a_.set_title(t); a_.axis('off')
plt.suptitle('Simulation v2 (niveaux de gris adaptés à Owandy)'); plt.tight_layout(); plt.savefig(f'{OUT}/figures/fig1_simulations_v2.png', dpi=130); plt.show()

# === 2. Jeux simulés v2 ===
MEAN, STD = 0.45, 0.225
def to_tensor(im):
    x = (im.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(np.repeat(x[None], 3, 0))
REGIMES = {'v2': True, 'v2_sans_flou': False}   # le second teste si le flou est le raccourci
class SimDS(torch.utils.data.Dataset):
    def __init__(self, ids, use_blur):
        self.ids, self.use_blur = ids, use_blur
    def __len__(self): return len(self.ids) * REPEAT
    def __getitem__(self, i):
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        im, y = G.make_sample_v2(X[self.ids[i % len(self.ids)]], rng, True, self.use_blur)
        return to_tensor(im), torch.from_numpy(y / G.TARGET_SCALE)
def fixed_set(ids, draws, seed0):
    ims, ys = [], []
    for j, t in enumerate(ids):
        for d in range(draws):
            rng = np.random.default_rng(seed0 + 1000 * d + j)
            im, y = G.make_sample_v2(X[t], rng, True, True); ims.append(im); ys.append(y)
    return np.stack(ims), np.stack(ys)
VAL_IM, VAL_Y = fixed_set(VA, 3, 10_000)
TEST_IM, TEST_Y = fixed_set(TE, 5, 20_000)
def wif(worker_id): np.random.seed((torch.initial_seed() + worker_id) % 2**32)

# === 3. Entraînement des modèles v2 (ConvNeXt-T) ===
loss_fn = nn.SmoothL1Loss(beta=0.1)
@torch.no_grad()
def predict(model, ims, bs=24):
    model.eval(); out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[i:i + bs]]).to(DEV)
        with torch.autocast(device_type='cuda', enabled=DEV == 'cuda'):
            p = model(x).float(); pf = model(x.flip(-1)).float() * torch.tensor([1., 1., -1.], device=DEV)
        out.append(((p + pf) / 2).cpu().numpy())
    return np.concatenate(out) * G.TARGET_SCALE
def mae(P, Y): return np.abs(P - Y).mean(0)

def train_one(rname):
    tag = f'convnext_tiny__{rname}'; ck = f'{OUT}/ckpt/{tag}'
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(TR) * REPEAT / BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=.1)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == 'cuda'); best, log = 1e9, []
    dl = torch.utils.data.DataLoader(SimDS(TR, REGIMES[rname]), batch_size=BATCH, shuffle=True, drop_last=True,
                                     num_workers=NUM_WORKERS, worker_init_fn=wif, persistent_workers=True)
    for ep in range(EPOCHS):
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H - 3: print('Budget temps'); break
        model.train(); tl = []
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(device_type='cuda', enabled=DEV == 'cuda'):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl.append(loss.item())
        m = mae(predict(model, VAL_IM), VAL_Y); score = float((m / G.TARGET_SCALE).mean())
        log.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_mae_s=m[0], val_mae_theta=m[1], val_mae_phi=m[2], val_score=score))
        print(f'{tag} ep{ep:02d} loss={np.mean(tl):.4f} | val MAE s={m[0]:.4f} θ={m[1]:.2f}° φ={m[2]:.2f}°')
        if score < best: best = score; torch.save(model.state_dict(), ck + '.best.pt')
    pd.DataFrame(log).to_csv(ck + '_log.csv', index=False)
    model.load_state_dict(torch.load(ck + '.best.pt', map_location=DEV)); return model

MODELS = {r: train_one(r) for r in REGIMES}
if V1_CKPT:
    m1 = timm.create_model('convnext_tiny', pretrained=False, num_classes=3).to(DEV)
    m1.load_state_dict(torch.load(V1_CKPT, map_location=DEV)); MODELS['v1'] = m1
print('Modèles :', list(MODELS))

# === 4. E1-v2 (vérité connue) et D1 (erreurs ajoutées aux images Owandy) ===
rows = []
for name, mod in MODELS.items():
    P = predict(mod, TEST_IM)
    d = dict(modele=name)
    for k, t in enumerate(G.TARGET_NAMES):
        d[f'MAE_{t}'] = float(np.abs(P[:, k] - TEST_Y[:, k]).mean()); d[f'r_{t}'] = float(np.corrcoef(P[:, k], TEST_Y[:, k])[0, 1])
    rows.append(d)
E1 = pd.DataFrame(rows); E1.to_csv(f'{OUT}/E1v2_test_simule.csv', index=False); print(E1.round(3).to_string())

# D1 : on ajoute une erreur CONNUE à chaque image réelle ; l'erreur propre de l'image
# s'annule dans la différence prédiction(simulée) - prédiction(originale).
D1_IM, D1_Y, D1_BASE = [], [], []
for j in range(len(REAL)):
    for d in range(2):
        rng = np.random.default_rng(50_000 + 1000 * d + j)
        s, th, ph = C.draw_targets(rng)
        D1_IM.append(G.simulate_v2(REAL[j], s, th, ph, rng, nuisance=False, band=False)); D1_Y.append([s, th, ph]); D1_BASE.append(j)
D1_IM = np.stack(D1_IM); D1_Y = np.array(D1_Y, np.float32); D1_BASE = np.array(D1_BASE)
rows = []
for name, mod in MODELS.items():
    P0 = predict(mod, REAL); P1 = predict(mod, D1_IM); dP = P1 - P0[D1_BASE]
    d = dict(modele=name)
    for k, t in enumerate(G.TARGET_NAMES):
        d[f'r_{t}'] = float(np.corrcoef(dP[:, k], D1_Y[:, k])[0, 1]); d[f'pente_{t}'] = float(np.polyfit(D1_Y[:, k], dP[:, k], 1)[0])
        d[f'MAE_{t}'] = float(np.abs(dP[:, k] - D1_Y[:, k]).mean())
    rows.append(d)
D1 = pd.DataFrame(rows); D1.to_csv(f'{OUT}/D1_owandy_erreurs_ajoutees.csv', index=False)
print('\nD1 — pente 1 et r proche de 1 = le modèle voit les erreurs sur l anatomie Owandy'); print(D1.round(3).to_string())

# === 5. Segmentation des dents (pour la mesure géométrique G) ===
seg = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(DEV)
dice_loss = smp.losses.DiceLoss('binary', from_logits=True); bce = nn.BCEWithLogitsLoss()
class SegDS(torch.utils.data.Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1)); im, mk = X[self.ids[i]], Y[self.ids[i]]
        if rng.random() < .5: im, mk = im[:, ::-1].copy(), mk[:, ::-1].copy()
        im = C.photometric(im, rng)
        return to_tensor(im), torch.from_numpy(mk[None].astype(np.float32))
opt = torch.optim.AdamW(seg.parameters(), lr=3e-4); scaler = torch.cuda.amp.GradScaler(enabled=DEV == 'cuda')
dl = torch.utils.data.DataLoader(SegDS(TR), batch_size=6, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, worker_init_fn=wif)
for ep in range(SEG_EPOCHS):
    seg.train(); tl = []
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast(device_type='cuda', enabled=DEV == 'cuda'):
            o = seg(x); loss = dice_loss(o, y) + bce(o.float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl.append(loss.item())
    print(f'segmentation ep{ep:02d} loss={np.mean(tl):.4f}')
@torch.no_grad()
def seg_predict(ims, bs=8):
    seg.eval(); out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[i:i + bs]]).to(DEV)
        with torch.autocast(device_type='cuda', enabled=DEV == 'cuda'):
            p = (torch.sigmoid(seg(x)) + torch.sigmoid(seg(x.flip(-1))).flip(-1)) / 2
        out.append((p[:, 0] > .5).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)
Mte = seg_predict(X[TE])
dice = [2 * (p & g).sum() / max(p.sum() + g.sum(), 1) for p, g in zip(Mte.astype(bool), Y[TE].astype(bool))]
print('Dice segmentation (test STS) :', round(float(np.mean(dice)), 4))
M_REAL = seg_predict(REAL)
for j in range(6):   # contrôle visuel
    ov = cv2.cvtColor(REAL[j], cv2.COLOR_GRAY2BGR); cnt, _ = cv2.findContours(M_REAL[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (0, 0, 255), 1); xs, ys = G.occlusal_line(M_REAL[j])
    for x_, y_ in zip(xs[::4], ys[::4]): cv2.circle(ov, (int(x_), int(y_)), 1, (0, 255, 0), -1)
    cv2.imwrite(f'{OUT}/controle_masques/{df.image_id.iloc[j]}.jpg', ov)

# === 6. G — mesure géométrique : validation (STS simulé, D1) puis images réelles ===
def feats(ims, masks):
    return pd.DataFrame([G.geometric_features(i, m) for i, m in zip(ims, masks)])
# validation 1 : STS simulé, vérité connue
Gte = feats(TEST_IM, seg_predict(TEST_IM)); Gte[['s', 'theta', 'phi']] = TEST_Y
# validation 2 : D1 (erreurs ajoutées aux images Owandy), différences appariées
G0 = feats(REAL, M_REAL); G1 = feats(D1_IM, seg_predict(D1_IM))
rows = []
for name, gf, y, base in [('STS simulé', Gte, TEST_Y, None), ('Owandy + erreur ajoutée (D1)', G1, D1_Y, D1_BASE)]:
    for col, k in [('P_ratio', 0), ('theta_hat', 1), ('phi_hat', 2)]:
        v = gf[col].to_numpy(float) - (G0[col].to_numpy(float)[base] if base is not None else 0)
        ok = np.isfinite(v)
        rows.append(dict(jeu=name, mesure=col, cible=G.TARGET_NAMES[k], n=int(ok.sum()),
                         spearman=float(pd.Series(v[ok]).corr(pd.Series(y[ok, k]), method='spearman'))))
GVAL = pd.DataFrame(rows); GVAL.to_csv(f'{OUT}/G_validation.csv', index=False); print(GVAL.round(3).to_string())

out = pd.concat([df[['image_id']].reset_index(drop=True), G0], axis=1); out.to_csv(f'{OUT}/G_mesures_owandy.csv', index=False)
LABELS = list(C.LABEL_SCORES); rows = []
for lab in LABELS:
    y = df[lab].fillna(0).to_numpy() >= 1; sc = G.geo_score(G0, lab)
    a, lo, hi = C.auc_ci(y, sc); rows.append(dict(methode='G géométrique', erreur=lab, n_pos=int(y.sum()), AUC=a, IC95_bas=lo, IC95_haut=hi))
sg = []
n, acc, p = G.sign_test(df['Trop en arrière'].fillna(0) >= 1, df['Trop en avant'].fillna(0) >= 1, np.log(G0.P_ratio) - np.nanmedian(np.log(G0.P_ratio)))
sg.append(dict(methode='G géométrique', test='avant/arrière (P_ratio vs médiane)', n=n, exactitude=acc, p=p))
n, acc, p = G.sign_test(df['Menton trop bas'].fillna(0) >= 1, df['Menton trop haut'].fillna(0) >= 1, G0.theta_hat - np.nanmedian(G0.theta_hat))
sg.append(dict(methode='G géométrique', test='menton bas/haut (θ̂ vs médiane)', n=n, exactitude=acc, p=p))

# === E2-v2 : modèles appris ===
for name, mod in MODELS.items():
    pdf = pd.DataFrame(predict(mod, REAL), columns=G.TARGET_NAMES)
    pdf.assign(image_id=df.image_id.values).to_csv(f'{OUT}/E2v2_predictions_{name}.csv', index=False)
    for lab in LABELS:
        y = df[lab].fillna(0).to_numpy() >= 1; a, lo, hi = C.auc_ci(y, C.score_for(pdf, lab))
        rows.append(dict(methode=f'CNN {name}', erreur=lab, n_pos=int(y.sum()), AUC=a, IC95_bas=lo, IC95_haut=hi))
    n, acc, p = G.sign_test(df['Trop en arrière'].fillna(0) >= 1, df['Trop en avant'].fillna(0) >= 1, pdf.s - pdf.s.median())
    sg.append(dict(methode=f'CNN {name}', test='avant/arrière (s vs médiane)', n=n, exactitude=acc, p=p))
    n, acc, p = G.sign_test(df['Menton trop bas'].fillna(0) >= 1, df['Menton trop haut'].fillna(0) >= 1, pdf.theta_deg - pdf.theta_deg.median())
    sg.append(dict(methode=f'CNN {name}', test='menton bas/haut (θ vs médiane)', n=n, exactitude=acc, p=p))
E2 = pd.DataFrame(rows); E2.to_csv(f'{OUT}/E2v2_auc_reel.csv', index=False); print(E2.round(3).to_string())
SG = pd.DataFrame(sg); SG.to_csv(f'{OUT}/E2v2_test_du_signe.csv', index=False); print(SG.round(4).to_string())

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
grp = np.where(df['Trop en avant'].fillna(0) >= 1, 0, np.where(df['Trop en arrière'].fillna(0) >= 1, 2, 1))
ax[0].boxplot([G0.P_ratio[grp == k].dropna() for k in range(3)]); ax[0].set_xticks([1, 2, 3], ['avant', 'aucune', 'arrière']); ax[0].set_title('Largeur relative des incisives (G)')
grc = np.where(df['Menton trop haut'].fillna(0) >= 1, 0, np.where(df['Menton trop bas'].fillna(0) >= 1, 2, 1))
ax[1].boxplot([G0.theta_hat[grc == k].dropna() for k in range(3)]); ax[1].set_xticks([1, 2, 3], ['haut', 'aucune', 'bas']); ax[1].set_title('θ̂ courbure occlusale (°)')
gri = df['Inclinaison tête'].fillna(0).astype(int).to_numpy()
ax[2].boxplot([np.abs(G0.phi_hat[gri == k]).dropna() for k in sorted(set(gri))]); ax[2].set_title('|φ̂| pente occlusale (°) selon sévérité')
plt.suptitle('G — mesures physiques sur les images Owandy, selon le dentiste'); plt.tight_layout(); plt.savefig(f'{OUT}/figures/fig4_G_owandy.png', dpi=130); plt.show()

# === 7. Archive ===
import shutil
PACK = '/kaggle/working/pack'; shutil.rmtree(PACK, ignore_errors=True); os.makedirs(PACK)
for f in glob.glob(f'{OUT}/*.csv'): shutil.copy(f, PACK)
os.makedirs(f'{PACK}/logs'); [shutil.copy(f, f'{PACK}/logs') for f in glob.glob(f'{OUT}/ckpt/*_log.csv')]
shutil.copytree(f'{OUT}/figures', f'{PACK}/figures'); shutil.copytree(f'{OUT}/controle_masques', f'{PACK}/controle_masques')
json.dump(dict(regimes=list(REGIMES), v1_teste=bool(V1_CKPT), seg_dice=float(np.mean(dice)), seed=SEED, epochs=EPOCHS), open(f'{PACK}/config.json', 'w'), indent=2)
shutil.make_archive('/kaggle/working/resultats_article7_v2', 'zip', PACK); shutil.rmtree(PACK)
print('Terminé en %.1f h → Output : resultats_article7_v2.zip' % ((time.time() - T_START) / 3600))
