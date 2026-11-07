"""XAI on the segmentation-classifier (frozen CT-FM encoder + trained decoder) and a head-to-head with the OLD masked-pool model.

Target scalar S = case score from the logit map inside the brain mask (default pooling lse2, chosen on validation).
  attr     compute Grad-CAM (encoder stages 2/3/4, ReLU kept) + IG (24-step Gauss-Legendre; baselines IG_u uniform 0.5 in head, IG_bl blurred case)
           + SEGP (sigmoid of the logit map, a lesion-detector reference) for the same 55 cases that have OLD attributions in ref/attributions.
  analyze  block-level (8 voxels = 20 mm, eligible >= 25 % head) lesion enrichment vs the population prior, voxel-level lesion contrast,
           agreement between methods, spatial-prior strength; NEW and OLD maps scored by the SAME code.
  faith    deletion (M = highest score first, L = lowest first, prior orders, random) with blur / uniform replacement inside the head; effect on S.

  python segcls_xai.py attr    --pack PACK --ckpt RUN/best_auc.pt --out RUN/xai
  python segcls_xai.py analyze --pack PACK --out RUN/xai
  python segcls_xai.py faith   --pack PACK --ckpt RUN/best_auc.pt --out RUN/xai
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from segcls_model import SegCls, case_scores

B = 8
SIGMA = 2.0


# ------------------------------------------------------------------ data
def load_case(pack, cid):
    d = torch.load(Path(pack) / "data" / "cases" / f"{cid}.pt", weights_only=False)
    return dict(id=cid, x=d["x"].float(), brain=d["brain"].bool(), head=d["head"].bool(), lesion=d["lesion"].bool(), label=int(d["meta"]["label"]))


def xai_ids(pack):
    return sorted(p.stem for p in (Path(pack) / "ref" / "attributions").glob("*.pt"))


def load_model(pack, ckpt, dev):
    m = SegCls(None if os.environ.get("SEGCLS_NOPRE") else Path(pack) / "weights" / "ct_fm_feature_extractor").to(dev)
    m.decoder.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["decoder"])
    return m.eval()


class Scorer:
    """x (n,1,D,H,W) -> S (n,)"""
    def __init__(self, model, brain, pool):
        self.model, self.brain, self.pool = model, brain, pool

    def __call__(self, x):
        z = self.model(x)
        return case_scores(z, self.brain[None, None].expand(x.shape[0], 1, *self.brain.shape))[self.pool]


# ------------------------------------------------------------------ attributions
def gradcam(model, x, brain, pool, stages=(2, 3, 4)):
    fs = [f.detach().requires_grad_() for f in model.features(x)]
    z = model(x, feats=fs)
    S = case_scores(z, brain[None, None])[pool]
    gs = torch.autograd.grad(S.sum(), fs)
    out = {}
    for k in stages:
        a = gs[k].mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((a * fs[k]).sum(1, keepdim=True)).detach()
        out[f"GradCAM_s{k}"] = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0].cpu()
    return float(S), out, torch.sigmoid(z)[0, 0].detach().cpu()


def integrated_gradients(score, x, base, steps=24, chunk=2):
    nodes, w = np.polynomial.legendre.leggauss(steps)
    al, w = (nodes + 1) / 2, w / 2
    diff = x - base
    tot = torch.zeros_like(x)
    for i in range(0, steps, chunk):
        a = torch.tensor(al[i:i + chunk], dtype=x.dtype, device=x.device).view(-1, 1, 1, 1, 1)
        ww = torch.tensor(w[i:i + chunk], dtype=x.dtype, device=x.device).view(-1, 1, 1, 1, 1)
        xs = (base + a * diff).requires_grad_(True)
        g, = torch.autograd.grad(score(xs).sum(), xs)
        tot += (ww * g).sum(0, keepdim=True)
    attr = (diff * tot).detach()
    with torch.no_grad():
        s_x, s_0 = float(score(x)[0]), float(score(base)[0])
    return attr[0, 0].cpu(), dict(S=s_x, S_base=s_0, err=float(attr.sum()) - (s_x - s_0))


def blur_in_head(x, head, s=SIGMA):
    xa, h = x[0].cpu().numpy(), head.cpu().numpy().astype(np.float32)
    num, den = ndi.gaussian_filter(xa * h, s), ndi.gaussian_filter(h, s)
    return torch.from_numpy(np.where(h > 0, num / np.maximum(den, 1e-6), 0.0)).float()[None]


def uniform_in_head(head, v=0.5):
    return torch.where(head[None], torch.tensor(float(v)), torch.tensor(0.0))


def cmd_attr(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / "attributions"; out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.pack, a.ckpt, dev)
    ids = xai_ids(a.pack)
    if a.max_cases:
        ids = ids[:a.max_cases]
    print(f"{len(ids)} cases, pool {a.pool}, steps {a.steps}", flush=True)
    t0, errs = time.time(), []
    for n, cid in enumerate(ids):
        f = out / f"{cid}.pt"
        d = torch.load(f, weights_only=True) if f.exists() else {"id": cid}
        c = load_case(a.pack, cid)
        x, brain, head = c["x"][None].to(dev), c["brain"].to(dev), c["head"]
        if "GradCAM_s4" not in d:
            S, maps, p = gradcam(model, x, brain, a.pool)
            d["S"] = S; d["SEGP"] = p.half()
            for k, v in maps.items():
                d[k] = v.half()
        score = Scorer(model, brain, a.pool)
        for key, base in (("IG_u", uniform_in_head(head)), ("IG_bl", blur_in_head(c["x"], head))):
            if key in d:
                continue
            attr, info = integrated_gradients(score, x, base[None].to(dev), a.steps, a.chunk)
            d[key] = attr.half(); d[key + "_err"] = info["err"]; d[key + "_base_S"] = info["S_base"]
            errs.append((abs(info["err"]), abs(info["S"] - info["S_base"])))
        torch.save(d, f)
        print(f"  {n + 1}/{len(ids)} {cid} ({time.time() - t0:.0f}s)", flush=True)
    if errs:
        e = np.array(errs)
        print(f"IG completeness error: median {np.median(e[:, 0]):.3f}, 95th pct {np.percentile(e[:, 0], 95):.3f}, max {e[:, 0].max():.3f}; "
              f"typical |S - S_base| = {np.median(e[:, 1]):.2f}")


# ------------------------------------------------------------------ block machinery
def block_stats(head):
    D, H, W = head.shape
    iz, iy, ix = torch.meshgrid(torch.arange(D) // B, torch.arange(H) // B, torch.arange(W) // B, indexing="ij")
    nh, nw = (H + B - 1) // B, (W + B - 1) // B
    bid = (iz * nh + iy) * nw + ix
    nb = int(bid.max()) + 1
    cnt = torch.zeros(nb).scatter_add_(0, bid.flatten(), head.flatten().float())
    size = torch.zeros(nb).scatter_add_(0, bid.flatten(), torch.ones(bid.numel()))
    return bid, nb, cnt >= 0.25 * size, cnt


def block_mean(m, head, bid, nb, cnt):
    s = torch.zeros(nb).scatter_add_(0, bid.flatten(), (m * head).flatten().float())
    return s / cnt.clamp(min=1)


def block_z_index(shape, nb):
    D, H, W = shape; nh, nw = (H + B - 1) // B, (W + B - 1) // B
    return (torch.arange(nb) // (nh * nw)).float()


def zscore(s, elig):
    v = s[elig]
    return torch.where(elig, (s - v.mean()) / v.std().clamp(min=1e-9), torch.zeros_like(s))


def order_desc(score, elig, gen, ascending=False):
    s = score.clone().double()
    s = s + torch.rand(len(s), generator=gen).double() * 1e-9 * (s.abs().max() + 1e-12)
    s[~elig] = float("inf") if ascending else float("-inf")
    o = torch.argsort(s, descending=not ascending)
    return o[:int(elig.sum())]


# ------------------------------------------------------------------ analysis
NEW_MAPS = {"GC2": ("GradCAM_s2", "relu"), "GC3": ("GradCAM_s3", "relu"), "GC4": ("GradCAM_s4", "relu"),
            "IG_u": ("IG_u", "net"), "IG_bl": ("IG_bl", "net"), "SEGP": ("SEGP", "relu")}
OLD_MAPS = {k: v for k, v in NEW_MAPS.items() if k != "SEGP"}
PAIRS = [("GC3", "IG_u"), ("GC3", "IG_bl"), ("GC4", "IG_u"), ("GC4", "IG_bl"), ("IG_u", "IG_bl"), ("GC3", "GC4"),
         ("GC3", "SEGP"), ("GC4", "SEGP"), ("IG_u", "SEGP"), ("IG_bl", "SEGP")]      # SEGP pairs exist for the NEW model only


def case_blocks(pack, cid, attr_dir, maps, segp_dir=None):
    c = load_case(pack, cid)
    d = torch.load(Path(attr_dir) / f"{cid}.pt", weights_only=True)
    bid, nb, elig, cnt = block_stats(c["head"])
    sc, vox, pred = {}, {}, {}
    R = None
    if segp_dir is not None:                                   # region the NEW model predicts as bleeding: top 0.5 % SEGP voxels inside the head
        sp = torch.load(Path(segp_dir) / f"{cid}.pt", weights_only=True)["SEGP"].float()
        kk = max(1, int(0.005 * int(c["head"].sum())))
        thr = sp[c["head"]].topk(kk).values[-1]
        R = (sp >= thr) & c["head"]
    for n, (k, kind) in maps.items():
        if k not in d:
            continue
        m = d[k].float()
        m = m.clamp(min=0) if kind == "relu" else m
        sc[n] = block_mean(m, c["head"], bid, nb, cnt)
        if R is not None:
            sdh = m[c["head"]].std().clamp(min=1e-12)
            pr = dict(d=float((m[R].mean() - m[c["head"] & ~R].mean()) / sdh))
            if kind == "relu" and float(m[c["head"]].sum()) > 0:
                pr["er"] = float(m[R].sum() / m[c["head"]].sum() / (R.sum() / c["head"].sum()))
            pred[n] = pr
        les = c["lesion"] & c["head"]; oth = c["head"] & ~c["lesion"]
        if les.any():
            sd = m[c["head"]].std().clamp(min=1e-12)
            vox[n] = float((m[les].mean() - m[oth].mean()) / sd)
    lcnt = torch.zeros(nb).scatter_add_(0, bid.flatten(), c["lesion"].flatten().float())
    det = float(((d["SEGP"].float() > 0.5) & c["lesion"] & c["head"]).any()) if "SEGP" in d else np.nan   # lesion overlapped by the predicted mask
    return dict(id=cid, label=c["label"], det=det, pred=pred, bid=bid, nb=nb, elig=elig, sc=sc, vox=vox, lesblk=(lcnt >= 5) & elig, shape=tuple(c["head"].shape))


def prior_maps(cases, names):
    out = {}
    for n in names:
        Z = torch.stack([zscore(c["sc"][n], c["elig"]) for c in cases]); E = torch.stack([c["elig"].float() for c in cases])
        out[n] = [(Z.sum(0) - Z[i]) / (E.sum(0) - E[i]).clamp(min=1) for i in range(len(cases))]
    return out


def rho(a, b, elig):
    a, b = a[elig].numpy(), b[elig].numpy()
    return float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else np.nan


def analyse_model(pack, attr_dir, maps, tag, segp_dir=None):
    ids = sorted(p.stem for p in Path(attr_dir).glob("*.pt"))
    cases = [case_blocks(pack, i, attr_dir, maps, segp_dir) for i in ids]
    names = [n for n in maps if all(n in c["sc"] for c in cases)]
    pri = prior_maps(cases, names)
    rows = []
    for i, c in enumerate(cases):
        zi = block_z_index(c["shape"], c["nb"])
        for n in names:
            r = dict(model=tag, id=c["id"], label=c["label"], det=c["det"], method=n, rho_z=rho(c["sc"][n], zi, c["elig"]),
                     rho_prior=rho(c["sc"][n], pri[n][i], c["elig"]), vox_d=c["vox"].get(n, np.nan),
                     pred_d=c["pred"].get(n, {}).get("d", np.nan), pred_er=c["pred"].get(n, {}).get("er", np.nan))
            y = c["lesblk"][c["elig"]].numpy()
            if c["label"] == 1 and 0 < y.sum() < len(y):
                s, p = c["sc"][n][c["elig"]].numpy(), pri[n][i][c["elig"]].numpy()
                r["auc"], r["auc_prior"] = roc_auc_score(y, s), roc_auc_score(y, p)
                k = max(1, int(round(0.1 * len(y))))
                r["prec10"], r["prec10_prior"], r["base_rate"] = y[np.argsort(-s)[:k]].mean(), y[np.argsort(-p)[:k]].mean(), y.mean()
            rows.append(r)
        for a_, b_ in PAIRS:
            if a_ in names and b_ in names:
                pa = [pri[a_][i], pri[b_][i]]
                rows.append(dict(model=tag, id=c["id"], label=c["label"], method=f"{a_}~{b_}", rho_pair=rho(c["sc"][a_], c["sc"][b_], c["elig"]),
                                 rho_pair_prior=np.nanmean([rho(c["sc"][a_], pa[1], c["elig"]), rho(c["sc"][b_], pa[0], c["elig"])])))
    return pd.DataFrame(rows)


def wil(x, y=None):
    x = np.asarray(x, float) - (0 if y is None else np.asarray(y, float)); x = x[np.isfinite(x)]
    return float(wilcoxon(x)[1]) if len(x) >= 6 and np.any(x != 0) else np.nan


def cmd_analyze(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new = analyse_model(a.pack, out / "attributions", NEW_MAPS, "new", out / "attributions")
    old = analyse_model(a.pack, Path(a.pack) / "ref" / "attributions", OLD_MAPS, "old", out / "attributions")   # same region R (new model prediction) for both
    df = pd.concat([new, old]); df.to_csv(out / "analysis_cases.csv", index=False)
    f = open(out / "summary_analysis.txt", "w")
    P = lambda *s: (print(*s), print(*s, file=f))
    pos = df[df.label == 1]
    P("=== 1. LESION ENRICHMENT, block level (positives; lesion block = eligible block with >= 5 lesion voxels) ===")
    P("auc = within-case AUC of block score for lesion vs other eligible blocks; auc_prior = same with the leave-one-out population prior (location only);")
    P("gain = auc - auc_prior; prec10 = share of lesion blocks among the top-10% blocks; base_rate = share of lesion blocks overall.")
    rows = []
    for (n, m), g in pos[pos.method.isin(list(NEW_MAPS))].groupby(["method", "model"]):
        g = g.dropna(subset=["auc"])
        rows.append(dict(method=n, model=m, n=len(g), auc=g.auc.mean(), auc_prior=g.auc_prior.mean(), gain=(g.auc - g.auc_prior).mean(), p_gain=wil(g.auc, g.auc_prior),
                         prec10=g.prec10.mean(), prec10_prior=g.prec10_prior.mean(), base_rate=g.base_rate.mean()))
    t = pd.DataFrame(rows).sort_values(["method", "model"]); P(t.round(3).to_string(index=False))
    P("\nPaired NEW vs OLD on the same positives (gain = auc - auc_prior):")
    for n in OLD_MAPS:
        a_, b_ = [pos[(pos.method == n) & (pos.model == m)].set_index("id") for m in ("new", "old")]
        j = a_.join(b_, lsuffix="_new", rsuffix="_old", how="inner").dropna(subset=["auc_new", "auc_old"])
        if len(j):
            dn, do = j.auc_new - j.auc_prior_new, j.auc_old - j.auc_prior_old
            P(f"  {n:6s} gain new {dn.mean():+.3f}  old {do.mean():+.3f}  diff {(dn - do).mean():+.3f}  p = {wil(dn, do):.3g}  (n = {len(j)})")
    nw = pos[(pos.model == "new") & pos.method.isin(list(NEW_MAPS))].dropna(subset=["auc"])
    P("\n-- NEW model, positives split by whether the predicted mask (SEGP > 0.5) overlaps the lesion (det = 1) or not (det = 0):")
    rows = []
    for (n, dt), g in nw.groupby(["method", "det"]):
        rows.append(dict(method=n, det=int(dt), n=len(g), auc=g.auc.mean(), auc_prior=g.auc_prior.mean(), gain=(g.auc - g.auc_prior).mean(), prec10=g.prec10.mean()))
    P(pd.DataFrame(rows).round(3).to_string(index=False))
    P("\n=== 2. LESION CONTRAST, voxel level: d = (mean attribution inside lesion - mean outside lesion, in head) / SD in head; share of cases with d > 0 ===")
    rows = []
    for (m, n), g in pos[pos.method.isin(list(NEW_MAPS))].groupby(["model", "method"]):
        g = g.dropna(subset=["vox_d"]); rows.append(dict(model=m, method=n, n=len(g), d_mean=g.vox_d.mean(), d_median=g.vox_d.median(), share_pos=(g.vox_d > 0).mean()))
    P(pd.DataFrame(rows).sort_values(["method", "model"]).round(3).to_string(index=False))
    P("\n=== 2b. ENERGY IN THE PREDICTED REGION (all 55 cases): R = top 0.5% voxels of the NEW model's SEGP inside the head (same R for new and old maps) ===")
    P("d = (mean attribution in R - mean outside R, in head) / SD in head;  er = (share of Grad-CAM mass in R) / (share of head volume in R)  (Grad-CAM only; 1 = no enrichment)")
    al = df[~df.method.str.contains("~") & (df.method != "SEGP")]
    t = al.groupby(["method", "model"]).agg(n=("id", "size"), d=("pred_d", "mean"), d_share_pos=("pred_d", lambda v: (v > 0).mean()), er=("pred_er", "mean"))
    P(t.round(3).to_string())
    P("Paired NEW vs OLD on the same cases (d):")
    for n in OLD_MAPS:
        a_, b_ = [al[(al.method == n) & (al.model == m)].set_index("id").pred_d for m in ("new", "old")]
        j = pd.concat([a_, b_], axis=1, keys=["new", "old"]).dropna()
        P(f"  {n:6s} new {j.new.mean():+.3f}  old {j.old.mean():+.3f}  diff {(j.new - j.old).mean():+.3f}  p = {wil(j.new, j.old):.3g}  (n = {len(j)})")
    P("\n=== 3. AGREEMENT between methods (block Spearman; prior = same agreement with the other method's leave-one-out prior), positives and negatives ===")
    ag = df[df.method.str.contains("~")]
    t = ag.groupby(["method", "model"]).agg(n=("id", "size"), rho=("rho_pair", "mean"), rho_prior=("rho_pair_prior", "mean")); t["gain"] = t.rho - t.rho_prior
    P(t.round(3).to_string())
    P("\n=== 4. SPATIAL PRIOR: rho_z = Spearman(block score, z index; LOW index = inferior); rho_prior = Spearman(block score, own leave-one-out population prior) ===")
    sp = df[~df.method.str.contains("~")].groupby(["method", "model"]).agg(rho_z=("rho_z", "mean"), rho_prior=("rho_prior", "mean"))
    P(sp.round(3).to_string())
    P("\nReading: if enrichment gain over the prior rises for the NEW model, adding lesion supervision made the attributions more lesion-specific; "
      "SEGP is the lesion detector's own output (upper reference). Gain ~ 0 means the method follows location, not the lesion.")
    f.close()


# ------------------------------------------------------------------ faithfulness (deletion on S)
FR = (0.1, 0.2, 0.3, 0.5)


@torch.no_grad()
def cmd_faith(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.pack, a.ckpt, dev)
    out = Path(a.out)
    ids = [i for i in xai_ids(a.pack) if load_case(a.pack, i)["label"] == 1]
    if a.max_cases:
        ids = ids[:a.max_cases]
    cases = [case_blocks(a.pack, i, out / "attributions", NEW_MAPS) for i in ids]
    names = ["GC3", "GC4", "IG_u", "IG_bl", "SEGP"]
    pri = prior_maps(cases, names)
    rows = []
    for i, cs in enumerate(cases):
        c = load_case(a.pack, cs["id"]); gen = torch.Generator().manual_seed(i)
        x, brain, head = c["x"].to(dev), c["brain"].to(dev), c["head"]
        score = Scorer(model, brain, a.pool)
        S0 = float(score(x[None])[0])
        repl = {"blur": blur_in_head(c["x"], head).to(dev), "uniform": uniform_in_head(head).to(dev)}
        elig = cs["elig"]; nel = int(elig.sum())
        orders = {}
        for n in names:
            orders[f"{n}|M"] = order_desc(cs["sc"][n], elig, gen); orders[f"{n}|L"] = order_desc(cs["sc"][n], elig, gen, True)
            orders[f"{n}|pM"] = order_desc(pri[n][i], elig, gen); orders[f"{n}|pL"] = order_desc(pri[n][i], elig, gen, True)
        for r_ in range(3):
            orders[f"random{r_}"] = torch.where(elig)[0][torch.randperm(nel, generator=gen)]
        bid = cs["bid"]
        for rn, rv in repl.items():
            for on, o in orders.items():
                vols = []
                for fr in FR:
                    memb = torch.zeros(cs["nb"], dtype=torch.bool); memb[o[:max(1, int(round(fr * nel)))]] = True
                    vols.append(torch.where((memb[bid] & head)[None].to(dev), rv, x))
                S = score(torch.stack(vols))
                for fr, s in zip(FR, S.tolist()):
                    rows.append(dict(id=cs["id"], repl=rn, order=on, frac=fr, dS=s - S0, S0=S0))
        print(f"  faith {i + 1}/{len(cases)} {cs['id']}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "faith_curves.csv", index=False)
    f = open(out / "summary_faith.txt", "w"); P = lambda *s: (print(*s), print(*s, file=f))
    P("=== DELETION on the case score S (dS = S(replaced) - S(original); mean over positives). M: highest-scoring blocks first; L: lowest first; pM/pL: same with the population prior; random: mean of 3 draws ===")
    P(f"mean S0 over cases: {df.drop_duplicates('id').S0.mean():.2f}")
    for rn in repl:
        g = df[df.repl == rn]
        wide = g.pivot_table(index=["id", "order"], columns="frac", values="dS").reset_index()
        wide["auc"] = wide[list(FR)].mean(axis=1)
        P(f"\n-- replacement: {rn}.   columns = replaced fraction of eligible blocks; auc = mean over the 4 fractions")
        rnd = wide[wide.order.str.startswith("random")].groupby("id")[list(FR) + ["auc"]].mean()
        tab = {"random": rnd.mean()}
        for n in names:
            for suf in ("M", "L", "pM", "pL"):
                w = wide[wide.order == f"{n}|{suf}"].set_index("id")
                tab[f"{n}|{suf}"] = w[list(FR) + ["auc"]].mean()
        P(pd.DataFrame(tab).T.round(3).to_string())
        P("   gap = AUC(L) - AUC(M);  gap gain = gap - prior gap (pL - pM);  p = paired Wilcoxon of the gain over cases")
        for n in names:
            gm, gl = [wide[wide.order == f"{n}|{s}"].set_index("id").auc for s in ("M", "L")]
            pm, pl = [wide[wide.order == f"{n}|{s}"].set_index("id").auc for s in ("pM", "pL")]
            gap, pgap = gl - gm, pl - pm
            P(f"   {n:6s} gap {gap.mean():+.3f}  prior gap {pgap.mean():+.3f}  gain {(gap - pgap).mean():+.3f}  p = {wil(gap, pgap):.3g}   "
              f"M vs random {(gm - rnd.auc).mean():+.3f}")
    P("\nNote: dS is in units of the segmentation-classifier score, not the old logit; compare signs and gains over the prior, not raw magnitudes. "
      "IG_u is matched with the uniform replacement and IG_bl with the blur replacement (partly true by IG completeness).")
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attr", "analyze", "faith"])
    ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt")
    ap.add_argument("--pool", default="lse2"); ap.add_argument("--steps", type=int, default=24); ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--max-cases", type=int, default=None)
    a = ap.parse_args()
    {"attr": cmd_attr, "analyze": cmd_analyze, "faith": cmd_faith}[a.cmd](a)


if __name__ == "__main__":
    main()
