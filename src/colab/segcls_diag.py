"""Diagnostics for a trained segcls model (CSV analyses + threshold sweep).  Nothing is trained or selected on the test set.
  python segcls_diag.py --pack PACK --ckpt RUN/best_auc.pt --run RUN --eval RUN/eval_best_auc --ref PACK/ref/test_predictions_maskpool.csv
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from segcls_model import SegCls
from segcls_data import Cases, split_ids

VOX_ML = 2.5 ** 3 / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--val-fold", type=int, default=0); ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    P = lambda *s: print(*s, flush=True)
    log = pd.read_csv(Path(a.run) / "train_log.csv")
    P("=== training curve (every 5th epoch) ===")
    P(log[["epoch", "loss", "dice_pos", "fp_mass_neg", "auc_lse1", "auc_top200", "auc_softvol", "sec"]].iloc[::5].round(3).to_string(index=False))
    P(f"best val Dice at epoch {int(log.loc[log.dice_pos.idxmax(), 'epoch'])} ({log.dice_pos.max():.3f}); best val AUC(lse1) at epoch {int(log.loc[log.auc_lse1.idxmax(), 'epoch'])} ({log.auc_lse1.max():.3f}); last epoch {int(log.epoch.max())}")

    te = pd.read_csv(Path(a.eval) / "test_cases.csv"); pool = json.load(open(Path(a.eval) / "chosen_pooling.json"))["pooling"]
    pos = te[te.label == 1].copy(); pos["ml"] = pos.lesion_vox * VOX_ML
    pos["bin"] = pd.cut(pos.ml, [0, 1, 10, 1e9], labels=["<1 mL", "1-10 mL", ">10 mL"])
    P("\n=== TEST positives by lesion volume (threshold 0.5) ===")
    P(pos.groupby("bin", observed=True).agg(n=("id", "size"), overlap_frac=("dice", lambda d: (d > 0).mean()), mean_dice=("dice", "mean"),
                                           median_dice=("dice", "median")).round(3).to_string())

    ref = pd.read_csv(a.ref).set_index("id")
    d = te.assign(old=te.id.map(ref["score"]))
    y, new, old = d.label.values, d[pool].values, d.old.values
    rng = np.random.default_rng(0); P_, N_ = np.where(y == 1)[0], np.where(y == 0)[0]; diffs = []
    for _ in range(2000):
        i = np.r_[rng.choice(P_, len(P_)), rng.choice(N_, len(N_))]
        diffs.append(roc_auc_score(y[i], new[i]) - roc_auc_score(y[i], old[i]))
    P(f"\n=== TEST AUC, pooling '{pool}' (chosen on val) vs old masked-pool head, paired bootstrap ===")
    P(f"new {roc_auc_score(y, new):.3f}  old {roc_auc_score(y, old):.3f}  diff {np.mean(diffs):+.3f}  95% CI [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]  "
      f"Spearman(new, old) = {spearmanr(new, old)[0]:.2f}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegCls(None if a.no_pretrained else Path(a.pack) / "weights" / "ct_fm_feature_extractor").to(dev)
    model.decoder.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["decoder"]); model.eval()
    _, va, tes = split_ids(a.pack, a.val_fold)
    thrs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    res = {}
    for name, ids in (("val", list(va.id)), ("test", list(tes.id))):
        rows = []
        with torch.no_grad():
            for b in DataLoader(Cases(a.pack, ids), batch_size=1, num_workers=2):
                x, br, yy = b["x"].to(dev), b["brain"].to(dev), b["y"].to(dev)
                p = torch.sigmoid(model(x)) * br
                for t in thrs:
                    m = (p > t).float(); inter = float((m * yy).sum()); s = float(m.sum() + yy.sum())
                    rows.append(dict(thr=t, label=int(b["label"][0]), dice=2 * inter / max(s, 1), any=float(m.sum() > 0), overlap=float(inter > 0)))
        res[name] = pd.DataFrame(rows)
    P("\n=== threshold sweep (Dice on positives / positives overlapped / negatives with any voxel) ===")
    out = []
    for t in thrs:
        r = {"thr": t}
        for n in ("val", "test"):
            g = res[n][res[n].thr == t]
            r[f"{n}_dice"] = g[g.label == 1].dice.mean(); r[f"{n}_pos_overlap"] = g[g.label == 1].overlap.mean(); r[f"{n}_neg_any"] = g[g.label == 0]["any"].mean()
        out.append(r)
    out = pd.DataFrame(out); P(out.round(3).to_string(index=False))
    tbest = out.loc[out.val_dice.idxmax(), "thr"]
    P(f"threshold chosen on VAL Dice: {tbest} -> test Dice {out.loc[out.thr == tbest, 'test_dice'].iloc[0]:.3f}")
    out.to_csv(Path(a.eval) / "diag_threshold_sweep.csv", index=False)


if __name__ == "__main__":
    main()
