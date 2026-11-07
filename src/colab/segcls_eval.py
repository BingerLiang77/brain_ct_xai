"""Evaluate a checkpoint: all case scores on val fold and test, Dice, FP mass; pick the pooling on VAL only, then report test AUC for it.
  python segcls_eval.py --pack /content/segcls_pack --ckpt /content/drive/MyDrive/segcls_run1/best_auc.pt --out /content/drive/MyDrive/segcls_run1/eval_best_auc
"""
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from segcls_model import SegCls, case_scores
from segcls_data import Cases, split_ids


@torch.no_grad()
def run(model, pack, ids, dev, save_prob_dir=None):
    model.eval(); rows = []
    for b in DataLoader(Cases(pack, ids), batch_size=1, num_workers=2):
        x, br, y = b["x"].to(dev), b["brain"].to(dev), b["y"].to(dev)
        z = model(x)                                              # fp32 for evaluation
        sc = case_scores(z, br); p = (torch.sigmoid(z) > 0.5).float() * br
        inter = float((p * y).sum()); s = float(p.sum() + y.sum())
        rows.append(dict(id=b["id"][0], label=int(b["label"][0]), **{k: float(v[0]) for k, v in sc.items()},
                         dice=2 * inter / max(s, 1), pred_vox=float(p.sum()), lesion_vox=float(y.sum()), fp_mass=float((torch.sigmoid(z) * br).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = SegCls(None if a.no_pretrained else Path(a.pack) / "weights" / "ct_fm_feature_extractor").to(dev)
    model.decoder.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["decoder"])
    _, va, te = split_ids(a.pack, a.val_fold)
    dv, dt = run(model, a.pack, list(va.id), dev), run(model, a.pack, list(te.id), dev)
    dv.to_csv(out / "val_cases.csv", index=False); dt.to_csv(out / "test_cases.csv", index=False)
    names = [c for c in dv.columns if c not in ("id", "label", "dice", "pred_vox", "lesion_vox", "fp_mass")]
    tab = pd.DataFrame({k: dict(val_auc=roc_auc_score(dv.label, dv[k]), test_auc=roc_auc_score(dt.label, dt[k]),
                                test_auprc=average_precision_score(dt.label, dt[k])) for k in names}).T
    best = tab.val_auc.idxmax()
    lines = [tab.round(3).to_string(), "",
             f"pooling chosen on VAL only: {best} (val AUC {tab.loc[best,'val_auc']:.3f}) -> TEST AUC {tab.loc[best,'test_auc']:.3f}, AUPRC {tab.loc[best,'test_auprc']:.3f}   [old masked-pool head: 0.849 / 0.576]",
             f"Dice on positives: val {dv[dv.label==1].dice.mean():.3f}, test {dt[dt.label==1].dice.mean():.3f}; "
             f"fraction of test positives with any predicted voxel {(dt[dt.label==1].pred_vox>0).mean():.2f}; "
             f"test negatives with any predicted voxel {(dt[dt.label==0].pred_vox>0).mean():.2f}"]
    open(out / "summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))
    json.dump(dict(pooling=best), open(out / "chosen_pooling.json", "w"))


if __name__ == "__main__":
    main()
