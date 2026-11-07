"""Train the decoder (encoder frozen) on folds != VAL_FOLD of the training set.  Test cases are never loaded here.
Resumable: every epoch writes last.pt to --out (put --out on Google Drive).  Re-running the same command resumes.

  python segcls_train.py --pack /content/segcls_pack --out /content/drive/MyDrive/segcls_run1
  python segcls_train.py --pack ... --out ... --smoke        # 2 epochs on a handful of cases, plus gradient/resume checks
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from segcls_model import SegCls, case_scores, seg_loss
from segcls_data import Cases, split_ids


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def validate(model, loader, dev, amp):
    model.eval()
    rows, dices, fp = [], [], []
    for b in loader:
        x, br, y = b["x"].to(dev), b["brain"].to(dev), b["y"].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            z = model(x)
        sc = case_scores(z, br)
        p = (torch.sigmoid(z) > 0.5).float() * br
        for i in range(x.shape[0]):
            r = dict(id=b["id"][i], label=int(b["label"][i]), **{k: float(v[i]) for k, v in sc.items()})
            inter = float((p[i] * y[i]).sum()); s = float(p[i].sum() + y[i].sum())
            if r["label"] == 1:
                dices.append(2 * inter / max(s, 1))
            else:
                fp.append(float((torch.sigmoid(z[i]) * br[i]).sum()))
            rows.append(r)
    lab = np.array([r["label"] for r in rows])
    aucs = {k: float(roc_auc_score(lab, [r[k] for r in rows])) for k in rows[0] if k not in ("id", "label")}
    return dict(dice_pos=float(np.mean(dices)), fp_mass_neg=float(np.mean(fp)), **{"auc_" + k: v for k, v in aucs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=2, help="pairs: bs/2 positives + bs/2 negatives per batch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="testing only: random encoder")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (dev.type == "cuda") and not a.no_amp
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    tr, va, _ = split_ids(a.pack, a.val_fold)
    if a.smoke:
        tr = tr.groupby("label").head(4); va = va.groupby("label").head(3); a.epochs = 2
    pos, neg = list(tr[tr.label == 1].id), list(tr[tr.label == 0].id)
    print(f"device {dev} amp {amp} | train {len(tr)} (pos {len(pos)}, neg {len(neg)}) | val fold {a.val_fold}: {len(va)} (pos {int((va.label==1).sum())})")
    json.dump(dict(vars(a), n_pos=len(pos), n_neg=len(neg), val_ids=list(va.id)), open(out / "config.json", "w"), indent=1)

    wdir = None if a.no_pretrained else Path(a.pack) / "weights" / "ct_fm_feature_extractor"
    model = SegCls(wdir).to(dev)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = len(pos) * 2 // a.bs
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    val_loader = DataLoader(Cases(a.pack, list(va.id)), batch_size=2, num_workers=a.workers)

    start, best = 0, dict(auc=-1, dice=-1)
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(ck["decoder"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}")

    logf = out / "train_log.csv"
    for ep in range(start, a.epochs):
        t0 = time.time(); model.train()
        rng = random.Random(a.seed * 1000 + ep)
        negs = rng.sample(neg, min(len(pos), len(neg)))
        ids = [i for pair in zip(pos, negs) for i in pair]            # positive, negative, positive, ...
        order = list(range(0, len(ids), 2)); rng.shuffle(order)
        ids = [j for o in order for j in ids[o:o + 2]]
        loader = DataLoader(Cases(a.pack, ids, aug=True), batch_size=a.bs, shuffle=False, num_workers=a.workers, drop_last=True)
        tl = tb = td = 0.0; n = 0
        for b in loader:
            x, br, y = b["x"].to(dev), b["brain"].to(dev), b["y"].to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                z = model(x)
            loss, bce, dice = seg_loss(z, y, br)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); tb += bce.item(); td += dice.item(); n += 1
            if a.smoke and n == 1:
                print(f"  step time first batch {time.time()-t0:.2f}s, loss {loss.item():.3f}")
        v = validate(model, val_loader, dev, amp)
        sec = time.time() - t0
        row = dict(epoch=ep, loss=tl / n, bce=tb / n, dice_loss=td / n, lr=sched.get_last_lr()[0], sec=round(sec, 1), **v)
        new = not logf.exists()
        with open(logf, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)
        print(f"ep {ep:02d} loss {row['loss']:.3f} | val Dice(pos) {v['dice_pos']:.3f} FPmass(neg) {v['fp_mass_neg']:.1f} "
              f"AUC lse1 {v['auc_lse1']:.3f} top50 {v['auc_top50']:.3f} softvol {v['auc_softvol']:.3f} | {sec:.0f}s", flush=True)
        if v["auc_lse1"] > best["auc"]:
            best["auc"] = v["auc_lse1"]; torch.save({"decoder": model.decoder.state_dict(), "epoch": ep, "val": v}, out / "best_auc.pt")
        if v["dice_pos"] > best["dice"]:
            best["dice"] = v["dice_pos"]; torch.save({"decoder": model.decoder.state_dict(), "epoch": ep, "val": v}, out / "best_dice.pt")
        torch.save({"decoder": model.decoder.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    if a.smoke:      # gradient check: does the case score backpropagate to the INPUT (needed for IG / Grad-CAM later)?
        model.eval(); b = next(iter(val_loader))
        x = b["x"].to(dev).requires_grad_(True)
        z = model(x)
        for k, v in case_scores(z, b["brain"].to(dev)).items():
            g, = torch.autograd.grad(v.sum(), x, retain_graph=True)
            print(f"  d{k}/dx  |grad|_1 = {float(g.abs().sum()):.3e}")
    print("done")


if __name__ == "__main__":
    main()
