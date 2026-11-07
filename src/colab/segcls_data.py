import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Cases(Dataset):
    """Loads data/cases/<id>.pt from the pack. aug=True: random left-right flip (last axis) + small intensity shift inside the brain."""
    def __init__(self, pack, ids, aug=False):
        self.dir = Path(pack) / "data" / "cases"
        self.ids, self.aug = list(ids), aug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = torch.load(self.dir / f"{self.ids[i]}.pt", weights_only=False)
        x = d["x"].float()                                   # (1,96,128,128)
        brain = d["brain"].float()[None]                     # (1,96,128,128)
        y = d["lesion"].float()[None]
        if self.aug:
            if random.random() < 0.5:
                x, brain, y = x.flip(-1), brain.flip(-1), y.flip(-1)
            x = (x + brain * random.gauss(0, 0.02)).clamp(0, 1)      # +-~2.4 HU shift inside the brain; outside stays 0
        return dict(id=self.ids[i], x=x, brain=brain, y=y, label=int(d["meta"]["label"]))


def split_ids(pack, val_fold=0):
    coh = pd.read_csv(Path(pack) / "cohort.csv")
    tr = coh[(coh.split == "train") & (coh.fold != val_fold)]
    va = coh[(coh.split == "train") & (coh.fold == val_fold)]
    te = coh[coh.split == "test"]
    return tr, va, te
