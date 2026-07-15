"""
MR-RATE real data loader — optimized: pre-cached file paths, NAS-aware prefetch.

Magic numbers (empirically tuned for PRO 6000 + NAS):
  num_workers = 8   — sweet spot: keeps 16-core CPU pipes full without GIL contention
  prefetch_factor = 4   — 4× pipeline depth so NAS seek latency doesn't stall GPU
  pin_memory = True   — zero-copy CPU→GPU
  persistent_workers = True   — avoids re-fork overhead every epoch
"""
import numpy as np
import torch
import random
import pandas as pd
import nibabel as nib
from pathlib import Path
import scipy.ndimage


def _percentile_clip(vol, lower=1, upper=99):
    lo, hi = np.percentile(vol[vol > 0] if (vol > 0).any() else vol, [lower, upper])
    return np.clip(vol, lo, hi)


def _zscore_norm(vol):
    vol = _percentile_clip(vol, 1, 99)
    mu, std = vol.mean(), vol.std()
    return ((vol - mu) / max(std, 1e-8)).astype(np.float32)


class MRRateDataset(torch.utils.data.Dataset):
    """Loads MR-RATE dataset with pre-cached file paths for zero-glob performance.

    Data structure:
      {root}/splits.csv           -> batch_id, patient_uid, study_uid, split
      {root}/mri/{batch}/{study}/img/{study}_{modality}.nii.gz
      {root}/reports/{batch}_reports.csv -> study_uid, report, findings, ...
    """

    MODALITY_PATTERNS = {
        "t1": "t1w-raw-sag",
        "flair": "flair-raw-sag",
        "t2": "t2w-raw-axi",
    }

    def __init__(self, root, split="train", target_size=(128, 128, 128),
                 normalize="zscore", augment=False, batch_filter=None):
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment and split == "train"
        self.batch_filter = batch_filter

        splits_df = pd.read_csv(self.root / "splits.csv")
        self.samples = splits_df[splits_df["split"] == split].reset_index(drop=True)
        if self.batch_filter:
            batch_ids = [b.strip() for b in self.batch_filter.split(",")]
            self.samples = self.samples[self.samples["batch_id"].isin(batch_ids)].reset_index(drop=True)

        self.reports = {}
        batch_ids = self.samples["batch_id"].unique()
        for batch_id in batch_ids:
            rp = self.root / "reports" / f"{batch_id}_reports.csv"
            if rp.exists():
                rdf = pd.read_csv(rp)
                for _, row in rdf.iterrows():
                    self.reports[str(row["study_uid"])] = str(row.get("report", ""))

        N = len(self.samples)
        self._file_map = [None] * N
        n_missing = 0
        for i in range(N):
            row = self.samples.iloc[i]
            img_dir = self.root / "mri" / str(row["batch_id"]) / str(row["study_uid"]) / "img"
            paths = {}
            for mod_name, pattern in self.MODALITY_PATTERNS.items():
                candidates = sorted(img_dir.glob(f"{row['study_uid']}_{pattern}*.nii.gz")) if img_dir.exists() else []
                if not candidates:
                    paths[mod_name] = None
                    n_missing += 1
                    continue
                chosen = None
                exact = f"{row['study_uid']}_{pattern}.nii.gz"
                for c in candidates:
                    if c.name == exact:
                        chosen = c
                        break
                paths[mod_name] = chosen if chosen is not None else candidates[0]
            self._file_map[i] = paths

        print(f"[{split}] {N} samples, {len(self.reports)} with reports, "
              f"{n_missing} modality files missing (using zeros), paths pre-cached")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        study_uid = str(row["study_uid"])
        pre_cached = self._file_map[idx]

        volumes = {}
        has_mask = {}
        for mod_name in self.MODALITY_PATTERNS:
            chosen = pre_cached[mod_name]
            if chosen is None:
                volumes[mod_name] = torch.zeros(1, *self.target_size)
                has_mask[mod_name] = False
                continue

            try:
                vol = nib.load(str(chosen)).get_fdata().astype(np.float32)
            except (EOFError, OSError):
                alt_idx = random.randint(0, len(self.samples) - 1)
                return self.__getitem__(alt_idx)

            vol = self._resize(vol, self.target_size)

            if self.augment:
                if random.random() < 0.5:
                    vol = np.flip(vol, axis=random.randint(0, 2))
                vol = vol + np.random.normal(0, 0.02, vol.shape).astype(np.float32)

            if self.normalize == "zscore":
                vol = _zscore_norm(vol)
            else:
                vmin, vmax = vol.min(), vol.max()
                if vmax > vmin:
                    vol = (vol - vmin) / (vmax - vmin)

            volumes[mod_name] = torch.from_numpy(np.expand_dims(vol, 0))
            has_mask[mod_name] = True

        report = self.reports.get(study_uid, "")

        return {
            "t1": volumes["t1"],
            "flair": volumes["flair"],
            "t2": volumes["t2"],
            "has_t1": torch.tensor(has_mask["t1"]),
            "has_flair": torch.tensor(has_mask["flair"]),
            "has_t2": torch.tensor(has_mask["t2"]),
            "report": report,
            "study_uid": study_uid,
            "patient_uid": str(row.get("patient_uid", study_uid)),
        }

    @staticmethod
    def _resize(vol, target):
        cur = np.array(vol.shape, dtype=np.float64)
        tgt = np.array(target, dtype=np.float64)
        return scipy.ndimage.zoom(vol, tgt / cur, order=1)
