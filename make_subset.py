"""CPU 전처리 캐시 생성. 기본값은 전체 데이터 사용, 기존 피험자 분할 유지.

  python make_subset.py --csv split.csv --out full.csv
  python train.py --model small --csv full.csv --out ckpt/small.pt

흑백 -> 64x64 -> 히스토그램 평활화를 CPU 1스레드에서 한 번 수행.
uint8 .npy를 디스크 mmap에 한 장씩 기록하므로 전체 이미지를 RAM/GPU에 올리지 않음.
--ratio N을 명시하면 종전처럼 축소 가능. train/val/test 분할은 섞지 않음.
"""
import argparse
import re
import os
import tempfile
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

from common import IMG_SIZE, load_gray


def img_no(path):
    m = re.search(r"_(\d+)_", str(path).rsplit("/", 1)[-1])
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split.csv")
    ap.add_argument("--ratio", type=int, default=1, help="N장 중 1장 사용 (4 -> 1/4)")
    ap.add_argument("--out", default="full.csv")
    ap.add_argument("--reduce_test", action="store_true", help="test도 같이 줄이기")
    ap.add_argument("--closed_ratio", type=int, default=None,
                    help="감은 눈(label 0)만 따로 N장 중 1장. 1이면 감은 눈은 전부 유지")
    a = ap.parse_args()

    if a.ratio < 1 or (a.closed_ratio is not None and a.closed_ratio < 1):
        ap.error("ratio와 closed_ratio는 1 이상이어야 합니다")
    out = Path(a.out)
    if out.suffix.lower() != ".csv":
        ap.error("out은 .csv 파일이어야 합니다")
    if out.resolve() == Path(a.csv).resolve():
        ap.error("원본 split CSV와 다른 출력 경로를 사용하세요")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    df = pd.read_csv(a.csv)
    if df.empty:
        ap.error("입력 CSV가 비어 있습니다")
    df["img_no"] = df["path"].map(img_no)
    keys = [c for c in ["split", "subject", "label", "glasses", "lighting"] if c in df]

    parts = []
    for _, g in df.groupby(keys, dropna=False):
        g = g.sort_values("img_no")
        if g["split"].iloc[0] == "test" and not a.reduce_test:
            parts.append(g)
        else:
            r = a.closed_ratio if (a.closed_ratio and g["label"].iloc[0] == 0) else a.ratio
            parts.append(g.iloc[::r])               # 그룹이 작아도 최소 1장은 남음
    sub = pd.concat(parts).sort_values(["split", "subject", "img_no"]).reset_index(drop=True)

    # 실패한 전처리가 완성된 캐시로 오인되지 않도록 임시 파일에서 생성.
    npy = out.with_suffix(".npy")
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=out.parent) as tmp:
        tmp_npy, tmp_csv = Path(tmp) / "cache.npy", Path(tmp) / "cache.csv"
        X = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=np.uint8,
                                      shape=(len(sub), IMG_SIZE, IMG_SIZE))
        for i, path in enumerate(tqdm(sub["path"], desc="preprocess (CPU)")):
            X[i] = load_gray(path)
        nbytes = X.nbytes
        X.flush()
        del X
        sub["idx"] = np.arange(len(sub))
        sub.drop(columns="img_no").to_csv(tmp_csv, index=False)
        os.replace(tmp_npy, npy)
        os.replace(tmp_csv, out)

    print("\n원본 -> 캐시")
    for s in ["train", "val", "test"]:
        o, n = (df.split == s).sum(), (sub.split == s).sum()
        r = sub[sub.split == s].label.mean() if n else 0
        c = ((sub.split == s) & (sub.label == 0)).sum()
        print(f"  {s:5s}: {o:6d} -> {n:6d} (감은 눈 {c}장, {1 - r:.0%})")
    print(f"저장: {a.out}, {npy} ({nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
