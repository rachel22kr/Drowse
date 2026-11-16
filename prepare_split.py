"""데이터 -> split CSV 생성 (사람 단위 분할: 학습에 안 쓴 사람으로 평가).

사용 예 1) MRL Eye Dataset (흑백 적외선 눈 이미지, 파일명에 피험자/안경/조명 정보 포함)
  python prepare_split.py --mrl /data/mrlEyes_2018_01 --out split.csv
  파일명 형식: s0001_00001_0_0_0_0_0_01.png
     = subject_imageid_gender_glasses_eyestate_reflections_lighting_sensor
     eyestate: 0=감음, 1=뜸  (다운로드 후 README로 꼭 다시 확인)

사용 예 2) 직접 촬영한 데이터: root/<subject>/<open|closed>/*.png
  python prepare_split.py --folder /data/our_eyes --out split.csv

이 split.csv를 PCA 담당에게도 그대로 전달하세요(동일 분할 조건).
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def from_mrl(root):
    rows = []
    for p in Path(root).rglob("*.png"):
        f = p.stem.split("_")
        if len(f) < 8:
            continue
        rows.append(dict(path=str(p), subject=f[0], label=int(f[4]),
                         glasses=int(f[3]), reflections=int(f[5]), lighting=int(f[6])))
    return pd.DataFrame(rows)


def from_folder(root):
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        cls = p.parent.name.lower()
        if cls not in {"open", "closed"}:
            continue
        rows.append(dict(path=str(p), subject=p.parent.parent.name,
                         label=1 if cls == "open" else 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrl")
    ap.add_argument("--folder")
    ap.add_argument("--out", default="split.csv")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = from_mrl(a.mrl) if a.mrl else from_folder(a.folder)
    assert len(df), "이미지를 찾지 못했습니다. 경로를 확인하세요."

    subs = sorted(df.subject.unique())
    random.Random(a.seed).shuffle(subs)
    n = len(subs)
    n_test, n_val = max(1, round(n * a.test)), max(1, round(n * a.val))
    test, val = set(subs[:n_test]), set(subs[n_test:n_test + n_val])
    df["split"] = df.subject.map(lambda s: "test" if s in test else "val" if s in val else "train")
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "label"]).size().unstack())
    print(f"subjects: train={n - n_test - n_val} val={n_val} test={n_test} -> {a.out}")


if __name__ == "__main__":
    main()
