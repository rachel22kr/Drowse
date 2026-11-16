"""테스트셋 평가: Accuracy, macro-F1, 조건별(안경/조명 등) 성능, 파라미터 수, 모델 크기, latency.
  python evaluate.py --csv full.csv --ckpt ckpt/densenet.pt ckpt/small.pt
결과는 results.csv 로 저장 -> 최아영(평가·발표)에게 전달.
* latency는 '같은 장치'에서 재야 비교가 됩니다. --device cpu 로 CPU(온디바이스 가정) 측정 권장.
"""
import argparse
import os
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from common import IMG_SIZE, EyeDataset, build_model


def load(ckpt, dev):
    c = torch.load(ckpt, map_location=dev)
    m = build_model(c["model"], pretrained=False)
    m.load_state_dict(c["state_dict"])
    return m.to(dev).eval()


@torch.no_grad()
def latency_ms(model, dev, n=200):
    x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=dev)
    for _ in range(20):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="full.csv")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--device", default="cpu", help="latency 측정 장치 (온디바이스 가정: cpu)")
    ap.add_argument("--pred_device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="test 예측 장치. GPU가 빠르고 RAM을 덜 씀")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    ds = EyeDataset(a.csv, "test")
    df = ds.df.copy()
    loader = DataLoader(ds, a.bs, num_workers=0)
    rows = []
    for ck in a.ckpt:
        m = load(ck, a.pred_device)
        with torch.no_grad():
            preds = torch.cat([m(x.to(a.pred_device)).argmax(1).cpu() for x, _ in loader]).numpy()
        # latency는 저사양 장치 가정: CPU 1스레드로 따로 측정
        m = m.to(a.device)
        if a.device == "cpu":
            torch.set_num_threads(1)
        lat = latency_ms(m, a.device)
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        if a.pred_device == "cuda":
            torch.cuda.empty_cache()
        df["pred"] = preds
        r = dict(model=os.path.basename(ck),
                 params=sum(p.numel() for p in m.parameters()),
                 size_MB=os.path.getsize(ck) / 1e6,
                 latency_ms=lat,
                 acc=accuracy_score(df.label, df.pred),
                 macroF1=f1_score(df.label, df.pred, average="macro"),
                 # 감은 눈(label 0) 기준: 놓치지 않는 비율(recall)이 졸음 감지에서 가장 중요
                 closed_recall=recall_score(df.label, df.pred, pos_label=0),
                 closed_precision=precision_score(df.label, df.pred, pos_label=0, zero_division=0))
        for cond in [c for c in ["glasses", "lighting", "reflections"] if c in df]:
            for v, g in df.groupby(cond):
                r[f"F1_{cond}={v}"] = f1_score(g.label, g.pred, average="macro")
        rows.append(r)
        print(r)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
