"""KD 없이 CE로 단독 학습. 전체 데이터 CPU 캐시 사용 권장.

  python make_subset.py --csv split.csv --out full.csv
  python train.py --model small --csv full.csv --out ckpt/small.pt
  python train.py --model densenet --csv full.csv --freeze head_only --out ckpt/densenet.pt

GPU 기본값: AMP 사용, small 배치 32 / DenseNet 배치 16.
--accum-steps로 배치 메모리를 늘리지 않고 여러 배치의 gradient를 누적 가능.
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import EyeDataset, build_model


def init_worker(_):
    # worker 수 × OpenCV 내부 스레드 수가 폭증하지 않도록 제한.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


@torch.no_grad()
def evaluate(model, loader, dev, amp=False):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=dev, enabled=amp):
            ps.append(model(x.to(dev, non_blocking=True)).argmax(1).cpu())
        ys.append(y)
    ys, ps = torch.cat(ys), torch.cat(ps)
    return (ys == ps).float().mean().item(), f1_score(ys, ps, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["small", "densenet"], required=True)
    ap.add_argument("--csv", default="full.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, help="실제 배치 크기 (small=32, densenet=16)")
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient 누적 배치 수")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="CUDA 혼합 정밀도 (기본 사용, --no-amp로 해제)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--resume", action="store_true", help="완료한 epoch 다음부터 이어서 학습")
    ap.add_argument("--freeze", choices=["none", "head_only", "last_block"], default="none",
                    help="densenet 전용. head_only=분류층만, last_block=마지막 블록+분류층")
    a = ap.parse_args()
    if a.bs is None:
        a.bs = 32 if a.model == "small" else 16
    if min(a.bs, a.epochs, a.accum_steps, a.cpu_threads) < 1 or a.workers < 0:
        ap.error("배치/epoch/누적/스레드는 1 이상, workers는 0 이상이어야 합니다")
    if a.model == "small" and a.freeze != "none":
        ap.error("freeze는 densenet에서만 사용합니다")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    amp = a.amp and dev == "cuda"
    cv2.setNumThreads(1)
    torch.set_num_threads(a.cpu_threads)
    print(f"device: {dev}, AMP: {amp}, batch: {a.bs}, accumulation: {a.accum_steps}")
    loader_opts = dict(num_workers=a.workers, pin_memory=dev == "cuda")
    if a.workers:
        loader_opts.update(prefetch_factor=1, worker_init_fn=init_worker)
    tr_ds = EyeDataset(a.csv, "train", augment=True)
    va_ds = EyeDataset(a.csv, "val")
    if not len(tr_ds) or not len(va_ds):
        ap.error("train과 val 데이터가 모두 필요합니다")
    # 마지막 잔여 배치까지 사용. 검증도 학습과 같은 배치 크기로 메모리 제한.
    tr = DataLoader(tr_ds, a.bs, shuffle=True, drop_last=False, **loader_opts)
    va = DataLoader(va_ds, a.bs, **loader_opts)
    print(f"samples: train={len(tr_ds):,}, val={len(va_ds):,}")

    out_path = Path(a.out)
    last = out_path.with_name(out_path.stem + "_last" + out_path.suffix)
    if a.resume and not last.exists():
        ap.error(f"이어하기 체크포인트가 없습니다: {last}")
    model = build_model(a.model, pretrained=not a.resume).to(dev)
    lr = a.lr if a.model == "small" else a.lr * 0.1
    if a.model == "densenet" and a.freeze != "none":
        train_keys = ["classifier"] + (["denseblock4", "norm5"] if a.freeze == "last_block" else [])
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in train_keys)
        lr = a.lr
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, a.epochs * math.ceil(len(tr) / a.accum_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best, start = -1, 0
    # KD/구형 스케줄을 새 학습의 resume으로 잘못 이어붙이지 않음.
    config = dict(objective="ce", model=a.model, csv=str(Path(a.csv).resolve()),
                  samples=len(tr_ds), bs=a.bs, accum_steps=a.accum_steps,
                  epochs=a.epochs, freeze=a.freeze, lr=a.lr, amp=amp)
    if a.resume:
        st = torch.load(last, map_location="cpu")
        if st.get("config") != config:
            ap.error("이어하기 설정이 다르거나 구형 체크포인트입니다. 새 --out으로 학습하세요")
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best = st["epoch"], st["best"]
        del st
        print(f"resume from epoch {start} (best F1 {best:.4f})")
    for ep in range(start, a.epochs):
        model.train()
        if a.model == "densenet" and a.freeze != "none":
            # 고정한 블록의 BatchNorm 통계도 고정.
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d) and not module.weight.requires_grad:
                    module.eval()
        t0, tot = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # 잔여 배치/마지막 누적 그룹도 실제 샘플 수로 가중.
            group_start = (step // a.accum_steps) * a.accum_steps * a.bs
            group_samples = min(a.accum_steps * a.bs, len(tr_ds) - group_start)
            with torch.autocast(device_type=dev, enabled=amp):
                logits = model(x)
                loss_sum = F.cross_entropy(logits, y, label_smoothing=0.05, reduction="sum")
                loss = loss_sum / group_samples
            scaler.scale(loss).backward()
            if (step + 1) % a.accum_steps == 0 or step + 1 == len(tr):
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss_sum.detach().item()
        acc, f1 = evaluate(model, va, dev, amp)
        print(f"ep {ep + 1:3d} loss {tot / len(tr_ds):.4f} val_acc {acc:.4f} "
              f"val_macroF1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best = f1
            torch.save({"model": a.model, "state_dict": model.state_dict()}, out_path)
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "config": config, "epoch": ep + 1, "best": best}, last)
    print(f"best val macro-F1 {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
