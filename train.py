"""데이터 전처리 → 신경망 학습 → PCA 계산과 비교. 이 파일에서 전체 학습을 실행한다."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from train_preprocess import ARTIFACTS_DIR, EyeDataset, LABEL_NAMES, PREPROCESSING, PipelinePaths
from inf import NNPredictor, PCAPredictor, build_model
from inf_alarm import DrowsinessConfig
from train_pca import run as train_pca
from train_preprocess import run as preprocess


@torch.inference_mode()
def predict(model, loader, device, amp=False):
    model.eval()
    labels, predictions, scores = [], [], []
    for images, targets in loader:
        with torch.autocast(device_type=device, enabled=amp):
            logits = model(images.to(device))
        probabilities = logits.float().softmax(1).cpu().numpy()
        labels.append(targets.numpy())
        predictions.append(probabilities.argmax(1))
        scores.append(probabilities[:, 0])
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(scores)


def summarize(labels, predictions):
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "closed_recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "closed_precision": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def plot_accuracy(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history["epoch"], history["test_accuracy"], marker="o")
    axis.set(xlabel="Epoch", ylabel="Test accuracy", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_network(csv, output_dir, *, model="small", epochs=30, bs=32, lr=1e-3,
        device="auto", seed=42, workers=0, pretrained=False):
    assert epochs > 0 and bs > 0, "epochs와 bs는 1 이상이어야 합니다"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    amp = device == "cuda"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    paths = PipelinePaths(output_dir)
    paths.create()
    train = EyeDataset(csv, "train", augment=True)
    test = EyeDataset(csv, "test")
    assert len(train) and len(test), "train과 test 데이터가 모두 필요합니다"
    loader_options = dict(batch_size=bs, num_workers=workers, pin_memory=amp)
    train_loader = DataLoader(train, shuffle=True, **loader_options)
    test_loader = DataLoader(test, shuffle=False, **loader_options)
    network = build_model(model, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr if model == "small" else lr * 0.1,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                loss = F.cross_entropy(network(images), labels, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
        scheduler.step()
        labels, predictions, scores = predict(network, test_loader, device, amp)
        metrics = summarize(labels, predictions)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(train),
                            test_accuracy=metrics["accuracy"], test_macro_f1=metrics["macro_f1"]))
        print(f"NN {epoch}/{epochs}: loss={history[-1]['train_loss']:.4f}, "
              f"test accuracy={metrics['accuracy']:.4f}", flush=True)
    # 테스트 정확도로 epoch를 선택하지 않고 지정한 마지막 epoch를 사용한다.
    torch.save(dict(model=model, state_dict=network.state_dict(), preprocessing=PREPROCESSING,
                    label_names=LABEL_NAMES), paths.models / "eye_model.pt")
    table = test.df[["path", "label"]].copy()
    table["pred"], table["p_closed"] = predictions, scores
    table.to_csv(paths.reports / "nn_predictions.csv", index=False)
    metrics.update(model=model, epoch=epochs)
    (paths.reports / "nn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history = pd.DataFrame(history)
    history.to_csv(paths.reports / "history.csv", index=False)
    plot_accuracy(history, paths.reports / "test_accuracy.png")
    return metrics


def save_comparison(paths, nn_report, pca_report):
    keys = ('samples', 'accuracy', 'macro_f1', 'closed_recall', 'closed_precision')
    comparison = pd.DataFrame([dict(method=method, **{key: report[key] for key in keys})
                               for method, report in (('neural_network', nn_report), ('pca', pca_report))])
    comparison.to_csv(paths.reports / 'comparison.csv', index=False)
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(['Neural network', 'PCA + mean direction'], comparison['accuracy'])
    axis.bar_label(bars, labels=[f'{value:.1%}' for value in comparison['accuracy']], padding=4)
    axis.set(ylabel='Accuracy', ylim=(0, 1.12), title='Same held-out test images')
    figure.tight_layout()
    figure.savefig(paths.reports / 'accuracy_comparison.png', dpi=160)
    plt.close(figure)
    print(comparison.to_string(index=False))


def evaluate_saved(output_dir=ARTIFACTS_DIR, device='auto'):
    """이미 저장한 신경망과 PCA를 같은 테스트셋에서 재평가한다."""
    paths = PipelinePaths(output_dir)
    torch.set_num_threads(1)
    dataset = EyeDataset(paths.data / 'split.csv', 'test')
    predictors = {'neural_network': NNPredictor(paths.models / 'eye_model.pt', device),
                  'pca': PCAPredictor(paths.models / 'pca_model.npz')}
    rows = []
    for name, predictor in predictors.items():
        predictions = []
        for start in range(0, len(dataset), 32):
            eyes = np.stack([dataset.image(i) for i in range(start, min(start + 32, len(dataset)))])
            labels, _ = predictor.predict(eyes)
            predictions.extend(labels)
        metrics = summarize(dataset.df.label.to_numpy(), predictions)
        metrics.pop('confusion_matrix')
        rows.append(dict(method=name, **metrics))
    results = pd.DataFrame(rows)
    results.to_csv(paths.reports / 'evaluation.csv', index=False)
    print(results.to_string(index=False))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--folder', type=Path)
    source.add_argument('--csv', type=Path)
    source.add_argument('--mrl', type=Path)
    parser.add_argument('--artifacts-dir', type=Path, default=ARTIFACTS_DIR)
    parser.add_argument('--split-by', choices=['sample', 'subject'], default='sample')
    parser.add_argument('--model', choices=['small', 'densenet'], default='small')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pca-components', type=int, default=64)
    parser.add_argument('--evaluate-only', action='store_true', help='저장 모델을 재학습 없이 테스트셋으로 평가')
    args = parser.parse_args(argv)
    paths = PipelinePaths(args.artifacts_dir).create()
    if args.evaluate_only:
        evaluate_saved(paths.root, args.device)
        return
    csv = preprocess(args.csv, paths.root, folder=args.folder, mrl=args.mrl,
                     seed=args.seed, split_by=args.split_by)
    nn_report = train_network(csv, paths.root, model=args.model, epochs=args.epochs, bs=args.bs,
                         lr=args.lr, device=args.device, seed=args.seed, pretrained=args.pretrained)
    pca_report = train_pca(csv, paths.root, args.pca_components, args.seed)
    save_comparison(paths, nn_report, pca_report)
    DrowsinessConfig().save(paths.models / 'alarm_config.json')
    print(f'학습 완료 → {paths.root}')


if __name__ == '__main__':
    main()
