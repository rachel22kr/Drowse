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
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from train_preprocess import RESULTS_DIR, EyeDataset, LABEL_NAMES, PREPROCESSING, PipelinePaths
from inf import NNPredictor, PCAPredictor, build_model
from inf_alarm import DrowsinessConfig
from train_pca import run as train_pca
from train_preprocess import run as preprocess


@torch.inference_mode()
def predict(model, loader, device, amp=False, return_loss=False):
    model.eval()
    labels, predictions, scores, loss_sum = [], [], [], 0.0
    for images, targets in loader:
        with torch.autocast(device_type=device, enabled=amp):
            logits = model(images.to(device))
        logits = logits.float()
        if return_loss:
            loss_sum += F.cross_entropy(logits, targets.to(device), reduction="sum").item()
        probabilities = logits.softmax(1).cpu().numpy()
        labels.append(targets.numpy())
        predictions.append(probabilities.argmax(1))
        scores.append(probabilities[:, 0])
    result = np.concatenate(labels), np.concatenate(predictions), np.concatenate(scores)
    return (*result, loss_sum / len(result[0])) if return_loss else result


def summarize(labels, predictions, closed_scores=None):
    report = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "closed_recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "closed_precision": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if closed_scores is not None:
        targets = np.asarray(labels) == 0
        both_classes = np.unique(targets).size == 2
        report['roc_auc'] = float(roc_auc_score(targets, closed_scores)) if both_classes else None
        report['average_precision'] = float(average_precision_score(targets, closed_scores)) if both_classes else None
    return report


def plot_accuracy(history, path, best_epoch):
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history["epoch"], history["test_accuracy"], marker="o", label="Test accuracy (monitor only)")
    axis.axvline(best_epoch, color="tab:red", linestyle="--", label="Best validation epoch")
    axis.set(xlabel="Epoch", ylabel="Test accuracy", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for column, label in (("train_loss", "Train"), ("val_loss", "Validation")):
        axes[0].plot(history["epoch"], history[column], label=label)
    for column, label in (("train_accuracy_augmented", "Train (augmented)"),
                          ("val_accuracy", "Validation"), ("test_accuracy", "Test (monitor only)")):
        axes[1].plot(history["epoch"], history[column], label=label)
    for axis, ylabel in zip(axes, ("Loss", "Accuracy")):
        axis.axvline(best_epoch, color="tab:red", linestyle="--", label="Best validation epoch")
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_ylim(0, 1.02)
    figure.tight_layout()
    figure.savefig(path.parent / "training_curves.png", dpi=160)
    plt.close(figure)


def train_network(csv, output_dir, *, model="small", epochs=300, bs=32, lr=1e-3,
        device="auto", seed=42, workers=0, patience=10, min_delta=1e-4, val_ratio=0.1):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    if min(epochs, bs, patience) < 1 or min_delta < 0 or not 0 < val_ratio < 1:
        raise ValueError("epochs/bs/patience는 양수, min-delta는 0 이상, val-ratio는 0~1이어야 합니다")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    amp = device == "cuda"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    paths = PipelinePaths(output_dir).create()
    outer_train = EyeDataset(csv, "train", augment=True)
    validation_data = EyeDataset(csv, "train")
    test = EyeDataset(csv, "test")
    fit_indices, val_indices = train_test_split(np.arange(len(outer_train)), test_size=val_ratio,
                                               stratify=outer_train.df.label, random_state=seed)
    split_table = outer_train.df.copy()
    split_table["nn_split"] = "train"
    split_table.loc[val_indices, "nn_split"] = "validation"
    split_table.to_csv(paths.reports / "nn_validation_split.csv", index=False)
    loader_options = dict(batch_size=bs, num_workers=workers, pin_memory=amp)
    train_loader = DataLoader(Subset(outer_train, fit_indices), shuffle=True, **loader_options)
    val_loader = DataLoader(Subset(validation_data, val_indices), shuffle=False, **loader_options)
    test_loader = DataLoader(test, shuffle=False, **loader_options)
    print(f"학습 장치: {device}, train={len(fit_indices):,}, val={len(val_indices):,}, "
          f"test={len(test):,}, 저장: {paths.models}", flush=True)
    network = build_model(model).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr if model == "small" else lr * 0.1,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint = paths.models / "best_model.pt"
    history, best_loss, improvement_anchor, best_epoch, wait = [], float("inf"), float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                logits = network(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum().item())
        scheduler.step()
        # 검증/테스트는 저장 모델의 추론과 같은 FP32로 계산한다.
        val_labels, val_predictions, _, val_loss = predict(network, val_loader, device, return_loss=True)
        val_metrics = summarize(val_labels, val_predictions)
        if not np.isfinite(val_loss):
            raise ValueError("검증 손실이 유한하지 않습니다. 학습률과 입력 데이터를 확인하세요")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model, state_dict=network.state_dict(), preprocessing=PREPROCESSING,
                            label_names=LABEL_NAMES, epoch=epoch, val_loss=val_loss,
                            selection="validation_loss"), temporary)
            temporary.replace(checkpoint)
        # 작은 개선도 베스트 모델에는 반영하되, min_delta 이상 누적 개선될 때 대기를 초기화한다.
        if val_loss < improvement_anchor - min_delta:
            improvement_anchor, wait = val_loss, 0
        else:
            wait += 1
        labels, predictions, _ = predict(network, test_loader, device)
        test_metrics = summarize(labels, predictions)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(fit_indices),
                            train_accuracy_augmented=correct / len(fit_indices),
                            val_loss=val_loss, val_accuracy=val_metrics["accuracy"],
                            val_macro_f1=val_metrics["macro_f1"], test_accuracy=test_metrics["accuracy"],
                            test_macro_f1=test_metrics["macro_f1"]))
        pd.DataFrame(history).to_csv(paths.reports / "history.csv", index=False)
        print(f"NN {epoch}/{epochs}: val_loss={val_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
              f"test_acc={test_metrics['accuracy']:.4f}, patience={wait}/{patience}", flush=True)
        if wait >= patience:
            print(f"조기 종료: epoch {epoch}, best epoch {best_epoch}", flush=True)
            break
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    labels, predictions, scores = predict(network, test_loader, device)
    metrics = summarize(labels, predictions, scores)
    metrics.update(model=model, best_epoch=best_epoch, epochs_run=len(history), best_val_loss=best_loss,
                   selection="validation_loss", train_samples=len(fit_indices), validation_samples=len(val_indices))
    table = test.df[["path", "label"]].copy()
    table["pred"], table["p_closed"] = predictions, scores
    table.to_csv(paths.reports / "nn_predictions.csv", index=False)
    (paths.reports / "nn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = dict(best_epoch=best_epoch, best_val_loss=best_loss, epochs_run=len(history),
                   max_epochs=epochs, stopped_early=len(history) < epochs, monitor="validation_loss",
                   patience=patience, min_delta=min_delta, val_ratio=val_ratio,
                   train_samples=len(fit_indices), validation_samples=len(val_indices), test_samples=len(test))
    (paths.reports / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_accuracy(pd.DataFrame(history), paths.reports / "test_accuracy.png", best_epoch)
    return metrics


def save_comparison(paths, nn_report, pca_report):
    """동일 테스트셋에서 신경망과 PCA의 분류·순위 지표를 비교한다."""
    tables = [pd.read_csv(paths.reports / name) for name in ('nn_predictions.csv', 'pca_predictions.csv')]
    if not tables[0][['path', 'label']].equals(tables[1][['path', 'label']]):
        raise ValueError('두 모델의 테스트 이미지와 레이블이 다릅니다')
    reports = [nn_report, pca_report]
    names = ['Neural network', 'PCA']
    methods = ['neural_network', 'pca']
    score_columns = ['p_closed', 'closed_score']
    keys = ('samples', 'accuracy', 'macro_f1', 'closed_precision', 'closed_recall', 'roc_auc', 'average_precision')
    for table, report, column, filename in zip(tables, reports, score_columns, ('nn_metrics.json', 'pca_metrics.json')):
        report.update(summarize(table.label.to_numpy(), table.pred.to_numpy(), table[column].to_numpy()))
        (paths.reports / filename).write_text(json.dumps(report, indent=2), encoding='utf-8')
    comparison = pd.DataFrame([dict(method=method, **{key: report[key] for key in keys})
                               for method, report in zip(methods, reports)])
    comparison.to_csv(paths.reports / 'comparison.csv', index=False)

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(names, comparison['accuracy'])
    axis.bar_label(bars, labels=[f'{value:.1%}' for value in comparison['accuracy']], padding=4)
    axis.set(ylabel='Accuracy', ylim=(0, 1.12), title='Same held-out test images')
    figure.tight_layout()
    figure.savefig(paths.reports / 'accuracy_comparison.png', dpi=160)
    plt.close(figure)

    columns = keys[1:]
    x = np.arange(len(columns))
    figure, axis = plt.subplots(figsize=(11, 5))
    for offset, name, report in zip((-0.2, 0.2), names, reports):
        values = [report[column] if report[column] is not None else np.nan for column in columns]
        bars = axis.bar(x + offset, values, width=0.4, label=name)
        axis.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    axis.set(xticks=x, xticklabels=['Accuracy', 'Macro F1', 'Closed precision', 'Closed recall', 'ROC AUC', 'Average precision'],
             ylabel='Score', ylim=(0, 1.14), title='Held-out test performance (closed = positive)')
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths.reports / 'performance_comparison.png', dpi=160)
    plt.close(figure)

    targets = tables[0].label.to_numpy() == 0
    for curve in ('roc', 'pr'):
        figure, axis = plt.subplots(figsize=(6, 5))
        if np.unique(targets).size == 2:
            for name, table, column, report in zip(names, tables, score_columns, reports):
                if curve == 'roc':
                    x, y, _ = roc_curve(targets, table[column])
                    label = f'{name} (AUC={report["roc_auc"]:.3f})'
                else:
                    y, x, _ = precision_recall_curve(targets, table[column])
                    label = f'{name} (AP={report["average_precision"]:.3f})'
                axis.plot(x, y, label=label)
            axis.legend()
        if curve == 'roc':
            axis.plot([0, 1], [0, 1], '--', color='gray')
            axis.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC: closed eyes are positive')
        else:
            axis.axhline(targets.mean(), linestyle='--', color='gray')
            axis.set(xlabel='Recall', ylabel='Precision', title='Precision-recall: closed eyes are positive')
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        figure.tight_layout()
        figure.savefig(paths.reports / f'{curve}_comparison.png', dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, name, report in zip(axes, names, reports):
        matrix = np.asarray(report['confusion_matrix'])
        axis.imshow(matrix, cmap='Blues')
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha='center', va='center',
                          color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
        axis.set(title=name, xlabel='Predicted', ylabel='True', xticks=[0, 1], yticks=[0, 1],
                 xticklabels=['Closed', 'Open'], yticklabels=['Closed', 'Open'])
    figure.tight_layout()
    figure.savefig(paths.reports / 'confusion_matrices.png', dpi=160)
    plt.close(figure)
    print(comparison.to_string(index=False))
    return comparison


def evaluate_saved(output_dir=RESULTS_DIR, device='auto'):
    """베스트 신경망과 PCA를 재평가하고 비교 지표·그림을 갱신한다."""
    paths = PipelinePaths(output_dir).create()
    torch.set_num_threads(1)
    dataset = EyeDataset(paths.data / 'split.csv', 'test')
    predictors = {'neural_network': NNPredictor(paths.models / 'best_model.pt', device),
                  'pca': PCAPredictor(paths.models / 'pca_model.npz')}
    reports = []
    for method, predictor in predictors.items():
        predictions, scores = [], []
        for start in range(0, len(dataset), 32):
            eyes = np.stack([dataset.image(i) for i in range(start, min(start + 32, len(dataset)))])
            labels, closed_scores = predictor.predict(eyes)
            predictions.extend(labels)
            scores.extend(closed_scores)
        table = dataset.df[['path', 'label']].copy()
        table['pred'] = predictions
        prefix, column = ('nn', 'p_closed') if method == 'neural_network' else ('pca', 'closed_score')
        table[column] = scores
        table.to_csv(paths.reports / f'{prefix}_predictions.csv', index=False)
        report = summarize(dataset.df.label.to_numpy(), np.asarray(predictions), np.asarray(scores))
        metadata = paths.reports / f'{prefix}_metrics.json'
        reports.append({**(json.loads(metadata.read_text()) if metadata.is_file() else {}), **report})
    results = save_comparison(paths, *reports)
    results.to_csv(paths.reports / 'evaluation.csv', index=False)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--folder', type=Path)
    source.add_argument('--csv', type=Path)
    source.add_argument('--mrl', type=Path)
    parser.add_argument('--results-dir', '--artifacts-dir', dest='artifacts_dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--split-by', choices=['sample', 'subject'], default='sample')
    parser.add_argument('--model', choices=['small', 'densenet'], default='small')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.1)
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
                         lr=args.lr, device=args.device, seed=args.seed, patience=args.patience,
                         min_delta=args.min_delta, val_ratio=args.val_ratio)
    pca_report = train_pca(csv, paths.root, args.pca_components, args.seed)
    save_comparison(paths, nn_report, pca_report)
    DrowsinessConfig().save(paths.models / 'alarm_config.json')
    print(f'학습 완료 → {paths.root}')


if __name__ == '__main__':
    main()
