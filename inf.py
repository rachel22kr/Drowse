"""눈 사진·웹캠의 신경망/PCA 추론. 웹캠에서는 4초 이력으로 졸음 조건·알람을 실행한다."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_preprocess import (RESULTS_DIR, IMG_SIZE, LABEL_NAMES, MEAN, PREPROCESSING,
                              STD, PipelinePaths, load_gray, project_path, to_tensor)
from inf_alarm import AlarmPlayer, DrowsinessConfig, DrowsinessMonitor
from inf_preprocess import EyePreprocessor, FrameBuffer


def conv_bn(cin, cout, stride=1, groups=1):
    k = 3
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self, width=32, n_cls=2):
        super().__init__()
        w = width

        def ds(cin, cout, s):  # depthwise + pointwise
            return nn.Sequential(conv_bn(cin, cin, s, groups=cin),
                                 nn.Conv2d(cin, cout, 1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            conv_bn(1, w, 2),          # 32x32
            ds(w, w * 2, 1),
            ds(w * 2, w * 2, 2),       # 16x16
            ds(w * 2, w * 4, 1),
            ds(w * 4, w * 4, 2),       # 8x8
            ds(w * 4, w * 8, 2),       # 4x4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(w * 8, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


class DenseNetClassifier(nn.Module):
    def __init__(self, n_cls=2):
        super().__init__()
        from torchvision import models
        m = models.densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.classifier = nn.Linear(m.classifier.in_features, n_cls)
        self.m = m

    def forward(self, x):
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.m(x)


def build_model(name):
    if name == "small":
        return SmallCNN()
    if name == "densenet":
        return DenseNetClassifier()
    raise ValueError(name)


class NNPredictor:
    def __init__(self, path, device="auto"):
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if device == "auto" else device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("preprocessing", PREPROCESSING) != PREPROCESSING:
            raise ValueError("학습 모델의 전처리 규격이 다릅니다")
        labels = {int(key): value for key, value in state.get("label_names", LABEL_NAMES).items()}
        if labels != LABEL_NAMES:
            raise ValueError("모델 레이블은 0=closed, 1=open이어야 합니다")
        self.model = build_model(state["model"]).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, eyes):
        batch = torch.stack([to_tensor(image) for image in eyes]).to(self.device)
        probabilities = self.model(batch).softmax(1).cpu().numpy()
        return probabilities.argmax(1), probabilities[:, 0]


class PCAPredictor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as state:
            if state["format"].item() != "eye_pca_centroid_direction_v2":
                raise ValueError("train.py에서 생성한 PCA 모델을 사용하세요")
            if json.loads(state["labels"].item()) != {str(key): value for key, value in LABEL_NAMES.items()}:
                raise ValueError("PCA 레이블은 0=closed, 1=open이어야 합니다")
            if json.loads(state["preprocessing"].item()) != PREPROCESSING:
                raise ValueError("PCA 모델과 현재 전처리 규격이 다릅니다")
            self.mean = state["mean"].copy()
            self.components = state["components"].copy()
            self.direction = state["direction"].copy()
            self.threshold = float(state["threshold"])
            self.score_scale = float(state["score_scale"])

    def decision_function(self, eyes):
        """저장한 PCA 경계 기준 margin: 양수=뜸, 0 이하=감음."""
        pixels = np.asarray(eyes, dtype=np.float32).reshape(-1, IMG_SIZE * IMG_SIZE)
        vectors = (pixels / 255 - MEAN) / STD
        return (vectors - self.mean) @ self.components.T @ self.direction - self.threshold

    def predict(self, eyes):
        margin = self.decision_function(eyes)
        closed_score = 1 / (1 + np.exp(np.clip(margin.astype(np.float64) / self.score_scale, -60, 60)))
        return (margin > 0).astype(np.int64), closed_score


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", "--artifacts-dir", dest="artifacts_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--method", choices=["both", "nn", "pca"], default="both")
    parser.add_argument("--alarm-source", choices=["nn", "pca"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path, help="웹캠 대신 눈 사진 한 장을 판정")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if args.image is not None and args.preview_only:
        parser.error("--image와 --preview-only는 함께 사용할 수 없습니다")
    args.alarm_source = args.alarm_source or ("pca" if args.method == "pca" else "nn")
    if not args.preview_only and args.method != "both" and args.alarm_source != args.method:
        parser.error("알람 기준은 실행하는 분류기와 같아야 합니다")
    return args


def predict_image(path, predictors, output_dir=RESULTS_DIR):
    """개별 눈 사진의 개폐만 판정한다. 단일 사진에는 시간 기반 알람을 적용하지 않는다."""
    path = project_path(path)
    eyes = [load_gray(path)]
    results = {}
    for name, predictor in predictors.items():
        labels, scores = predictor.predict(eyes)
        result = dict(label=int(labels[0]), state=LABEL_NAMES[int(labels[0])], closed_score=float(scores[0]))
        results[name] = result
        print(f"{name.upper()}: {result['state']} (closed_score={result['closed_score']:.4f})")
    output = PipelinePaths(output_dir).create().reports / "image_prediction.json"
    output.write_text(json.dumps(dict(image=str(path), predictions=results), indent=2), encoding="utf-8")
    return results


def draw_preview(frame, observation, lines):
    for x1, y1, x2, y2 in observation.boxes.values():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for row, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    cv2.imshow("Eye state | q or Esc to quit", frame)


def run(args):
    paths = PipelinePaths(args.artifacts_dir)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    predictors = {}
    if not args.preview_only:
        required = {"nn": "best_model.pt", "pca": "pca_model.npz"}
        missing = [paths.models / filename for name, filename in required.items()
                   if args.method in ("both", name) and not (paths.models / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"모델이 없습니다: {missing[0]}. python train.py --results-dir {paths.root}로 먼저 학습하세요")
        if args.method in ("both", "nn"):
            predictors["nn"] = NNPredictor(paths.models / "best_model.pt", args.device)
        if args.method in ("both", "pca"):
            predictors["pca"] = PCAPredictor(paths.models / "pca_model.npz")
    if args.image is not None:
        return predict_image(args.image, predictors, paths.root)
    config_path = paths.models / "alarm_config.json"
    config = DrowsinessConfig.load(config_path) if config_path.is_file() else DrowsinessConfig()
    if not args.preview_only:
        config.save(config_path)
    monitor, alarm = DrowsinessMonitor(config), AlarmPlayer()
    recent_frames = FrameBuffer(config.buffer_sec)
    with EyePreprocessor(paths.models / "face_landmarker.task") as preprocessor:
        camera = cv2.VideoCapture(args.camera)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다")
            origin = last_log = time.monotonic()
            processed = 0
            while True:
                ok, frame = camera.read()
                observed_at = time.monotonic()
                if not ok:
                    monitor.update(None, observed_at)
                    recent_frames.prune(observed_at)
                    break
                observation = preprocessor.extract(frame, (observed_at - origin) * 1000)
                lines, scores = [], {}
                if observation.eyes:
                    for name, predictor in predictors.items():
                        labels, scores[name] = predictor.predict(observation.eyes)
                        states = ["CLOSED" if label == 0 else "OPEN" for label in labels]
                        lines.append(f"{name.upper()}: L={states[0]} R={states[1]}")
                if not observation.eyes:
                    lines.append(observation.reason)
                now = time.monotonic()
                current = scores.get(args.alarm_source) if now - observed_at <= config.max_gap else None
                result = monitor.update(current, observed_at)
                recent_frames.append(observed_at, frame, result, scores)
                recent_frames.prune(now)
                if result.triggered:
                    alarm.trigger()
                period = f"{result.period:.2f}s" if result.period is not None else "waiting"
                lines.append(f"{result.status}: closed {result.elapsed:.2f}s / {config.closed_sec:g}s")
                lines.append(f"period {period} / <= {config.closure_period_sec:g}s | "
                             f"buffer {config.buffer_sec:g}s: {len(recent_frames)} frames")
                if now - last_log >= 1:
                    print(" | ".join(lines), flush=True)
                    last_log = now
                processed += 1
                if not args.no_display:
                    draw_preview(frame, observation, lines)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.max_frames and processed >= args.max_frames:
                    break
        finally:
            camera.release()
            recent_frames.clear()
            alarm.close()
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
