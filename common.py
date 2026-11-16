"""공통 모듈: 데이터셋, 모델(경량 CNN / DenseNet-121), 전처리.
PCA 담당(오승민)과 같은 split CSV, 같은 전처리(흑백, IMG_SIZE)를 써야 공정 비교가 됩니다.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models

IMG_SIZE = 64          # 경량 CNN 입력. DenseNet은 내부에서 224로 업샘플
MEAN, STD = 0.5, 0.5   # 흑백 1채널 정규화


def load_gray(path, size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.equalizeHist(img)          # 조명 변화 완화(PCA도 동일하게 적용 권장)
    return img


def to_tensor(img):
    x = torch.from_numpy(img).float().div(255.0)
    return ((x - MEAN) / STD).unsqueeze(0)   # (1,H,W)


class EyeDataset(Dataset):
    """split CSV 컬럼: path,label(0=감음,1=뜸),subject,split[,glasses,lighting ...]"""

    def __init__(self, csv, split, augment=False):
        df = pd.read_csv(csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment
        # 프로세스마다 읽기 전용 mmap을 열어 전체 이미지를 RAM에 복제하지 않음.
        self.cache_path = Path(csv).with_suffix(".npy") if "idx" in df.columns else None
        self.cache = None
        if self.cache_path is not None:
            self._open_cache()

    def _open_cache(self):
        self.cache = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        if self.cache.dtype != np.uint8 or self.cache.shape[1:] != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"잘못된 캐시 형식: {self.cache_path}")
        idx = self.df["idx"]
        if len(idx) and (idx.isna().any() or (idx != idx.astype(int)).any()
                         or idx.min() < 0 or idx.max() >= len(self.cache)):
            raise ValueError(f"캐시 idx 범위/형식 오류: {self.cache_path}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = None
        return state

    def __len__(self):
        return len(self.df)

    def _aug(self, img):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)                       # 좌/우 눈 대칭
        if np.random.rand() < 0.8:                       # 밝기/대비(조명 변화 모사)
            a = np.random.uniform(0.6, 1.4)
            b = np.random.uniform(-40, 40)
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if np.random.rand() < 0.5:                       # 작은 회전/이동(크롭 오차 모사)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-15, 15), 1.0)
            M[:, 2] += np.random.uniform(-0.08, 0.08, 2) * w
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return img

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.cache_path is not None and self.cache is None:
            self._open_cache()
        img = self.cache[int(r["idx"])].copy() if self.cache is not None else load_gray(r["path"])
        if self.augment:
            img = self._aug(img)
        return to_tensor(img), int(r["label"])


# ---------------- 모델 ----------------
def conv_bn(cin, cout, stride=1, groups=1):
    k = 3
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """경량 CNN 모델 (depthwise-separable, 약 0.1M 파라미터)."""

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
    """논문(Hamzah et al.) 기반 분류기: ImageNet 사전학습 DenseNet-121, 흑백 입력 대응."""

    def __init__(self, pretrained=True, n_cls=2):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=weights)
        old = m.features.conv0                      # 3ch 가중치를 합쳐 1ch로
        new = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(1, keepdim=True))
        m.features.conv0 = new
        m.classifier = nn.Linear(m.classifier.in_features, n_cls)
        self.m = m

    def forward(self, x):
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.m(x)


def build_model(name, pretrained=True):
    if name == "small":
        return SmallCNN()
    if name == "densenet":
        return DenseNetClassifier(pretrained)
    raise ValueError(name)

