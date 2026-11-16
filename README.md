# Drowse 학습

KD 없이 SmallCNN 또는 DenseNet-121을 CE(label smoothing 0.05)로 단독 학습합니다.

## 전체 데이터 전처리

```bash
.venv/bin/python make_subset.py --csv split.csv --out full.csv
```

기본 `--ratio 1`은 모든 이미지를 사용합니다. 기존 사람 단위 train/val/test 분할을 유지하며,
학습에는 train만 사용합니다. 현재 전체 84,898장 중 train 63,358장, val 6,621장,
test 14,919장입니다. `subset.csv`는 예전 축소 데이터이므로 전체 학습에는 `full.csv`를 사용하세요.

전처리 순서는 흑백 → 64×64 리사이즈 → 히스토그램 평활화입니다.
CPU 1스레드에서 한 장씩 처리하여 `full.npy`에 uint8로 기록합니다(약 348 MB).
학습과 평가에서는 읽기 전용 mmap으로 필요한 이미지만 읽고,
CPU에서 증강(학습만)과 [-1, 1] 정규화를 수행한 뒤 배치만 GPU로 전송합니다.
CSV와 같은 이름의 .npy를 함께 보관하세요. 캐시를 다시 만들 때는 사용 중인 학습을 종료하세요.

## 학습

```bash
.venv/bin/python train.py --model small --out ckpt/small_full.pt
.venv/bin/python train.py --model densenet --freeze head_only --out ckpt/densenet_full.pt
```

`--csv` 기본값은 `full.csv`입니다. CUDA에서는 AMP를 기본 사용하며,
기본 배치는 SmallCNN 32, DenseNet 16입니다. 검증도 같은 배치 크기로 실행합니다.
마지막 잔여 배치도 학습합니다. DenseNet은 내부에서 224×224로 확대하므로 더 많은 GPU 메모리를 씁니다.
worker 2개, worker별 CPU 1스레드, prefetch 1로 미리 읽는 양을 제한합니다.
CPU 부하를 더 줄이려면 `--workers 0`을 사용하세요.

메모리가 부족하면 `--bs 16 --accum-steps 2`처럼 실제 배치를 줄이고 gradient를 누적하세요.
누적 학습은 큰 배치와 BatchNorm 통계가 같지는 않습니다.
AMP를 끄려면 `--no-amp`, CPU 실행은 `--device cpu`를 사용합니다.
GPU 메모리는 모델·배치·함께 실행 중인 작업에 따라 달라집니다.

동일 명령에 `--resume`을 추가하면 `_last.pt`에서 다음 epoch부터 재개합니다.
모델·CSV·배치·누적 수·epoch 수·freeze·학습률·AMP 설정을 유지해야 합니다.
기존 KD/구형 체크포인트는 새 학습의 resume 대상으로 사용하지 말고 새 출력 경로를 사용하세요.
추론용 체크포인트의 state_dict 구조는 유지됩니다.

## 평가

```bash
.venv/bin/python evaluate.py --ckpt ckpt/small_full.pt --out results_full.csv
```

전체 test 분할로 평가하며 학습 중 test를 사용하지 않습니다.
