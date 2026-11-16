# 눈 개폐 분류와 졸음 알람

눈 사진으로 신경망과 PCA 분류기를 만들고, 웹캠에서 양쪽 눈의 개폐 상태를 비교합니다. 모델은 **눈 개폐만 분류**하며, 졸음 여부는 별도의 지속시간 조건으로 판단합니다. 모든 과정의 레이블은 **`0=감음(closed)`, `1=뜸(open)`**입니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `train_preprocess.py` | 경로·입력 규격, 이미지 전처리, 데이터셋, 80:20 분할·캐시, 사진/영상의 눈 ROI 수집 |
| `train.py` | 신경망 학습, 테스트 정확도 기록·플롯, PCA와 성능 비교, 저장 모델 재평가, 전체 학습 실행 |
| `train_pca.py` | PCA 계산·저장·평가, 클래스 구분 방향 시각화 |
| `inf_preprocess.py` | MediaPipe 얼굴 랜드마크, 웹캠 양안 ROI 크롭·전처리 |
| `inf.py` | 신경망 구조, 저장 신경망·PCA 추론, 전체 웹캠 실행 |
| `inf_alarm.py` | 졸음 조건·누적 상태·설정 저장, 비차단 경고음 |

신경망 구조는 `inf.py`에 정의해 `train.py`가 함께 사용합니다. 경로와 이미지 전처리는 `train_preprocess.py`에서 공유합니다.

## 학습 파이프라인

```text
train_preprocess.py → train.py의 신경망 학습 → train_pca.py → 정확도 비교·설정 저장
```

```bash
# 기본: data/our_eyes, SmallCNN, 30 epochs, PCA 최대 64성분
python train.py

# 다른 입력 또는 학습 설정
python train.py --folder /data/eyes --epochs 30 --bs 32 --device cpu
python train.py --csv data/eyes.csv
python train.py --mrl /data/mrlEyes_2018_01 --split-by subject
python train.py --model densenet --pretrained
```

기본은 클래스 비율을 유지한 이미지 기준 **80:20** 분할입니다. 신경망과 PCA는 한 번 만든 동일 분할을 읽습니다. 같은 사람·영상의 비슷한 사진이 양쪽에 섞이지 않게 하려면 `--split-by subject`를 사용하세요. 이 경우 사람 수를 80:20으로 나누므로 이미지 수의 비율은 달라질 수 있습니다.

전처리는 **흑백 → 64×64 → [-1,1] 정규화**입니다. 캐시는 원본 사진으로 다시 만들어 uint8 메모리맵으로 저장하며, 신경망 학습에만 좌우 반전·밝기·회전 등의 증강을 적용합니다. 기존에 히스토그램 평활화를 적용한 캐시와 모델은 새 전처리로 다시 생성해야 합니다.

신경망은 매 epoch 테스트 정확도를 기록하고 지정한 마지막 epoch의 가중치를 저장합니다. 테스트 정확도로 모델을 선택하거나 조기 종료하지 않습니다.

PCA 자체는 전체 분산이 큰 축을 찾습니다. 그 공간에서 학습 레이블별 평균을 구하고 **뜬 눈 평균 − 감은 눈 평균** 방향으로 다시 투영합니다. 두 평균의 중간이 판정 경계이며, 평균·축·방향·경계는 모두 학습셋에서만 계산합니다. 성분 수는 `--pca-components`로 지정합니다.

## 추론 파이프라인

```text
inf_preprocess.py의 웹캠 양안 전처리 → inf.py의 신경망/PCA 판정 → inf_alarm.py
```

```bash
python inf.py                    # 두 방법 비교, 신경망 점수로 알람
python inf.py --method nn        # 신경망만 실행
python inf.py --method pca       # PCA만 실행, PCA 점수로 알람
python inf.py --preview-only     # 모델 없이 웹캠 전처리 확인
python inf.py --camera 1         # 다른 카메라 사용
```

기본 졸음 조건은 **양안 감음 점수의 최솟값을 최근 7회 평균한 값이 0.5 이상으로 1.5초 지속**되는 것입니다. 알람 반복 간격은 0.5초입니다. 얼굴·눈 검출 실패나 0.5초를 넘는 관측 공백에는 평균 이력과 누적 시간을 초기화합니다.

설정은 `artifacts/models/alarm_config.json`에서 바꿉니다. 경고음은 Windows의 `winsound`, Linux/macOS의 터미널 벨을 사용하며 터미널 설정에 따라 소리가 나지 않을 수 있습니다.

검증은 합성 이미지의 분할·전처리·학습·PCA·저장 모델 재평가와 모의 카메라 추론·알람 상태를 대상으로 수행합니다. 실제 데이터 정확도와 웹캠·스피커 동작은 사용 환경에서 확인해야 합니다.
