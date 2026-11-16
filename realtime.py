"""실시간 졸음 알람 데모 (웹캠이 있는 노트북에서 실행. SSH 서버엔 웹캠이 없음!)
  pip install mediapipe==0.10.14 opencv-python torch
  python realtime.py --ckpt ckpt/small.pt --closed_sec 1.5

흐름: MediaPipe Face Mesh -> 양쪽 눈 개별 크롭(35% 패딩) -> 흑백 64x64 -> CNN
     -> 최근 7프레임 확률 평균 -> 양쪽 눈 모두 '감음'이 closed_sec 이상 지속 -> 경고음
얼굴/눈 검출 실패 시 '판정 불가'로 표시(타이머 진행 X).
"""
import argparse
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from common import IMG_SIZE, build_model, to_tensor

LEFT = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT = [362, 263, 387, 386, 385, 373, 374, 380]


def crop_eye(gray, lm, idx, pad=0.35):
    h, w = gray.shape
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in idx])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + 2 * pad) / 2
    x0, x1 = int(max(cx - s, 0)), int(min(cx + s, w))
    y0, y1 = int(max(cy - s, 0)), int(min(cy + s, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    eye = cv2.resize(gray[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(eye)


def beep():
    try:
        import winsound
        winsound.Beep(2000, 300)
    except ImportError:
        print("\a", end="", flush=True)   # macOS/Linux: 터미널 벨 (필요시 playsound 등으로 교체)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--closed_sec", type=float, default=1.5)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()

    c = torch.load(a.ckpt, map_location="cpu")
    model = build_model(c["model"], pretrained=False)
    model.load_state_dict(c["state_dict"])
    model.eval()

    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5)
    cap = cv2.VideoCapture(a.cam)
    hist = collections.deque(maxlen=a.smooth)   # P(감음) 이력
    closed_since, last_beep = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status, color = "UNKNOWN (no face)", (0, 255, 255)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            eyes = [crop_eye(gray, lm, LEFT), crop_eye(gray, lm, RIGHT)]
            if all(e is not None for e in eyes):
                x = torch.stack([to_tensor(e) for e in eyes])
                with torch.no_grad():
                    p_closed = torch.softmax(model(x), 1)[:, 0]   # label 0 = 감음
                hist.append(p_closed.min().item())    # 양쪽 모두 감아야 '감음'
                p = float(np.mean(hist))
                if p > a.thr:
                    closed_since = closed_since or time.time()
                    dur = time.time() - closed_since
                    status, color = f"CLOSED {dur:.1f}s (p={p:.2f})", (0, 0, 255)
                    if dur >= a.closed_sec and time.time() - last_beep > 0.5:
                        beep()
                        last_beep = time.time()
                        status = "!!! WAKE UP !!!"
                else:
                    closed_since = None
                    status, color = f"OPEN (p_closed={p:.2f})", (0, 255, 0)
        else:
            hist.clear()
            closed_since = None
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
