"""직접 찍은 영상/사진 -> MediaPipe로 양쪽 눈 개별 크롭 (Hamzah 논문 방식, 35% 패딩).
realtime.py와 '완전히 같은' 크롭 함수를 써서 학습/실시간 입력을 일치시킵니다.

  pip install mediapipe==0.10.14 opencv-python
  # 영상 하나 = 한 사람, 한 상태 (예: 시영_open.mp4, 시영_closed.mp4)
  python crop_eyes.py --input videos/시영_open.mp4 --subject siyoung --label open --out data/our_eyes --every 3
결과: data/our_eyes/siyoung/open/xxxx_L.png, xxxx_R.png
그다음: python prepare_split.py --folder data/our_eyes --out split_ours.csv
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from realtime import LEFT, RIGHT, crop_eye


def frames(path, every):
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            img = cv2.imread(str(f))
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(str(p))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield fr
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="영상 파일 또는 사진 폴더")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--label", choices=["open", "closed"], required=True)
    ap.add_argument("--out", default="data/our_eyes")
    ap.add_argument("--every", type=int, default=3, help="영상에서 N프레임마다 1장 (비슷한 사진 중복 방지)")
    a = ap.parse_args()

    out = Path(a.out) / a.subject / a.label
    out.mkdir(parents=True, exist_ok=True)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         refine_landmarks=True)
    saved = fail = 0
    for k, fr in enumerate(frames(a.input, a.every)):
        res = fm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            fail += 1
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        lm = res.multi_face_landmarks[0].landmark
        for tag, idx in (("L", LEFT), ("R", RIGHT)):
            eye = crop_eye(gray, lm, idx)
            if eye is not None:
                cv2.imwrite(str(out / f"{k:05d}_{tag}.png"), eye)
                saved += 1
    print(f"saved {saved} eye images -> {out} (face not found: {fail} frames)")


if __name__ == "__main__":
    main()
