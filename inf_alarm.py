"""양안 졸음 조건을 판단하고 추론을 멈추지 않는 경고음을 낸다."""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
import time


@dataclass(frozen=True)
class DrowsinessConfig:
    closed_sec: float = 1.5
    smooth: int = 7
    threshold: float = 0.5
    cooldown: float = 0.5
    max_gap: float = 0.5

    def __post_init__(self):
        if not (self.closed_sec > 0 and self.smooth > 0 and 0 <= self.threshold <= 1
                and self.cooldown >= 0 and self.max_gap > 0):
            raise ValueError("알람 시간·평균 길이는 양수, 임계값은 0~1이어야 합니다")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DrowsinessResult:
    status: str
    elapsed: float = 0.0
    triggered: bool = False
    score: float | None = None


class DrowsinessMonitor:
    def __init__(self, config=None):
        self.config = config or DrowsinessConfig()
        self.history = deque(maxlen=self.config.smooth)
        self.last_trigger = None
        self.reset()

    def reset(self):
        self.history.clear()
        self.closed_since = self.last_observation = None

    def update(self, closed_scores, now=None):
        now = time.monotonic() if now is None else now
        if closed_scores is None:
            self.reset()
            return DrowsinessResult("UNKNOWN")
        scores = list(closed_scores)
        if len(scores) != 2 or not all(0 <= value <= 1 for value in scores):
            raise ValueError("양쪽 눈의 감음 점수(0~1)가 필요합니다")
        if self.last_observation is not None and not 0 <= now - self.last_observation <= self.config.max_gap:
            self.reset()
        self.last_observation = now
        self.history.append(min(scores))
        score = sum(self.history) / len(self.history)
        if score < self.config.threshold:
            self.closed_since = None
            return DrowsinessResult("OPEN", score=score)
        if self.closed_since is None:
            self.closed_since = now
        elapsed = now - self.closed_since
        alarm = elapsed >= self.config.closed_sec
        triggered = alarm and (self.last_trigger is None or now - self.last_trigger >= self.config.cooldown)
        if triggered:
            self.last_trigger = now
        return DrowsinessResult("ALARM" if alarm else "CLOSED", elapsed, triggered, score)


def beep():
    if sys.platform == "win32":
        import winsound
        winsound.Beep(2000, 300)
    else:
        print("\a", end="", flush=True)


class AlarmPlayer:
    def __init__(self):
        self.thread = None

    def trigger(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=beep, daemon=True)
            self.thread.start()

    def close(self):
        if self.thread is not None:
            self.thread.join(timeout=0.5)
