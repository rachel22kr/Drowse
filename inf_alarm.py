"""양안 감음 지속시간과 감기 시작 시점 간 주기로 졸음을 판단한다."""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
import time


@dataclass(frozen=True)
class DrowsinessConfig:
    closed_sec: float = 0.5
    smooth: int = 7
    threshold: float = 0.5
    cooldown: float = 0.5
    max_gap: float = 0.5
    closure_period_sec: float = 3.0
    buffer_sec: float = 4.0

    def __post_init__(self):
        if not (self.closed_sec > 0 and self.smooth > 0 and 0 <= self.threshold <= 1
                and self.cooldown >= 0 and self.max_gap > 0
                and self.closure_period_sec > 0 and self.buffer_sec > 0):
            raise ValueError("알람 시간·평균 길이는 양수, 임계값은 0~1이어야 합니다")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        # 이전 지속시간 전용 설정은 새 기본 조건으로 전환한다.
        if "closure_period_sec" not in values or "buffer_sec" not in values:
            values["closed_sec"] = cls().closed_sec
        return cls(**values)


@dataclass(frozen=True)
class DrowsinessResult:
    status: str
    elapsed: float = 0.0
    triggered: bool = False
    score: float | None = None
    period: float | None = None


class DrowsinessMonitor:
    def __init__(self, config=None):
        self.config = config or DrowsinessConfig()
        self.history = deque(maxlen=self.config.smooth)
        self.onsets = deque()
        self.last_trigger = None
        self.reset()

    def reset(self):
        self.history.clear()
        self.onsets.clear()
        self.closed_since = self.last_observation = None
        self.period = None

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
        while self.onsets and now - self.onsets[0] > self.config.buffer_sec:
            self.onsets.popleft()
        self.history.append(min(scores))
        score = sum(self.history) / len(self.history)
        if score < self.config.threshold:
            self.closed_since = None
            self.period = None
            return DrowsinessResult("OPEN", score=score)
        if self.closed_since is None:
            self.closed_since = now
            self.period = now - self.onsets[-1] if self.onsets else None
            self.onsets.append(now)
        elapsed = now - self.closed_since
        alarm = (elapsed + 1e-9 >= self.config.closed_sec and self.period is not None
                 and self.period <= self.config.closure_period_sec + 1e-9)
        triggered = alarm and (self.last_trigger is None or now - self.last_trigger >= self.config.cooldown)
        if triggered:
            self.last_trigger = now
        return DrowsinessResult("ALARM" if alarm else "CLOSED", elapsed, triggered, score, self.period)


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
