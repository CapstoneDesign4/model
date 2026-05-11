# M2 Debounce K/N(2/3) 후처리 통합 스펙

> 대상: 모델 개발 에이전트
> 버전: v0.1 (2026-05-10)
> 참조: `docs/development-plan.md` §5.2, §5.3, §6, §10 R2 / `docs/m1-initial-model-spec.md` §6.3

---

## 1. 목표 (Scope / Non-Scope)

### 1.1 Scope — 이번 PR에서 만드는 것

| 항목 | 내용 |
|---|---|
| Debounce K/N 후처리 | 클래스별 슬라이딩 윈도우(최근 N=3) + K=2 이상 다수결 트리거 |
| DebounceState 내부 자료구조 | 클래스별 독립 deque(maxlen=N) 상태 관리 |
| 기존 Trigger 클래스 확장 | 내부에 DebounceState를 포함, 외부 인터페이스 최대한 호환 유지 |
| config/whitelist.yaml 스키마 확장 | 글로벌 `debounce` 블록 추가, 하위 호환 기본값 보장 |
| CLI 옵션 3종 추가 | `--debounce-window`, `--debounce-k`, `--no-debounce` |
| JSONL 로그 스키마 확장 | `debounce_votes` 필드 선택적 추가 |
| verbose 큐 상태 출력 | `--verbose` 시 클래스별 투표 이력 `[1,0,1]` 표시 |
| 단위 테스트 5케이스 | 합성 score 시퀀스 주입 방식 |

### 1.2 Non-Scope — 이번 PR에서 만들지 않는 것

| 항목 | 이유 / 담당 마일스톤 |
|---|---|
| WebRTC NS(노이즈 억제) 통합 | 다음 M2 후속 PR로 미룸. 별도 PR에서 `src/preprocess/noise_suppress.py` 구현 |
| NS on/off A/B 비교 스크립트 | NS 통합 PR에서 함께 작성 예정 |
| 경량 분류 헤드 학습 | M3 |
| TFLite 변환/양자화 | M4 |
| 임베디드 UART 송신 | M5 |
| 환경별 프로파일 분기 (home/street/public) | M3 이후 |
| 클래스별 개별 debounce 파라미터 | 이번은 글로벌 설정만. 클래스별 오버라이드는 M3 이후 필요시 |

---

## 2. 배경 및 동기

M1 베이스라인은 클래스별 단일 임계값 비교 + 5초 cooldown으로 구성되어 있다. 이 구조는 단일 윈도우(0.96초)에서 score가 임계값을 초과하면 즉시 이벤트를 발화한다. 결과적으로 TV에서 흘러나오는 사이렌 소리, 유사 유리 소리 등 **단발성 피크 노이즈**에 의한 false trigger가 발생할 수 있다.

`development-plan.md` §10 R2에서 "Debounce + Cooldown, 환경 프로파일별 threshold, 화이트리스트 축소 옵션"을 대응책으로 명시하고 있으며, M2 Debounce는 이 대응의 첫 번째 구현 단계이다.

### 2.1 기대 효과

| 상황 | M1(단순 cooldown) | M2(Debounce 추가) |
|---|---|---|
| 단발성 score 피크(1윈도우만 초과) | 즉시 trigger | 3개 중 1개만 초과 → trigger 안 됨 |
| 지속적 위험음(2윈도우 이상 초과) | 첫 윈도우에서 trigger | 2/3 충족 시 trigger (지연 최대 0.48s) |
| cooldown 중 재발화 | cooldown이 억제 | debounce + cooldown 이중 억제 |

### 2.2 트레이드오프

| 항목 | 내용 |
|---|---|
| 최대 추가 지연 | hop×(N-1) = 0.48s × 2 = 0.96s. M1 대비 최대 0.96초 추가. 전체 end-to-end ≤ 1.0s 목표 내 허용 범위. |
| 극단 단발음(총성·폭발) Recall 하락 가능성 | 총성·폭발은 0.96s 내에서도 여러 패치에 걸쳐 high score가 지속되어 영향 제한적. 단위 테스트에서 검증 필요. |
| 구현 복잡도 증가 | deque 상태 추가로 구조 복잡도 소폭 증가. 인터페이스 변경은 최소화. |

---

## 3. Debounce 알고리즘 사양

### 3.1 핵심 개념

```
슬라이딩 윈도우 크기 N = 3
트리거 임계 K = 2

각 클래스에 대해 독립적으로:
  - 최근 N개 윈도우의 이진 투표(0 or 1)를 deque에 보관
  - 투표 값 = 1: 해당 윈도우에서 score >= threshold
  - 투표 값 = 0: 해당 윈도우에서 score < threshold
  - sum(deque) >= K 이면 debounce 통과
  - debounce 통과 AND 해당 클래스 cooldown 미적용 중 → emit_event
```

### 3.2 파라미터 정의

| 파라미터 | 기본값 | 설명 | 변경 가능 |
|---|---|---|---|
| `N` (window) | 3 | 슬라이딩 윈도우 크기 (보관할 과거 윈도우 수) | CLI `--debounce-window`, YAML `debounce.window` |
| `K` (k) | 2 | 트리거에 필요한 최소 양성 투표 수 | CLI `--debounce-k`, YAML `debounce.k` |
| `cooldown_sec` | 5 | debounce 통과 후 동일 클래스 재발화 억제 시간(초) | whitelist.yaml 클래스별 `cooldown_sec` |
| `threshold` | 0.5 | 투표 이진화 기준 score | whitelist.yaml 클래스별 `threshold` |

### 3.3 K/N 제약 조건

- 반드시 `1 <= K <= N` 을 만족해야 한다.
- `K = 1` 이면 M1과 동일하게 단일 윈도우 트리거 (--no-debounce 대안).
- `K = N` 이면 모든 윈도우가 양성일 때만 트리거 (가장 보수적).
- 기본값 K=2, N=3 은 "3번 중 2번" 로 균형점.

### 3.4 클래스 독립성

모든 위험 클래스는 각자 독립적인 `DebounceState` 인스턴스를 갖는다. 한 클래스의 투표 이력은 다른 클래스에 영향을 주지 않는다. `glass_shatter`처럼 복수 인덱스를 `max()`로 통합한 클래스도 단일 `DebounceState`로 관리한다.

### 3.5 윈도우 경계 처리

| 상황 | 처리 방식 |
|---|---|
| 초기 상태(deque 미만) | deque 크기가 N 미만이어도 평가 실시. sum(deque) >= K 조건 자체로 자연스럽게 처리됨. (예: N=3, K=2, deque=[1,1] → sum=2 >= K → trigger 가능) |
| N 초과 시 | `deque(maxlen=N)` 에 의해 가장 오래된 항목이 자동 제거 |

---

## 4. 상태 머신 의사코드

### 4.1 자료구조

```
class DebounceState:
    votes   : deque[int]      # maxlen=N, 원소는 0 또는 1
    N       : int             # 슬라이딩 윈도우 크기
    K       : int             # 트리거 임계 투표 수
    last_trigger_ts : float   # 마지막 emit 시각 (Unix epoch), 초기값 -inf

class DangerClassState:
    key             : str     # 클래스 키 (예: "screaming")
    yamnet_indices  : list[int]
    threshold       : float
    cooldown_sec    : float
    debounce        : DebounceState
```

### 4.2 윈도우 처리 루프 의사코드

```
함수 process_window(frame: float32[15360], ts: float) -> list[EmitEvent]:
    # 1. YAMNet 추론
    scores_521 : float32[521] = yamnet(frame).mean(axis=0)

    # 2. 화이트리스트 클래스별 처리
    emitted : list[EmitEvent] = []

    for cls in danger_classes:                                    # 12종 독립
        # 2-1. 복수 인덱스 통합 (glass 계열)
        score : float = max(scores_521[i] for i in cls.yamnet_indices)

        # 2-2. 투표 이진화 및 deque 갱신
        vote : int = 1 if score >= cls.threshold else 0
        cls.debounce.votes.append(vote)                          # maxlen=N 자동 관리

        # 2-3. debounce 다수결 평가
        if sum(cls.debounce.votes) >= cls.debounce.K:
            # 2-4. cooldown 체크
            if (ts - cls.debounce.last_trigger_ts) >= cls.cooldown_sec:
                # 2-5. emit
                cls.debounce.last_trigger_ts = ts
                emitted.append(EmitEvent(
                    timestamp  = ts,
                    class_key  = cls.key,
                    score      = score,
                    votes      = list(cls.debounce.votes)        # 로그/verbose용
                ))

    return emitted
```

### 4.3 --no-debounce 모드 의사코드

```
# --no-debounce 플래그가 설정된 경우
# process_window 내부 2-3 단계를 아래로 대체:

if vote == 1:    # deque 무시, 단일 윈도우 판단
    if (ts - cls.debounce.last_trigger_ts) >= cls.cooldown_sec:
        cls.debounce.last_trigger_ts = ts
        emitted.append(EmitEvent(...))
```

### 4.4 상태 전이 다이어그램

```
[윈도우 도착]
      |
      v
[YAMNet 추론 → score]
      |
      v
[vote = (score >= threshold) ? 1 : 0]
      |
      v
[deque.append(vote)]  ← maxlen=N, 오래된 항목 자동 드롭
      |
      v
[sum(deque) >= K ?]
    /         \
  Yes          No
   |            |
   v            v
[cooldown 미적용 ?]   [skip, 다음 윈도우 대기]
    /         \
  Yes          No (cooldown 중)
   |            |
   v            v
[emit_event]   [skip, cooldown 대기]
[last_trigger_ts = ts]
```

---

## 5. 설정 파일 스키마 변경

### 5.1 config/whitelist.yaml 변경안

```yaml
# ─────────────────────────────────────────────────
# 글로벌 debounce 설정 (클래스별 오버라이드 없으면 이 값 사용)
# M2 신규 추가. 해당 블록이 없으면 기본값 window=3, k=2 적용.
# ─────────────────────────────────────────────────
debounce:
  window: 3     # int, 슬라이딩 윈도우 크기 (N)
  k: 2          # int, 트리거 최소 양성 투표 수 (K)

# ─────────────────────────────────────────────────
# 위험 클래스 목록 (M1과 동일 구조, 변경 없음)
# ─────────────────────────────────────────────────
danger_classes:
  - key: screaming
    yamnet_index: 11
    threshold: 0.5
    cooldown_sec: 5
  - key: baby_cry
    yamnet_index: 20
    threshold: 0.5
    cooldown_sec: 5
  - key: glass_shatter
    yamnet_indices: [435, 437]
    threshold: 0.5
    cooldown_sec: 5
  - key: breaking
    yamnet_index: 464
    threshold: 0.5
    cooldown_sec: 5
  - key: gunshot
    yamnet_index: 421
    threshold: 0.5
    cooldown_sec: 5
  - key: explosion
    yamnet_index: 420
    threshold: 0.5
    cooldown_sec: 5
  - key: fire_alarm
    yamnet_index: 394
    threshold: 0.5
    cooldown_sec: 5
  - key: smoke_alarm
    yamnet_index: 393
    threshold: 0.5
    cooldown_sec: 5
  - key: siren
    yamnet_index: 390
    threshold: 0.5
    cooldown_sec: 5
  - key: civil_defense_siren
    yamnet_index: 391
    threshold: 0.5
    cooldown_sec: 5
  - key: car_alarm
    yamnet_index: 304
    threshold: 0.5
    cooldown_sec: 5
  - key: vehicle_horn
    yamnet_index: 302
    threshold: 0.5
    cooldown_sec: 5
```

### 5.2 하위 호환 로딩 규칙

```
함수 load_config(path: str) -> Config:
    raw = yaml.safe_load(open(path))

    # debounce 블록이 없으면 기본값 주입 (M1 설정 파일 하위 호환)
    debounce_cfg = raw.get("debounce", {})
    global_window = debounce_cfg.get("window", 3)   # 기본 N=3
    global_k      = debounce_cfg.get("k", 2)        # 기본 K=2

    # danger_classes 구조는 M1과 동일하게 로드
    ...
    return Config(debounce_window=global_window, debounce_k=global_k, classes=...)
```

| 상황 | 동작 |
|---|---|
| M1 whitelist.yaml 그대로 사용 | `debounce` 블록 없음 → window=3, k=2 기본값 적용 |
| M2 whitelist.yaml 명시적 설정 | 해당 값 사용 |
| CLI 오버라이드 | CLI 값이 YAML보다 우선 |

---

## 6. CLI 옵션 추가

### 6.1 신규 옵션 목록

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--debounce-window` | int | 3 | 슬라이딩 윈도우 크기 N (≥ 1) |
| `--debounce-k` | int | 2 | 트리거 최소 양성 투표 수 K (1 ≤ K ≤ N) |
| `--no-debounce` | flag | False | debounce 비활성화. 단일 윈도우 트리거 (M1 동작과 동일). 디버깅·비교용 |

### 6.2 기존 옵션과의 관계

| 기존 옵션 | 변경 여부 | 비고 |
|---|---|---|
| `--input` | 변경 없음 | |
| `--threshold` | 변경 없음 | 투표 이진화 기준 score. debounce와 독립 적용 |
| `--config` | 변경 없음 | whitelist.yaml 경로 |
| `--hop` | 변경 없음 | |
| `--verbose` | 확장 | 기존 동작 유지 + debounce 큐 상태 추가 표시 |
| `--log` | 확장 | JSONL 스키마 변경 (§7 참조) |
| `--device` | 변경 없음 | |

### 6.3 파라미터 유효성 검증

```
실행 시 검증:
  if debounce_k > debounce_window:
      오류 출력: "--debounce-k ({k}) 은 --debounce-window ({N}) 이하여야 합니다."
      sys.exit(1)
  if debounce_window < 1 or debounce_k < 1:
      오류 출력: "debounce 파라미터는 1 이상이어야 합니다."
      sys.exit(1)
```

### 6.4 사용 예시

```
# 기본 debounce(2/3) 사용
python -m src.cli --input mic --threshold 0.4 --log output/run.jsonl

# debounce 비율 보수적으로 조정 (3/3)
python -m src.cli --input mic --debounce-k 3 --debounce-window 3

# debounce 비활성화 (M1 동작)
python -m src.cli --input data/sample/test.wav --no-debounce --threshold 0.0

# verbose로 큐 상태 확인
python -m src.cli --input mic --verbose --debounce-window 3 --debounce-k 2
```

---

## 7. 콘솔 출력 형식

### 7.1 기본 출력 (변경 없음)

M1과 동일한 형식을 유지한다. debounce가 내부적으로 작동하더라도 외부에 표시되는 트리거 출력 형식은 변경하지 않는다.

```
[2026-05-10 14:32:01.123] DANGER: screaming (score=0.82)
[2026-05-10 14:32:01.123] DANGER: glass_shatter (score=0.71)
[2026-05-10 14:32:03.610] -- no danger (top: siren=0.22)
```

### 7.2 --verbose 확장 출력

`--verbose` 플래그 활성화 시 매 윈도우마다 클래스별 debounce 큐 상태를 함께 출력한다.

```
[2026-05-10 14:32:01.123] WINDOW scores:
  screaming          : 0.8200  votes=[1,1,0]  sum=2/3  PASS
  baby_cry           : 0.1100  votes=[0,0,0]  sum=0/3  --
  glass_shatter      : 0.7100  votes=[0,1,1]  sum=2/3  PASS
  breaking           : 0.0400  votes=[0,0,0]  sum=0/3  --
  gunshot            : 0.0200  votes=[0,0,0]  sum=0/3  --
  explosion          : 0.0100  votes=[0,0,0]  sum=0/3  --
  fire_alarm         : 0.2300  votes=[0,0,0]  sum=0/3  --
  smoke_alarm        : 0.1900  votes=[0,0,0]  sum=0/3  --
  siren              : 0.2200  votes=[0,0,1]  sum=1/3  --
  civil_defense_siren: 0.0500  votes=[0,0,0]  sum=0/3  --
  car_alarm          : 0.0700  votes=[0,0,0]  sum=0/3  --
  vehicle_horn       : 0.0300  votes=[0,0,0]  sum=0/3  --
[2026-05-10 14:32:01.123] DANGER: screaming (score=0.82)
[2026-05-10 14:32:01.123] DANGER: glass_shatter (score=0.71)
```

출력 형식 규칙:
- `votes=[a,b,c]`: 가장 오래된 것부터 최신 순으로 표시. 현재 윈도우 투표는 맨 오른쪽(c).
- `sum=X/N`: 양성 투표 합계 / 윈도우 크기.
- `PASS`: 다수결 통과(debounce 통과), cooldown 여부 무관.
- `COOLDOWN`: 다수결 통과했으나 cooldown 중.
- `--`: 다수결 미달.

---

## 8. JSONL 로그 스키마 변경

### 8.1 M1 JSONL 스키마 (기준선)

```json
{
  "timestamp": 1746873600.123,
  "window_duration_ms": 960,
  "scores": {
    "screaming": 0.82,
    "baby_cry": 0.11,
    "glass_shatter": 0.71
  },
  "triggered": ["screaming", "glass_shatter"],
  "top_score": 0.82
}
```

### 8.2 M2 JSONL 스키마 (변경안)

```json
{
  "timestamp": 1746873600.123,
  "window_duration_ms": 960,
  "scores": {
    "screaming": 0.82,
    "baby_cry": 0.11,
    "glass_shatter": 0.71
  },
  "triggered": ["screaming", "glass_shatter"],
  "top_score": 0.82,
  "debounce_votes": {
    "screaming": [1, 1, 0],
    "baby_cry": [0, 0, 0],
    "glass_shatter": [0, 1, 1]
  }
}
```

### 8.3 스키마 변경 결정 근거

| 항목 | 결정 | 이유 |
|---|---|---|
| `triggered` 필드 유지 | 유지 | 임베디드 수신 측 파싱 코드 변경 없음 |
| `top_score` 필드 유지 | 유지 | 기존 분석 스크립트와 호환 |
| `debounce_votes` 추가 | 추가 (선택적) | 오프라인 분석, 임계값 튜닝, 디버깅에 필수. 파일 크기 소폭 증가는 허용 범위 |
| `debounce_votes` 상시 기록 여부 | 상시 기록 | verbose 플래그와 무관하게 JSONL에는 항상 기록. 분석 편의성 우선 |

`debounce_votes`는 `--log` 옵션 사용 시에만 기록된다. 콘솔 출력에서는 `--verbose` 시에만 표시.

---

## 9. postprocess/trigger.py 변경 설계

### 9.1 설계 방향 결정

| 방안 | 설명 | 장점 | 단점 | 결정 |
|---|---|---|---|---|
| A. 기존 Trigger 클래스 내부에 DebounceState 추가 | Trigger 클래스 유지, 내부 상태만 확장 | 외부 인터페이스 변경 최소, 하위 호환 | 클래스 내부 복잡도 증가 | **채택** |
| B. 새 DebouncedTrigger 클래스로 교체 | 기존 Trigger를 deprecated 처리 후 신규 클래스 도입 | 코드 명확성 높음 | cli.py 등 호출부 수정 필요 | 비채택 |
| C. Trigger를 상속하는 DebouncedTrigger | 상속 구조 | OOP 명확 | 불필요한 계층 추가 | 비채택 |

### 9.2 변경 후 Trigger 클래스 구조 (의사코드)

```
class DebounceState:
    속성:
        votes   : deque[int]    # maxlen=N
        N       : int
        K       : int
        last_trigger_ts : float = -inf

    메서드:
        push(vote: int) -> None
            # deque.append(vote)
        is_debounce_passed() -> bool
            # return sum(self.votes) >= self.K
        is_cooldown_active(now: float, cooldown_sec: float) -> bool
            # return (now - self.last_trigger_ts) < cooldown_sec
        record_trigger(ts: float) -> None
            # self.last_trigger_ts = ts

class Trigger:
    속성:
        classes          : list[DangerClassConfig]
        debounce_states  : dict[str, DebounceState]   # key: class_key
        no_debounce      : bool                        # --no-debounce 플래그

    생성자(classes, debounce_window, debounce_k, no_debounce=False):
        # 각 클래스에 대해 DebounceState 초기화
        for cls in classes:
            debounce_states[cls.key] = DebounceState(N=debounce_window, K=debounce_k)
        self.no_debounce = no_debounce

    메서드:
        evaluate(scores: dict[str, float], ts: float) -> TriggerResult
            # §4.2 의사코드 참조
            # no_debounce=True 시 §4.3 동작
            # 반환: TriggerResult(triggered, scores, votes)
```

### 9.3 TriggerResult 반환 타입

```
class TriggerResult:
    triggered  : list[str]              # 이번 윈도우에서 emit된 클래스 키 목록
    scores     : dict[str, float]       # 화이트리스트 12종 score
    votes      : dict[str, list[int]]   # 클래스별 최신 debounce deque 상태
    top_score  : float | None           # triggered 중 최고 score, 없으면 None
```

### 9.4 파일 모드 / 마이크 모드 공통 동작

`Trigger.evaluate()`는 오디오 소스에 독립적이다. `cli.py`가 파일 모드/마이크 모드 각각에서 추출한 `frame: float32[15360]`과 `ts: float`를 동일하게 `process_window()` → `Trigger.evaluate()`로 전달하므로, 두 모드에서 debounce 동작이 동일하게 적용된다.

단, 파일 모드에서는 타임스탬프 `ts`를 실제 벽시계 시각이 아닌 **파일 내 오프셋 시각** (hop 단위 누적 시각)으로 계산한다. cooldown 판정에 사용되므로 단조 증가값이면 충분하다.

---

## 10. 단위 테스트 케이스 목록

테스트 파일 경로: `tests/test_debounce_trigger.py`

모든 테스트는 합성 score 시퀀스를 직접 주입하는 방식으로 작성한다. YAMNet 모델 로딩 없이 실행 가능해야 한다.

### 10.1 케이스 정의

| ID | 케이스명 | 입력 시퀀스 (윈도우별 vote) | K/N | 기대 결과 | 검증 내용 |
|---|---|---|---|---|---|
| TC-1 | 첫 두 윈도우 양성 | `[1, 1, 0]` | 2/3 | **trigger 발생** | 1번째+2번째 양성 → 3번째 윈도우(vote=0) 처리 후 sum=2 >= K=2 |
| TC-2 | 비연속 양성 | `[1, 0, 1]` | 2/3 | **trigger 발생** | 1번째+3번째 양성 → sum=2 >= K=2 |
| TC-3 | 단발성 피크 | `[1, 0, 0]` | 2/3 | **trigger 없음** | sum=1 < K=2 → debounce 미통과 |
| TC-4 | 다수결 통과 후 cooldown 중 재시도 | `[1, 1, 0]` → (cooldown 내) `[1, 1, 0]` | 2/3 | 첫 번째만 trigger, 두 번째 **trigger 없음** | cooldown 5s 내 재발화 억제 확인 |
| TC-5 | --no-debounce 단일 윈도우 트리거 | `[1]` (vote=1) | N/A | **trigger 발생** | no_debounce=True 시 deque 무시, 단일 양성으로 즉시 trigger |

### 10.2 TC-4 타임스탬프 처리 방식

TC-4에서 cooldown 체크를 위해 `ts` 값을 직접 조작한다.

```
# TC-4 의사코드
trigger = Trigger(classes, debounce_window=3, debounce_k=2, no_debounce=False)
t0 = 0.0

# 첫 번째 트리거: ts=0.0, 1.0, 2.0 (hop=1.0 가정)
trigger.evaluate(scores_high, ts=0.0)  # vote=1
trigger.evaluate(scores_high, ts=1.0)  # vote=1
result1 = trigger.evaluate(scores_low, ts=2.0)   # vote=0, sum=2 -> PASS
assert "screaming" in result1.triggered

# 두 번째 트리거 시도: ts=2.5 (cooldown_sec=5, elapsed=0.5 < 5)
trigger.evaluate(scores_high, ts=2.5)
trigger.evaluate(scores_high, ts=3.0)
result2 = trigger.evaluate(scores_low, ts=3.5)
assert "screaming" not in result2.triggered   # cooldown 억제
```

### 10.3 TC-1/TC-2/TC-3 공통 픽스처

```
# 공통 설정
debounce_window = 3
debounce_k = 2
threshold = 0.5

scores_above = {"screaming": 0.8, ...}  # screaming score >= threshold
scores_below = {"screaming": 0.2, ...}  # screaming score < threshold
```

---

## 11. 검증 방법

### 11.1 합성 데이터 단위 테스트 (자동화)

```
pytest tests/test_debounce_trigger.py -v
```

YAMNet 모델을 로드하지 않으므로 네트워크 없이 실행 가능. CI에서 기본 실행 대상.

### 11.2 기존 스크립트 회귀 검증

```
python scripts/verify_inference.py
```

M1 파이프라인이 그대로 동작하는지 확인. debounce 추가 후에도 PASS를 유지해야 한다.

### 11.3 WAV 통합 테스트 (수동)

```
# cooldown + debounce 결합 확인
python -m src.cli --input data/sample/test.wav --threshold 0.4 --verbose --log output/debounce_test.jsonl

# --no-debounce 비교 (M1 동작 재현)
python -m src.cli --input data/sample/test.wav --threshold 0.4 --no-debounce --log output/no_debounce_test.jsonl
```

두 JSONL 파일에서 `triggered` 발화 횟수를 비교한다. debounce 활성 시 발화 횟수가 같거나 적어야 한다.

### 11.4 마이크 모드 정성 검증

```
python -m src.cli --input mic --threshold 0.4 --verbose --log output/mic_test.jsonl
```

1분 동작 중:
- 손가락을 한 번만 튕기는 소리(단발성 피크) → trigger 없음 확인.
- 손뼉을 2~3회 연속 치는 소리 → trigger 발생 확인.
- 발화 횟수를 `--no-debounce` 결과와 주관적으로 비교.

### 11.5 검증 행렬

| 검증 항목 | 방법 | 자동화 | 성공 기준 |
|---|---|---|---|
| TC-1 ~ TC-5 단위 테스트 | pytest | 자동 | 5케이스 모두 PASS |
| verify_inference.py 회귀 | 스크립트 실행 | 자동 | 기존 PASS 유지 |
| WAV 통합 JSONL 비교 | 수동 | 반수동 | debounce 활성 시 triggered 횟수 <= no-debounce 횟수 |
| 마이크 1분 정성 | 직접 테스트 | 수동 | 단발 피크 false trigger 육안 감소 확인 |
| CLI 옵션 유효성 오류 처리 | `--debounce-k 5 --debounce-window 3` 실행 | 수동 | 오류 메시지 출력 후 종료 코드 1 |

---

## 12. Exit Criteria (M2 Debounce 부분 완료 조건)

| 조건 | 측정 방법 | 합격 기준 |
|---|---|---|
| 단위 테스트 5케이스 통과 | `pytest tests/test_debounce_trigger.py -v` | TC-1 ~ TC-5 전체 PASS |
| 기존 verify_inference.py PASS 유지 | `python scripts/verify_inference.py` | PASS (회귀 없음) |
| CLI 신규 옵션 동작 | `python -m src.cli --help` | `--debounce-window`, `--debounce-k`, `--no-debounce` 항목 표시 |
| WAV 파일 debounce 결합 동작 확인 | JSONL 출력 비교 | debounce 활성 결과의 triggered 횟수 <= no-debounce 결과 |
| 마이크 1분 false trigger 정성 감소 | 직접 관찰 | 단발성 피크 이벤트가 눈에 띄게 감소 |
| JSONL debounce_votes 필드 확인 | JSONL 파일 확인 | 매 윈도우 레코드에 `debounce_votes` 포함 |

---

## 13. 다음 단계 연결

이 PR이 머지되면 M2의 나머지 작업으로 다음 후속 PR이 진행된다.

| 순서 | PR 내용 | 주요 파일 |
|---|---|---|
| 다음 PR (M2-NS) | WebRTC NS 통합 + `src/preprocess/noise_suppress.py` 구현 | `src/preprocess/noise_suppress.py`, `config/whitelist.yaml` |
| 그 다음 PR (M2-AB) | NS on/off A/B 비교 스크립트 + FAR 측정 | `scripts/compare_ns.py`, `experiments/` |
| M3 | 경량 분류 헤드 학습 (FSD50K + UrbanSound8K) | `src/model/head.py`, `scripts/train_head.py` |

### 13.1 NS 통합 PR에서 본 문서 갱신 예정 항목

- §2 배경: NS 통합 후 false trigger 감소 효과 수치 업데이트.
- §11 검증 방법: NS on/off 비교 방법론 추가.
- §12 Exit Criteria: FAR 관련 수치 기준 추가.

---

*문서 버전: v0.1 (2026-05-10). M2 NS 통합 PR에서 본 문서 §2, §11, §12 갱신 예정.*
