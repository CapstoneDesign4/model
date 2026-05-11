# M1 베이스라인 개선 노트 — M2 착수 전 분석

> 작성: 기획 에이전트  
> 버전: v0.1 (2026-05-11)  
> 참조: `docs/m1-initial-model-spec.md`, `docs/development-plan.md`, `docs/mic-quickstart.md`

---

## 1. 개요

### 현재 상태 (한 줄 요약)

M1 베이스라인은 YAMNet 추론 파이프라인과 12종 위험 클래스 필터링이 구동되며, 마이크 실시간 입력에서 크래시 없이 동작하지만, **실제 위험 소리로 트리거를 성공시킨 정량 검증이 아직 수행되지 않았다.**

### 실측 관찰 (약 2분 마이크 로그)

| 항목 | 관찰값 |
|---|---|
| 전체 실행 환경 | 조용한 실내, 노트북 내장 마이크 |
| 12종 클래스 점수 범위 | 0.0000 ~ 0.0025 |
| 가장 높은 반응 클래스 | `baby_cry` 0.0009~0.0025, `glass_shatter` 0.0008~0.0018 |
| `fire_alarm` 반응 | 0.0001 수준 (거의 0) |
| 트리거 발생 횟수 | 0회 (임계값 0.4 기준) |
| 해석 | 조용한 환경에서 false positive 없음 — **정상 동작** |
| 미검증 항목 | 실제 위험 소리 입력 시 트리거 성공률 전혀 측정되지 않음 |

### 결론

"조용한 환경에서 트리거가 없다"는 사실이 **시스템이 올바르게 동작한다는 증거가 아니다.** 실제 위험 소리(사이렌 재생, 유리 깨짐 영상, 비명 오디오)로 트리거 성공률을 측정해야 M1 Exit Criteria(`F1 ≥ 0.6`)를 판정할 수 있다. M2 착수 전에 이 검증이 최우선으로 완료되어야 한다.

---

## 2. 발견된 이슈 / 한계

카테고리별로 정리하며, 각 항목에 **영향도(상/중/하)** 및 **우선순위(P1~P3)** 를 표기한다.

- **P1**: M2 착수 전에 반드시 해결 또는 방향 결정이 필요한 항목
- **P2**: M2 구현과 병행하거나 M2 완료 전에 처리 권장
- **P3**: M3 이후 또는 여유 시 처리

---

### 2.1 검증/평가 부재 [영향도: 상, 우선순위: P1]

**현상**  
자체 평가셋이 존재하지 않으며, 실제 위험 소리(사이렌, 유리 깨짐, 비명 등)로 파이프라인을 통과시킨 기록이 없다. M1 Exit Criteria(`docs/m1-initial-model-spec.md` §12)에 "자체 평가셋 위험 클래스 F1 ≥ 0.6" 조건이 명시되어 있으나 아직 미측정이다.

**현재 구조의 한계**  
- `data/sample/` 디렉터리에 검증용 WAV가 없거나 부족한 상태로 추정된다.
- `scripts/verify_inference.py`는 파이프라인 통과 여부(크래시 없음)만 확인하며, 클래스별 정확도 수치를 생성하지 않는다.
- `src/cli.py:116`의 `run_file_mode`는 WAV 파일 처리를 지원하나, 정답 라벨과의 비교 로직이 없다.

**리스크**  
임계값 0.4가 실제 위험 소리에서 충분히 낮은지, 혹은 너무 낮아서 false positive가 발생하는지 알 수 없다. 임계값 적절성 없이 M2 노이즈 캔슬링을 붙이면 "NS가 도움이 됐는지 해가 됐는지" 판단 기준이 없다.

**필요한 조치**  
- ESC-50 또는 FSD50K에서 위험 클래스 샘플 최소 3개/클래스 수집 후 `data/sample/`에 배치
- 각 WAV에 대해 기대 클래스가 트리거되는지 확인하는 평가 스크립트 작성 (모델 개발 에이전트 담당)

---

### 2.2 임계값 0.4의 근거 부족 [영향도: 상, 우선순위: P1]

**현상**  
`config/whitelist.yaml`의 기본 임계값은 0.5이고, `docs/mic-quickstart.md:99`의 실행 예시는 0.4를 권장한다. 그러나 **YAMNet의 각 클래스별 사전학습 출력 분포가 동일하지 않다.** AudioSet에서 학습된 YAMNet은 자주 등장하는 클래스(Speech, Music 등)는 높은 점수를 내고, 희귀 클래스(Gunshot, Civil defense siren 등)는 같은 소리가 입력되어도 상대적으로 낮은 점수를 출력하는 경향이 있다.

**현재 구조의 한계**  
- `src/model/danger_filter.py:59`의 `override_threshold`는 모든 클래스에 단일 임계값을 적용한다. 클래스별 특성이 반영되지 않는다.
- `config/whitelist.yaml`에는 모든 클래스가 `threshold: 0.5`로 동일하게 설정되어 있다.
- 실제로 `screaming` 클래스(인덱스 11)와 `civil_defense_siren` 클래스(인덱스 391)는 YAMNet에서 정상 입력에도 출력 분포가 크게 다를 수 있다.

**가설** (검증 필요)  
위험 소리를 입력해도 일부 클래스(특히 gunshot, explosion 인덱스 421/420)의 점수가 0.4 미만으로 유지될 가능성이 있다. 이 경우 임계값 문제인지 모델 자체의 한계인지 구분이 불가능하다.

**필요한 조치**  
각 위험 클래스별로 정답 오디오를 입력했을 때의 YAMNet 점수 분포를 측정하고, PR 곡선을 그려 클래스별 최적 임계값을 도출해야 한다 (`development-plan.md` §9.1 참조).

---

### 2.3 노이즈 캔슬링 미구현 [영향도: 중, 우선순위: P2]

**현상**  
`src/preprocess/noise_suppress.py:20`은 현재 패스스루(입력을 그대로 반환)이다. `cli.py`의 데이터 흐름에서 `preprocess/` 단계 자체가 호출되지 않는다(`src/cli.py:116~170` 어디에도 `noise_suppress.suppress()` 호출이 없다).

**현재 구조의 한계**  
- M2 예정 기능이지만, WebRTC NS를 실제로 연결하는 코드가 전혀 없어서 M2에서 연결 포인트를 신규 추가해야 한다.
- `cli.py`의 `run_mic_mode`와 `run_file_mode` 양쪽에 전처리 단계 삽입 위치가 명시되어 있지 않다.

**리스크**  
- 배경 소음 환경(카페, 거리, 거실 TV)에서 false positive 또는 missed detection이 얼마나 발생하는지 현재로서는 측정 불가능하다.
- 연결 포인트 설계 없이 NS를 붙이면 파이프라인 구조가 변경될 위험이 있다.

**필요한 조치**  
- M2 전에 `run_mic_mode`/`run_file_mode` 내 전처리 삽입 위치를 인터페이스 수준에서 확정하고 `noise_suppress.suppress()` 호출 지점을 지정
- NS on/off A/B 비교 실험을 위한 플래그(`--noise-suppress` 등) 설계 필요

---

### 2.4 후처리 단순성 — debounce 미구현 [영향도: 상, 우선순위: P2]

**현상**  
`src/postprocess/trigger.py:62`는 단일 윈도우 점수가 임계값을 넘으면 즉시 이벤트를 발행한다. K/N 다수결(debounce)이 없다.

**구체적 위험**  
- **False positive**: 배경 소음 피크가 일시적으로 임계값을 넘는 경우 한 번의 윈도우만으로 알림이 발생한다. 0.96초짜리 윈도우 1개가 기준이므로 1초 미만의 잡음에도 알림이 발생 가능하다.
- **False negative**: 반대로 위험 소리가 지속되더라도 현재 debounce가 없어 cooldown(5초) 이후에만 재발행된다. 이 설계는 지속성 경보(화재 경보, 사이렌)에는 적합하나, 단발성 위험음(유리 깨짐, 총소리)은 한 번만 발생하므로 문제없다.
- **Cooldown 5초의 근거 부재**: 화재 경보는 지속음이므로 5초 cooldown이 적절할 수 있으나, 아기 울음과 같이 간헐적으로 발생하는 소리는 5초 내에 재발할 경우 놓칠 수 있다.

**필요한 조치**  
- M2에서 K/N 다수결 구현: 직전 N개 윈도우 중 K개 이상에서 임계값 초과 시 이벤트 발행 권장값: N=5, K=3 (2.4초 범위에서 3회 연속)
- 클래스별로 cooldown 값을 개별 설정할 수 있도록 `whitelist.yaml` 구조는 이미 준비되어 있으나, 실제 적정값은 검증 후 조정 필요

---

### 2.5 YAMNet 자체 한계 — 데이터 편향 [영향도: 중, 우선순위: P2]

**현상**  
YAMNet은 AudioSet(주로 YouTube) 기반으로 사전학습되어 있다. 다음 편향이 가설 수준에서 우려된다.

**가설 목록** (모두 검증 필요)

| 가설 ID | 내용 |
|---|---|
| H1 | 한국어 비명(screaming, 인덱스 11)은 영어/서구 언어 비명과 음향 특성이 달라 점수가 낮을 수 있다 |
| H2 | 한국식 사이렌(경찰/소방)은 미국·유럽식 사이렌과 주파수 패턴이 달라 YAMNet이 낮은 점수를 줄 수 있다 |
| H3 | gunshot(421)과 explosion(420)은 AudioSet에서 희귀 클래스이므로 일반 환경에서도 낮은 점수를 출력할 수 있다 |
| H4 | baby_cry(20)는 YouTube 영상에서 자주 등장하므로 실제 아기 울음에 상대적으로 높은 점수를 출력할 것으로 기대된다 |

**관련 관찰**  
실측에서 baby_cry(0.0025)가 가장 높은 반응을 보인 것은 H4와 부분적으로 일치한다. 그러나 조용한 환경에서도 baby_cry 점수가 발생하는 것은 마이크 배경 노이즈 패턴이 baby_cry 스펙트럼과 일부 겹치는 가능성도 있다.

**필요한 조치**  
M3 파인튜닝(경량 헤드 학습) 전에 각 클래스에 대한 YAMNet 베이스 점수 분포를 측정하여 H1~H4를 검증해야 한다.

---

### 2.6 로깅/모니터링 부족 [영향도: 하, 우선순위: P3]

**현상**  
`src/cli.py`의 로깅은 트리거 이벤트와 윈도우별 점수를 JSONL 파일에 기록하는 수준이다. 다음 통계 정보가 없다.

| 부재 항목 | 영향 |
|---|---|
| 분당 평균 점수 추이 | 임계값 캘리브레이션에 필요한 히스토그램 생성 불가 |
| 윈도우 처리 시간(latency) | M1 Exit Criteria "Latency 측정 완료" 미달 |
| 입력 신호 레벨(dBFS) | 마이크 게인 문제인지 모델 문제인지 구분 불가 |
| 추론 횟수 / 트리거 횟수 비율 | False alarm rate(FAR) 계산 불가 |

**관련 Exit Criteria 미달**  
`docs/m1-initial-model-spec.md` §12의 "Latency 측정 완료 — 파일 모드 100 윈도우 추론 시간 측정, 95퍼센타일 기록" 조건이 아직 수행되지 않았다.

---

### 2.7 클래스간 인덱스 혼용 가능성 [영향도: 중, 우선순위: P1]

**현상**  
`docs/development-plan.md` §2.2의 인덱스 표(초안)와 `docs/m1-initial-model-spec.md` §4의 인덱스 표가 일치하지 않는 항목이 있다. 예를 들어 development-plan.md는 `Screaming`을 인덱스 47로, m1-initial-model-spec.md는 인덱스 11로 기록한다.

**검증 필요 인덱스 불일치 목록**

| 클래스 | development-plan.md | m1-initial-model-spec.md | 실제 class_map.csv 기준 |
|---|---|---|---|
| Screaming | 47 | **11** | 검증 필요 |
| Baby cry | 67 | **20** | 검증 필요 |
| Glass/Shatter | 316 | **435, 437** | 검증 필요 |
| Gunshot | 388 | **421** | 검증 필요 |
| Explosion | 390 | **420** | 검증 필요 |
| Fire alarm | 396 | **394** | 검증 필요 |
| Siren | 391 | **390** | 검증 필요 |

`development-plan.md`는 초기 기획 문서이고 `m1-initial-model-spec.md`는 실제 `yamnet_class_map.csv`를 대조하여 수정된 값으로 표시되어 있다. 현재 구현(`config/whitelist.yaml`)이 m1-initial-model-spec.md §4의 인덱스를 따른다면 이 자체는 문제가 없다. 그러나 두 문서의 인덱스가 다르기 때문에 나중에 참조 오류가 발생할 위험이 있다.

**필요한 조치**  
`yamnet_class_map.csv`를 직접 조회하여 현재 `config/whitelist.yaml`에 등록된 인덱스가 정확한지 최종 확인. development-plan.md의 인덱스 표를 실제 값으로 정정 또는 "참고용 초안" 주석 추가.

---

### 2.8 입력 품질 변수 관리 부재 [영향도: 중, 우선순위: P2]

**현상**  
`src/audio_io/mic_stream.py`는 OS 기본 마이크 게인을 그대로 사용한다. 16kHz로 다운샘플링되지만 입력 신호 레벨 자체는 확인하지 않는다.

**문제점**

| 변수 | 영향 |
|---|---|
| 마이크 게인이 너무 낮음 | 위험 소리가 입력되어도 YAMNet 입력 범위(-1~+1)의 아주 작은 부분만 사용 → 점수 전반적으로 낮아짐 |
| 마이크 게인이 너무 높음 | 클리핑 발생 → 고조파 왜곡 → 잘못된 클래스 활성화 |
| 마이크~소리 거리 | 1m와 5m에서 같은 소리의 점수가 얼마나 다른지 미측정 |
| 16kHz 다운샘플링 품질 | `librosa.load`의 기본 리샘플러(kaiser_best)는 품질이 좋으나, 실시간 스트림(`sounddevice`)은 OS 리샘플러에 의존 — 동일한지 미검증 |

**필요한 조치**  
- 로깅에 윈도우별 RMS 또는 dBFS 값 추가 (P2)
- 추론 전 RMS가 임계값 이하이면 "무음 구간"으로 스킵하는 VAD-like 필터 고려 (P3)

---

### 2.9 임베디드 연동 인터페이스 미확정 [영향도: 하~중, 우선순위: P3]

**현상**  
`src/embedded/uart_sender.py`는 빈 스켈레톤이다. M5 예정이나 아래 항목은 M2~M3 중에 결정이 필요하다.

| 미결정 항목 | 이유 |
|---|---|
| 임베디드 타겟 보드 확정 (라즈베리파이 4 vs 5, MCU) | TFLite 변환(M4) 전에 결정해야 양자화 설정이 가능 |
| UART 보드레이트 및 핀 매핑 | ESP32/STM32 중 선택에 따라 다름 |
| 알림 우선순위 정책 | 다중 클래스 동시 트리거 시 어느 클래스를 먼저 전송할 것인지 미정 |

---

## 3. 단기 (즉시 시도 가능) 개선안

> 코드 변경 없이 또는 최소 변경으로 즉시 실행 가능한 항목.

### 3.1 위험 소리 WAV 파일 수집 후 파일 모드 검증 [P1]

**방법**: YouTube에서 "fire alarm sound 16kHz", "glass breaking sound effect", "police siren" 등을 내려받아 `data/sample/`에 배치 후 아래 명령으로 점수를 확인한다.

```
python -m src.cli --input data/sample/fire_alarm.wav --threshold 0.0 --verbose
```

`--threshold 0.0`으로 강제 출력하여 해당 클래스의 실제 YAMNet 점수를 확인한다. 이것이 **현재 할 수 있는 가장 중요한 검증**이다.

**기대 결과**: `fire_alarm` 또는 `smoke_alarm` 클래스에서 0.4 이상의 점수가 나와야 한다. 0.1 미만이 나온다면 인덱스 오류 또는 YAMNet 자체 한계 중 하나를 의심해야 한다.

### 3.2 클래스별 점수 분포 기록 [P1]

위 검증 시 `--log output/eval_xxx.jsonl`을 함께 사용하여 JSONL 기록을 남긴다. 각 파일에서 각 클래스가 어느 점수 범위를 보이는지 수작업으로라도 히스토그램을 작성하면 클래스별 임계값 설정의 근거가 된다.

### 3.3 YAMNet class_map.csv 인덱스 재확인 [P1]

TF-Hub에서 YAMNet을 로드한 후 `class_names()` 또는 `yamnet_class_map.csv`를 직접 조회하여 현재 `config/whitelist.yaml`의 13개 인덱스가 기대하는 클래스명과 정확히 일치하는지 확인한다. 이는 2.7항 이슈의 해결이다.

예상 확인 절차 (의사코드):
```
yamnet = hub.load(...)
class_map = yamnet.class_names()
for idx in [11, 20, 302, 304, 390, 391, 393, 394, 420, 421, 435, 437, 464]:
    print(idx, class_map[idx])
```

---

## 4. 중기 (M2~M3 범위) 개선안

### 4.1 K/N Debounce 후처리 구현 [M2, P2]

**설계 방향** (`src/postprocess/trigger.py` 수정):

현재 `Trigger.evaluate()` 메서드는 직전 N개 윈도우 상태를 보관하지 않는다. M2에서 클래스별 최근 N개 점수 큐를 유지하고, K개 이상이 임계값을 초과할 때만 이벤트를 발행하는 방식으로 변경한다.

권장 초기값: N=5(2.4초 범위), K=3(60% 다수결). 단발성 위험음(유리 깨짐, 총소리)은 N=3, K=1로 별도 설정을 허용하도록 `whitelist.yaml`에 `debounce_n`/`debounce_k` 필드 추가를 고려한다.

### 4.2 WebRTC NS 전처리 연결 [M2, P2]

**설계 방향**:

`src/cli.py`의 `run_mic_mode`와 `run_file_mode` 양쪽에서 `yamnet.infer_mean_scores(frame)` 호출 직전에 `noise_suppress.suppress(frame)` 호출을 삽입한다. 이 삽입 포인트는 코드 2개 위치에 불과하다.

- `src/cli.py:117`: `mean_scores = yamnet.infer_mean_scores(frame)` 앞에 전처리 삽입
- `src/cli.py:159`: 마이크 모드 동일 지점

중요한 점은 **NS on/off를 CLI 플래그로 전환할 수 있어야** A/B 비교가 가능하다는 것이다. `--no-noise-suppress` 플래그 또는 `--noise-suppress` 플래그로 제어하는 설계를 권장한다.

WebRTC NS 라이브러리 선택 기준:

| 라이브러리 | 장점 | 단점 | 현재 권장 |
|---|---|---|---|
| `webrtc-noise-gain` | 순수 Python/C 바인딩, pip 설치 | 유지보수 상태 불확실 | 1순위 시도 |
| `noisereduce` | pip 설치 용이, 문서 풍부 | 스펙트럼 감산 계열, 단발 transient 손실 위험 | 2순위 (A/B용) |
| `pyannote.audio` | 최신 VAD 포함 | 무거움, TF와 충돌 가능 | 비추천 |

`development-plan.md` §4.2의 경고 사항: NS aggressiveness가 높으면 glass shatter, gunshot 등 단발 transient 위험음의 Recall이 떨어질 수 있다. aggressiveness=1(약함)로 시작하고 A/B 비교를 통해 결정한다.

### 4.3 클래스별 임계값 캘리브레이션 [M2~M3, P2]

**방법**:

각 위험 클래스별 정답 오디오(ESC-50, FSD50K eval split)와 negative 오디오(조용한 환경, 음악, 대화)에 대해 YAMNet 점수 분포를 측정한다. 클래스별 PR 곡선에서 Recall ≥ 0.85를 만족하는 최소 임계값을 찾는다 (`development-plan.md` §9.1 참조).

**구현 위치**: `src/model/danger_filter.py`의 `DangerClassEntry`가 이미 `threshold` 필드를 클래스별로 갖고 있으므로 (`danger_filter.py:25`), `whitelist.yaml`에 클래스별 다른 임계값을 입력하면 된다. `override_threshold()` 메서드(`danger_filter.py:59`)는 전체 일괄 오버라이드이므로 클래스별 캘리브레이션과는 분리된다.

### 4.4 입력 신호 레벨 모니터링 추가 [M2, P2]

`src/audio_io/mic_stream.py`의 `iter_frames()` 또는 `cli.py`의 처리 루프에서 윈도우별 RMS를 계산하여 로그에 포함시킨다. RMS가 -40dBFS 이하이면 "무음 구간"으로 처리하여 YAMNet 추론을 스킵하는 방식을 고려한다. 이는 연산량 절감과 동시에 false positive 억제에도 기여한다.

### 4.5 평가 파이프라인 구축 [M2, P1]

`scripts/verify_inference.py`를 확장하거나 별도 스크립트를 추가하여:
1. 데이터셋 디렉터리(클래스별 WAV 폴더 구조)를 입력받는다
2. 각 파일에 대해 파이프라인을 실행하고 점수를 기록한다
3. 정답 라벨과 비교하여 클래스별 Precision/Recall/F1을 계산한다
4. FAR(시간당 false alarm 횟수) 측정을 위한 negative set 처리를 포함한다

이 스크립트가 있어야 M2에서 NS 효과를 정량 비교할 수 있고, M3 헤드 학습의 개선 효과도 측정 가능하다.

---

## 5. 장기 / 오픈 이슈

### 5.1 M3: YAMNet 임베딩 헤드 파인튜닝

현재 M1은 YAMNet의 `scores` 출력(521 클래스 확률)에서 12종 인덱스만 추출한다. M3에서는 `embeddings` 출력(1024차원)을 입력으로 받는 경량 헤드(Dense 256→12, sigmoid)를 FSD50K + ESC-50 데이터로 학습한다. 이 전환은 `src/model/yamnet_wrapper.py`의 `infer_mean_scores()` 대신 `infer()`의 embeddings를 사용하도록 변경하는 것을 의미한다.

**개방 질문**: 헤드 학습 시 class weight를 어떻게 설정할 것인가? gunshot/explosion은 학습 데이터가 희귀하므로 over-sampling 또는 class weight boost가 필요하다.

### 5.2 한국어/한국 환경 데이터 수집

YAMNet 사전학습 데이터에 한국 환경 소리가 충분히 포함되어 있지 않다는 가설(2.5항 H1, H2)을 검증하고, 필요 시 현장 녹음을 수행한다. `development-plan.md` §3.2는 "자체 수집 현장 녹음 각 위험 클래스당 최소 30 샘플"을 권장하고 있다.

### 5.3 M4: TFLite 변환 및 on-device 성능 검증

라즈베리파이 4에서의 실시간 추론 가능 여부는 아직 미검증이다. TFLite int8 양자화 후 모델 크기와 추론 시간이 `development-plan.md` §1.2 KPI(모델 ≤ 5MB, RAM ≤ 50MB, CPU ≤ 50%)를 만족하는지 확인이 필요하다. YAMNet 자체가 약 13MB이므로 헤드만 TFLite로 변환할 경우와 전체 파이프라인을 변환할 경우의 차이를 검토해야 한다.

### 5.4 다중 위험음 동시 발생 처리

현재 multi-label 설계(`trigger.py:58`의 for loop)로 복수 클래스가 동시에 트리거될 수 있다. 임베디드 모듈에서 다중 이벤트를 받았을 때의 처리 방침(우선순위, 배치 전송, 첫 번째만 전송 등)이 확정되지 않았다. UART 페이로드 설계(`development-plan.md` §7.3)에 이미 단일 클래스 포맷만 정의되어 있으므로, 다중 동시 이벤트 시 어떻게 처리할지 M5 전에 결정해야 한다.

### 5.5 오픈 이슈: 임베디드 타겟 보드 미확정

`development-plan.md` §7.4는 라즈베리파이 4를 SBC 기준으로 가정하고 있으나 확정되지 않았다. 타겟 보드에 따라 M4(TFLite 양자화 수준), M5(UART vs BLE 선택)가 달라진다. M2 시작 전에 팀 내부 결정이 필요하다.

---

## 6. 다음 액션 체크리스트

### 즉시 (M2 착수 전, P1)

- [ ] **[검증]** ESC-50 또는 FSD50K에서 위험 클래스별 WAV 최소 3개씩 수집 → `data/sample/` 배치
- [ ] **[검증]** `--threshold 0.0 --verbose`로 각 WAV 파일의 실제 YAMNet 점수 기록 및 정리
- [ ] **[검증]** `yamnet_class_map.csv` 직접 조회로 `config/whitelist.yaml`의 13개 인덱스 최종 확인 및 문서 정합성 수정
- [ ] **[기획]** M2에서 `noise_suppress.suppress()` 삽입 위치를 `cli.py` 기준으로 명시 (이 문서 §4.2 참조)
- [ ] **[기획]** 임베디드 타겟 보드 확정 (라즈베리파이 4 vs 기타) — 팀 내부 결정 필요

### M2 범위 (P2)

- [ ] **[구현]** WebRTC NS 전처리 통합 (`src/preprocess/noise_suppress.py`)
- [ ] **[구현]** K/N debounce 후처리 구현 (`src/postprocess/trigger.py`)
- [ ] **[구현]** 평가 스크립트 작성 (클래스별 F1, FAR 계산)
- [ ] **[구현]** 윈도우별 RMS/dBFS 로깅 추가
- [ ] **[검증]** NS on/off A/B 비교 실험 수행 — 클래스별 Recall 변화 측정
- [ ] **[설계]** 클래스별 임계값 캘리브레이션 수행 후 `config/whitelist.yaml` 갱신

### M3 범위 (P3)

- [ ] **[구현]** YAMNet 임베딩 헤드 학습 파이프라인 (FSD50K + ESC-50)
- [ ] **[검증]** 한국 환경 현장 녹음 수집 (각 클래스 30샘플 이상)
- [ ] **[검증]** M1 베이스라인 vs M2(+NS) vs M3(+헤드) 3단계 비교 평가

---

## 부록: 이슈 우선순위 요약표

| 이슈 ID | 항목 | 영향도 | 우선순위 | 담당 마일스톤 | 현재 차단 여부 |
|---|---|---|---|---|---|
| I-01 | 평가 데이터셋 및 검증 미수행 | 상 | P1 | M2 전 | M1 Exit Criteria 미달 |
| I-02 | 임계값 0.4 근거 부족 / 클래스별 분포 미측정 | 상 | P1 | M2 전 | M2 A/B 비교 기준 없음 |
| I-03 | 인덱스 문서 불일치 (development-plan vs m1-spec) | 중 | P1 | M2 전 | 잠재적 버그 |
| I-04 | K/N debounce 미구현 | 상 | P2 | M2 | false positive 위험 |
| I-05 | 노이즈 캔슬링 파이프라인 연결 미완 | 중 | P2 | M2 | 연결 포인트 설계 필요 |
| I-06 | 입력 신호 레벨 모니터링 없음 | 중 | P2 | M2 | 디버깅 어려움 |
| I-07 | YAMNet 한국 환경 데이터 편향 가설 | 중 | P2 | M3 | 가설 수준 |
| I-08 | 추론 latency 미측정 | 하 | P3 | M2 말 | M1 Exit Criteria 미달 |
| I-09 | 임베디드 타겟 보드 미확정 | 하~중 | P3 | M4 전 | M4/M5 설계 영향 |
| I-10 | 다중 이벤트 동시 발생 처리 정책 미정 | 하 | P3 | M5 전 | UART 페이로드 설계 영향 |

---

*문서 버전: v0.1 (2026-05-11). M2 완료 후 §3~§4 항목의 결과를 반영하여 v0.2로 갱신 예정.*
