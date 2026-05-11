# 기초 지식 가이드: YAMNet 기반 위험 소리 감지 시스템

> 대상 독자: 신규 팀원, 프로젝트 학습자
> 목적: 이 문서를 위에서부터 순서대로 읽으면 프로젝트 전체 흐름이 잡히도록 구성
> 참조: `CLAUDE.md`, `docs/development-plan.md`, `docs/embedded-architecture-analysis.md`

---

## 목차

1. [오디오 신호 처리 기초](#1-오디오-신호-처리-기초)
2. [딥러닝 / 오디오 분류 기초](#2-딥러닝--오디오-분류-기초)
3. [YAMNet 모델](#3-yamnet-모델)
4. [후처리 알고리즘](#4-후처리-알고리즘)
5. [노이즈 캔슬링 기초](#5-노이즈-캔슬링-기초)
6. [임베디드 / 엣지 컴퓨팅 기초](#6-임베디드--엣지-컴퓨팅-기초)
7. [통신 프로토콜 기초](#7-통신-프로토콜-기초)
8. [Python / 개발 환경](#8-python--개발-환경)
9. [학습 추천 경로](#9-학습-추천-경로)
10. [용어 사전 (Glossary)](#10-용어-사전-glossary)

---

## 1. 오디오 신호 처리 기초

### 1.1 샘플링 레이트, 비트 깊이, 채널

소리는 연속적인 공기 압력 변화다. 컴퓨터에서 다루려면 이 연속 신호를 일정 간격으로 측정해 숫자로 저장해야 한다.

| 개념 | 설명 | 본 프로젝트 설정 |
|---|---|---|
| 샘플링 레이트 | 초당 측정 횟수 (Hz) | **16,000 Hz (16kHz)** |
| 비트 깊이 | 각 측정값의 정밀도 | **16bit (정수 -32768 ~ 32767)** |
| 채널 | 동시 수음 트랙 수 | **Mono (1채널)** |

**왜 16kHz mono인가?**
- YAMNet의 입력 사양이 16kHz mono로 고정되어 있다.
- 인간 음성과 대부분의 환경음은 8kHz 이하에 에너지가 집중된다. 나이퀴스트 정리에 따라 최대 표현 가능 주파수는 샘플링 레이트의 절반(8kHz)이므로 16kHz로 충분하다.
- Stereo는 채널이 2개라 데이터 양이 2배가 되지만 방향 추정이 불필요한 이 프로젝트에서는 불필요하다.

> 프로젝트 매핑: `src/audio_io/mic_stream.py`에서 sounddevice를 통해 16kHz/16bit/mono로 마이크 입력을 캡처한다.

---

### 1.2 파형(Waveform)과 PCM

**PCM(Pulse Code Modulation)**: 샘플링된 오디오를 디지털 숫자 배열로 저장하는 가장 기본적인 방식이다. WAV 파일의 내부 포맷이 바로 PCM이다.

```
시간축 →
[312, 1045, 2318, 1876, 243, -512, -1890, -2231, -1023, 412, ...]
 ↑
 각 숫자 하나가 "샘플" (16bit 정수, -32768~32767)
```

- 16kHz 오디오 1초 = 16,000개의 숫자
- 0.96초 = 15,360개의 숫자 (YAMNet 한 윈도우 분량)

> 프로젝트 매핑: `src/audio_io/network_stream.py`(M5)에서 ESP32가 전송한 PCM raw bytes를 numpy 배열로 변환한다.

---

### 1.3 윈도우, Hop, 오버랩

연속으로 들어오는 오디오를 조각 단위로 분석하는 방식이다.

```
전체 오디오 스트림 (시간 →)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

윈도우 1: [████████████████]  ← 0.96초
윈도우 2:         [████████████████]  ← 0.96초
윈도우 3:                 [████████████████]  ← 0.96초
                  ↑
                  hop = 0.48초 (윈도우 간격)
                  오버랩 = 0.96 - 0.48 = 0.48초
```

| 파라미터 | 값 | 이유 |
|---|---|---|
| 윈도우 크기 | **0.96초** | YAMNet이 요구하는 최소 입력 길이 |
| Hop 크기 | **0.48초** | 윈도우 50% 오버랩으로 소리 경계를 놓치지 않음 |

오버랩이 있으면 두 윈도우 경계에 걸친 소리도 어느 한 윈도우에서 완전하게 포착된다. 총성처럼 짧은 소리도 놓치지 않는 이유다.

> 프로젝트 매핑: `src/audio_io/mic_stream.py`의 링 버퍼가 0.96초 분량을 유지하고 0.48초마다 YAMNet에 넘긴다.

---

### 1.4 멜 스펙트로그램 (Mel Spectrogram)

파형(시간 영역)만으로는 "어떤 주파수 성분이 언제 나타났는가"를 보기 어렵다. 스펙트로그램은 이를 시각화한다.

```
주파수 ↑   [멜 스펙트로그램]
(mel 스케일)
고주파    ░░░░██░░░░░░░░░░░
          ░░████░░░░░░░░░░
          ░███████░░░░░░░░
저주파     ████████████░░░
           ──────────────→ 시간
```

- **스펙트로그램**: 짧은 구간마다 FFT를 적용해 주파수별 에너지를 시간축과 함께 표현한 2D 이미지
- **멜 스케일**: 인간 청각이 저주파에 민감하고 고주파에 둔감한 특성을 반영해 주파수 축을 비선형으로 변환한 것

YAMNet은 내부적으로 입력 파형을 64개 멜 필터뱅크 기반 로그 멜 스펙트로그램으로 변환한 후 CNN에 입력한다. 이 과정은 자동으로 처리되므로 사용자가 직접 구현하지 않아도 된다.

> 프로젝트 매핑: YAMNet 내부에서 자동 처리. `yamnet_wrapper.py`에서 raw 파형을 넘기면 된다.

---

### 1.5 위험 소리의 음향적 특성

위험 소리는 크게 두 유형으로 나뉜다.

| 유형 | 설명 | 예시 | 특징 |
|---|---|---|---|
| **Transient (단발성)** | 매우 짧고 에너지가 순간 집중 | 총소리, 유리 깨짐, 폭발음 | 수십~수백 ms 이내, 넓은 주파수 대역 |
| **Steady-state (지속성)** | 시간이 지나도 일정한 패턴 반복 | 화재 경보, 사이렌, 아기 울음 | 수 초 이상, 특정 주파수 패턴 반복 |

이 구분이 중요한 이유: 노이즈 캔슬링(NS) 기법이 transient 소리의 에너지를 노이즈로 오인해 제거할 위험이 있다. 본 프로젝트에서 NS aggressiveness를 낮게 설정하는 이유다.

---

## 2. 딥러닝 / 오디오 분류 기초

### 2.1 다중 분류 vs 다중 레이블

| 비교 항목 | 다중 분류 (multi-class) | 다중 레이블 (multi-label) |
|---|---|---|
| 가정 | 클래스 중 하나만 정답 | 동시에 여러 클래스가 정답 가능 |
| 출력 함수 | Softmax (합=1) | Sigmoid (각 독립) |
| 예시 | "이 소리는 총소리다" | "이 소리는 비명+유리깨짐이 동시에 들린다" |

**본 프로젝트가 다중 레이블인 이유**: 실제 환경에서는 화재 경보음이 울리면서 동시에 비명이 들리는 상황이 있다. 단일 라벨 argmax는 이를 하나의 클래스로만 출력하므로 멀티-이벤트 감지가 불가능하다.

> 프로젝트 매핑: `model/danger_filter.py`가 12개 위험 클래스 각각에 독립 sigmoid를 적용해 multi-label 출력을 만든다.

---

### 2.2 Sigmoid vs Softmax

```
입력: [2.1, 0.5, -1.2, 3.0]  (4개 클래스의 로짓)

Softmax → [0.24, 0.05, 0.01, 0.70]  합 = 1.00  (경쟁적)
Sigmoid → [0.89, 0.62, 0.23, 0.95]  합 ≠ 1    (독립적)
```

- Softmax: 모든 클래스 확률의 합이 1이 되도록 정규화. 한 클래스 확률이 높아지면 나머지가 낮아진다.
- Sigmoid: 각 클래스를 0~1 사이 독립 확률로 변환. 동시에 여러 클래스가 높은 확률을 가질 수 있다.

> 프로젝트 매핑: `config/whitelist.yaml`에 정의된 12종 클래스 각각에 sigmoid 임계값을 독립 적용한다.

---

### 2.3 임계값(Threshold)과 F1 / Precision / Recall

분류 모델은 확률을 출력한다. 확률이 어느 값 이상이면 "위험 감지"로 결정하는 기준이 임계값이다.

```
Precision = TP / (TP + FP)   (감지한 것 중 실제 위험인 비율)
Recall    = TP / (TP + FN)   (실제 위험 중 감지한 비율)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

- 임계값을 낮추면: Recall 상승, Precision 하락 (더 민감하게 잡지만 오탐 증가)
- 임계값을 높이면: Precision 상승, Recall 하락 (정확하지만 놓치는 경우 증가)

**본 프로젝트 방침**: 위험 소리를 놓치는 것(FN)이 오탐(FP)보다 훨씬 심각하므로 Recall ≥ 0.85를 우선 목표로 삼는다.

> 프로젝트 매핑: 현재 기본 임계값은 0.4. `--threshold` CLI 옵션으로 조정 가능하다.

---

### 2.4 사전학습 모델과 전이학습, Backbone Freeze

```
[전이학습 개념도]

사전학습 모델 (예: YAMNet, AudioSet 521클래스로 학습)
    │
    ├── Backbone (특징 추출기, 동결 freeze)
    │     이미 학습된 1024차원 음향 특징 추출 능력 보존
    │
    └── Head (분류기, 새로 학습)
          우리 12개 위험 클래스에 맞게 학습
          Dense(256, ReLU) → Dropout(0.3) → Dense(12, Sigmoid)
```

- **Backbone freeze**: 사전학습된 가중치를 그대로 고정해 업데이트하지 않는다. 적은 데이터로도 안정적인 학습이 가능하고 연산량도 절약된다.
- **전이학습(Transfer Learning)**: 대량 데이터로 학습한 모델의 지식을 새 작업에 재활용하는 방법론.

> 프로젝트 매핑: M1~M2에서는 YAMNet을 그대로 사용(임계값 튜닝만). M3에서 경량 헤드를 추가 학습한다.

---

### 2.5 임베딩(Embedding) 벡터

딥러닝 모델의 중간 레이어 출력을 임베딩이라 한다. 원시 입력(파형)보다 훨씬 압축된 형태로 의미 있는 특징만 남아 있다.

YAMNet의 경우 0.96초 파형 → 내부 처리 → **1024차원 실수 벡터** 출력. 이 1024개 숫자가 해당 소리의 "음향 지문"이다. 비슷한 소리는 임베딩 공간에서 가깝게 위치한다.

> 프로젝트 매핑: M3에서 이 1024-d 임베딩 위에 Dense 레이어 헤드를 올려 12클래스 분류기를 학습한다.

---

## 3. YAMNet 모델

### 3.1 YAMNet이란

- **정식 명칭**: Yet Another Multiclass Network
- **기반 아키텍처**: MobileNetV1 (경량 depthwise separable convolution)
- **학습 데이터**: AudioSet (YouTube 클립 200만+, 521개 사운드 클래스)
- **목적**: 범용 환경음 분류
- **공개**: Google, Apache-2.0 라이선스

참조: [Google AI Blog - AudioSet](https://ai.googleblog.com/2017/03/announcing-audioset-dataset-for-audio.html)

---

### 3.2 입력/출력 사양

```
입력: float32 waveform, shape=(N,)
      샘플링 레이트 = 16,000 Hz
      최소 길이 = 0.96초 (15,360 샘플)

출력:
  scores:     (num_patches, 521)  각 패치별 521클래스 확률
  embeddings: (num_patches, 1024) 각 패치별 음향 임베딩
  spectrogram: (num_frames, 64)   로그 멜 스펙트로그램
```

> 프로젝트 매핑: `model/yamnet_wrapper.py`에서 scores를 받아 danger_filter로 넘긴다.

---

### 3.3 num_patches가 여러 개인 이유

YAMNet은 내부적으로 입력을 **0.48초 단위의 패치(patch)**로 쪼개서 처리한다.

```
0.96초 입력 →  패치 1 (0.00~0.48초)  → scores[0] (521,)
               패치 2 (0.48~0.96초)  → scores[1] (521,)

num_patches = 2  (0.96초 입력 기준)
```

여러 패치의 scores를 평균(mean)하면 0.96초 전체를 대표하는 단일 (521,) 벡터가 된다. 본 프로젝트에서는 이 평균 scores를 사용한다.

---

### 3.4 YAMNet class_map과 화이트리스트 12종

YAMNet은 521개 클래스 전체를 분류하지만, 본 프로젝트는 그 중 위험 소리 12종만 관심 대상으로 선별한다.

| 역할 | 설명 |
|---|---|
| `yamnet_class_map.csv` | 인덱스 0~520과 AudioSet 클래스명의 매핑 테이블 |
| `config/whitelist.yaml` | 프로젝트에서 사용할 위험 클래스 인덱스 13개 (12종 의미, glass+shatter 통합) |

**화이트리스트 인덱스 목록** (CLAUDE.md 기준):

| 인덱스 | 클래스 | 비고 |
|---|---|---|
| 11 | Screaming | 비명 |
| 20 | Baby cry, infant cry | 영아 울음 |
| 302 | Gunshot, gunfire | 총소리 |
| 304 | Explosion | 폭발음 |
| 390 | Siren | 사이렌 (상위) |
| 391 | Civil defense siren | 민방위 사이렌 |
| 393 | Ambulance (siren) | 구급차 사이렌 |
| 394 | Fire engine, fire truck (siren) | 소방차 사이렌 |
| 420 | Fire alarm | 화재 경보 |
| 421 | Smoke detector, smoke alarm | 연기 감지기 |
| 435 | Glass | 유리음 |
| 437 | Shatter | 파손음 |
| 464 | Vehicle horn, car horn, honking | 차량 경적 |

glass(435) + shatter(437)는 `max()`로 통합해 `glass_shatter` 단일 이벤트로 처리한다.

> 주의: 위 인덱스는 `yamnet_class_map.csv` 직접 조회로 최종 확인이 필요하다. (CLAUDE.md M2 전 P1 항목)

---

### 3.5 TF-Hub에서 모델을 받아 쓰는 방식

TF-Hub(TensorFlow Hub)는 사전학습 모델을 URL로 배포하는 플랫폼이다. 코드 첫 실행 시 모델 파일을 자동 다운로드해 로컬 캐시에 저장한다.

```
URL: https://tfhub.dev/google/yamnet/1
캐시 위치: 환경변수 TFHUB_CACHE_DIR (미설정 시 임시 디렉터리)
최초 실행: 모델 다운로드 (~26MB)
이후 실행: 캐시에서 즉시 로드
```

> 프로젝트 매핑: `scripts/verify_inference.py`를 처음 실행하면 다운로드가 발생한다. 인터넷 연결 필요.

---

## 4. 후처리 알고리즘

### 4.1 단순 임계값 비교의 한계

YAMNet은 0.96초마다 한 번씩 scores를 출력한다. 단순히 "score ≥ 0.4이면 위험"으로 판정하면 아래 문제가 생긴다.

- **한 번 튀는 score**: TV에서 총소리 효과음이 0.5초만 나왔는데 즉시 알림 발생
- **연속 트리거**: 사이렌이 지속되면 매 0.48초마다 알림이 쏟아짐

---

### 4.2 Debounce K/N 다수결

슬라이딩 윈도우 N개 중 K번 이상 임계값을 넘어야 실제 트리거로 인정하는 방식이다.

```
N=3, K=2 설정 예시:

시각      score  임계값 통과  votes 창
0.0초     0.12   아니오      [0]
0.48초    0.65   예          [0, 1]
0.96초    0.72   예          [0, 1, 1]  → sum=2 ≥ K=2 → 트리거!
1.44초    0.08   아니오      [1, 1, 0]  → sum=2 ≥ K=2 → (cooldown 중이라 무시)
1.92초    0.05   아니오      [1, 0, 0]  → sum=1 < K=2 → 미트리거
```

**왜 false positive를 줄이는가?** 단발성 노이즈로 인해 score가 한 번 튀더라도 N개의 창 중 K개 이상에서 연속적으로 나타나지 않으면 트리거가 발생하지 않는다.

> 프로젝트 매핑: `postprocess/trigger.py`의 `DebounceState` (deque maxlen=N). `config/whitelist.yaml`에서 `debounce: {window: 3, k: 2}`로 설정.

---

### 4.3 Cooldown — 연속 트리거 억제

트리거 직후 일정 시간(5초) 동안 같은 클래스의 트리거를 억제한다.

```
사이렌 소리가 30초간 지속된다고 가정:

debounce 통과 → 트리거 발생 → cooldown 5초 시작
  (5초 동안 동일 클래스 트리거 무시)
cooldown 종료 → debounce 재평가 → 트리거 발생 → cooldown 5초 시작
  ...
```

alrm이 계속 울리더라도 5초에 한 번만 임베디드로 알림을 보낸다. 진동 모터가 쉬지 않고 울리는 상황을 방지한다.

**처리 순서**: debounce 통과 → cooldown 확인 → 알림 발송

> 프로젝트 매핑: `postprocess/trigger.py`에서 클래스별 cooldown 타임스탬프 관리.

---

### 4.4 슬라이딩 윈도우 deque가 효율적인 이유

N=3 슬라이딩 윈도우를 일반 리스트로 구현하면 매번 앞 원소를 삭제하는 연산이 O(N)이다. Python의 `collections.deque(maxlen=N)`을 사용하면 자동으로 오래된 원소가 밀려나고 O(1) 삽입이 가능하다.

| 구조 | 삽입 | 오래된 원소 제거 | 구현 편의성 |
|---|---|---|---|
| list | O(1) append + O(N) pop | 수동 | 복잡 |
| deque(maxlen=N) | O(1) | 자동 | 단순 |

> 프로젝트 매핑: `trigger.py`의 `DebounceState`가 클래스별로 deque를 보유한다.

---

## 5. 노이즈 캔슬링 기초

### 5.1 노이즈 vs 신호, SNR 개념

```
SNR (신호 대 잡음비) = 10 × log10(신호 전력 / 노이즈 전력)  [dB]

SNR이 높을수록 신호가 노이즈보다 우세.
SNR = 0dB  → 신호 = 노이즈 (반반)
SNR = 20dB → 신호가 노이즈보다 100배 강함
```

실제 실외 환경에서는 SNR이 -5 ~ +10dB 범위일 수 있어 위험 소리가 배경음에 묻힌다. 노이즈 캔슬링은 SNR을 높여 모델이 위험 소리를 더 잘 인식하게 돕는다.

---

### 5.2 WebRTC NS 방식 개요

WebRTC NS(Noise Suppression)는 Google이 WebRTC 프로젝트에서 실시간 통화 품질 향상을 위해 개발한 노이즈 억제 라이브러리다.

**동작 원리 (스펙트럴 게이팅 직관)**:

```
1. 짧은 시간 구간(프레임)마다 스펙트럼 분석
2. 과거 프레임 통계로 "배경 노이즈 프로파일" 추정
3. 현재 프레임에서 노이즈 프로파일보다 약한 주파수 성분을 억제(게이팅)
4. 신호가 충분히 강한 성분만 통과
```

- 장점: 검증된 오픈소스, 실시간 처리, 낮은 연산량, 라즈베리파이/Jetson에서 동작
- 단점: 단발성 위험음(총소리, 유리 파손)을 노이즈로 오인할 위험

---

### 5.3 노이즈 캔슬링이 위험 소리에 미칠 수 있는 부작용

| 위험 소리 유형 | NS 영향 | 대응 |
|---|---|---|
| 총소리 (transient, 넓은 대역) | 에너지 감소, 순간 억제될 가능성 | NS aggressiveness=1~2 (약하게) |
| 유리 파손 (transient) | 동일 | 동일 |
| 화재 경보 (steady, 패턴 반복) | 배경음으로 학습될 위험 (VAD 없을 시) | 짧은 노이즈 추정 구간 사용 |
| 사이렌 (steady) | 지속되면 노이즈 프로파일에 포함될 위험 | aggressiveness 낮게, 학습 시 noise mix |

**본 프로젝트 전략**: NS를 약하게(aggressiveness=1) 적용하고, 모델 학습 시 노이즈 mix-in + SpecAugment로 모델 자체의 내성을 높인다.

> 프로젝트 매핑: `preprocess/noise_suppress.py` (M2-NS PR에서 구현 예정).

---

## 6. 임베디드 / 엣지 컴퓨팅 기초

### 6.1 ESP32와 Jetson Nano의 역할 차이

| 항목 | ESP32 | Jetson Nano |
|---|---|---|
| 분류 | MCU (마이크로컨트롤러) | SBC (싱글보드컴퓨터) |
| CPU | 240MHz 듀얼코어 Xtensa | Cortex-A57 1.43GHz 4코어 |
| RAM | 520KB SRAM | 4GB LPDDR4 |
| GPU | 없음 | 128-core Maxwell |
| OS | 없음 (펌웨어) | Linux (Ubuntu) |
| 역할 | 마이크 캡처 + Wi-Fi 전송 + 진동 출력 | YAMNet 추론 + 판정 + 결과 전송 |
| 딥러닝 추론 | 불가 (메모리/연산 부족) | 가능 (TF, TFLite) |

**왜 본 프로젝트는 Jetson에서 추론하는가?** YAMNet의 MobileNetV1 backbone은 약 3.7MFLOPS/윈도우를 요구한다. ESP32의 연산 능력으로는 실시간 처리가 불가능하다. Jetson의 GPU 또는 CPU에서 처리 후 판정 결과만 ESP32로 전달하는 구조가 현실적이다.

---

### 6.2 I2S 마이크란, MEMS vs 콘덴서

**I2S(Inter-IC Sound)**: 마이크, DAC, ADC 같은 오디오 칩 간 통신을 위한 디지털 직렬 버스 프로토콜이다. 아날로그 노이즈 없이 디지털 PCM 데이터를 직접 받을 수 있다.

| 비교 항목 | MEMS 마이크 (디지털 I2S) | 콘덴서 마이크 (아날로그) |
|---|---|---|
| 출력 | 디지털 PDM/I2S | 아날로그 전압 |
| ADC 필요 | 불필요 (내장) | 별도 ADC 필요 |
| 노이즈 | 낮음 | ADC 단에서 노이즈 유입 가능 |
| 크기 | 초소형 | 상대적으로 큼 |
| 권장 제품 | INMP441, SPH0645 | — |

**본 프로젝트 권장**: INMP441 (I2S MEMS). ESP32의 I2S 주변장치에 직결 가능하고, DMA로 CPU 부담 없이 데이터를 받을 수 있다.

---

### 6.3 TFLite 양자화(Quantization) — INT8과 정확도 trade-off

**양자화**: 모델 가중치와 활성화값을 float32(32비트)에서 int8(8비트)로 변환해 모델 크기와 연산량을 줄이는 기술이다.

```
float32 가중치 → INT8 가중치
크기: 1/4로 감소 (32bit → 8bit)
연산: 정수 연산으로 가속 (특히 ARM 프로세서)
정확도: 소폭 하락 (목표: 2%p 이내 손실)
```

| 구분 | float32 | INT8 |
|---|---|---|
| 모델 크기 | ~26MB | ~6.5MB |
| 추론 속도 | 기준 | 2~4배 빠름 (ARM) |
| 정확도 손실 | 기준 | 1~3%p |
| 배포 대상 | 서버/PC | 임베디드 SBC |

> 프로젝트 매핑: M4 마일스톤에서 TFLite 변환 및 INT8 양자화 실험. 목표: 모델 ≤ 5MB, 정확도 손실 ≤ 2%p.

---

### 6.4 엣지 추론 vs 게이트웨이 추론

```
[엣지 추론]                    [게이트웨이 추론]
마이크 → MCU/SBC              마이크 → ESP32 → Wi-Fi → Jetson(게이트웨이)
          ↓                                              ↓
       YAMNet 추론                                  YAMNet 추론
          ↓                                              ↓
       진동 모터                              Wi-Fi → ESP32 → 진동 모터
```

본 프로젝트는 **게이트웨이 추론** 방식이다. ESP32가 추론 능력이 부족하므로 Jetson이 게이트웨이 역할을 맡는다. 장점은 ESP32 부담이 낮고 모델 업데이트가 Jetson 측에서만 이루어진다는 점이다. 단점은 Wi-Fi 의존성과 게이트웨이 단일 장애점이다.

---

## 7. 통신 프로토콜 기초

### 7.1 TCP vs UDP — 위험 소리 도메인 관점

| 항목 | TCP | UDP |
|---|---|---|
| 전송 보장 | 보장 (재전송) | 비보장 (손실 허용) |
| 순서 보장 | 보장 | 비보장 |
| 지연 | 중간 (재전송 지연 있음) | 낮음 |
| 적합한 용도 | 판정 결과 전달 (중요, 소량) | 오디오 스트리밍 (실시간, 손실 허용) |

**본 프로젝트 결정**:
- ESP32 → Jetson 오디오 스트림: **UDP** (실시간성 우선, 소량 손실은 추론에 큰 영향 없음)
- Jetson → ESP32 판정 결과: **TCP** (유실 없이 전달 보장 필요)

---

### 7.2 Wi-Fi 패킷 손실과 오디오 스트리밍

16kHz/16bit mono PCM의 대역폭은 약 256kbps다. Wi-Fi 2.4GHz에서는 충분하지만 패킷 손실이 발생할 수 있다. 0.48초 청크(약 15KB) 하나가 손실되면 해당 구간 추론이 불가능하다. 이때 zero-fill(0으로 채움)하거나 해당 창을 skip하는 방식으로 처리한다.

손실이 잦은 환경에서는 청크를 더 작게 쪼개거나 5GHz Wi-Fi를 사용하는 것을 권장한다.

---

### 7.3 UART/Serial, BLE, MQTT 비교

| 프로토콜 | 거리 | 지연 | 구현 복잡도 | 특징 |
|---|---|---|---|---|
| UART/Serial | 수 cm~수 m (유선) | 매우 낮음 | 매우 낮음 | 단순, 안정. MCU↔SBC 직결 |
| BLE | ~10m (무선) | 낮음 | 높음 | 저전력. 256kbps 지속 스트리밍에는 빠듯 |
| MQTT | 제한 없음 (인터넷) | 중간 | 높음 | 브로커 서버 필요. 다중 구독자 지원. 오디오 스트리밍에는 부적합 |
| Wi-Fi (TCP/UDP) | LAN 범위 | 낮음~중간 | 낮음 | 본 프로젝트 선택 |

**UART의 위치**: 본 프로젝트는 Wi-Fi를 주 경로로 채택했으나, `uart_sender.py`를 폴백(유선 백업)으로 유지한다. M5-a에서 직렬 연결 PoC로 전체 파이프라인을 먼저 검증하는 데도 활용한다.

---

### 7.4 JSON 라인 프로토콜

**JSON Lines(JSONL)**: 각 줄이 유효한 JSON 객체 하나인 텍스트 포맷이다. `\n`으로 메시지 경계를 구분한다.

```
{"event":"danger","ts":1746873600.1,"class":"glass_shatter","score":0.87,"seq":142}
{"event":"heartbeat","ts":1746873605.0,"seq":143}
{"event":"danger","ts":1746873610.3,"class":"screaming","score":0.91,"seq":144}
```

- 사람이 읽기 쉽다 (디버깅 용이)
- 스트리밍 환경에서 줄 단위 파싱이 단순하다
- 파싱 오류 시 해당 줄만 skip하면 되므로 견고하다

> 프로젝트 매핑: `--log output/run.jsonl`로 로컬 로그 저장. `embedded/wifi_sender.py`에서 ESP32로 전송하는 포맷.

---

## 8. Python / 개발 환경

### 8.1 가상환경(venv)과 의존성 관리

**가상환경**: 프로젝트별로 Python 패키지를 격리하는 방법이다. 시스템 Python에 영향 없이 프로젝트별 버전을 독립 관리한다.

```
프로젝트 루트/
├── .venv/          ← 가상환경 (git에 포함 안 함)
├── requirements.txt ← 의존성 목록
└── src/
```

- `requirements.txt`: 필요한 패키지와 버전을 기록. `pip install -r requirements.txt`로 일괄 설치.
- `.venv`: 실제 패키지가 설치되는 디렉터리. `.gitignore`에 추가해 저장소에 포함하지 않는다.

> 이 프로젝트는 Python 3.11을 요구한다. `py -3.11 -m venv .venv`로 생성.

---

### 8.2 주요 라이브러리 역할

| 라이브러리 | 역할 | 본 프로젝트에서 쓰임 |
|---|---|---|
| `tensorflow` | 딥러닝 프레임워크 | YAMNet 모델 로드 및 추론 실행 |
| `tensorflow_hub` | TF-Hub 모델 다운로드/로드 | YAMNet을 URL로 로드 |
| `librosa` | 오디오 파일 읽기, 리샘플링 | WAV 파일 전처리, 16kHz 변환 |
| `sounddevice` | 마이크 실시간 캡처 | `--input mic` 모드에서 오디오 입력 |
| `numpy` | 수치 배열 연산 | 파형, scores, 임베딩 배열 처리 |
| `pytest` | 단위 테스트 프레임워크 | `tests/` 하위 테스트 실행 |
| `PyYAML` | YAML 파일 파싱 | `config/whitelist.yaml` 로드 |

---

### 8.3 pytest 단위 테스트 패턴 (mock, fixture)

**fixture**: 테스트 함수에 공통으로 필요한 준비 데이터나 객체를 제공하는 함수다. `@pytest.fixture` 데코레이터로 정의하고, 테스트 함수의 인자로 자동 주입된다.

**mock**: 실제 구현 대신 가짜 객체를 주입해 외부 의존성(네트워크, YAMNet 등) 없이 테스트하는 방법이다.

```
테스트 파일 예: tests/test_debounce_trigger.py

- YAMNet을 로드하지 않고 mock scores를 직접 주입
- Trigger 클래스의 debounce/cooldown 로직만 단독 테스트
- pytest 실행 시 네트워크 불필요
```

> 프로젝트 매핑: `pytest tests/ -v`로 네트워크 없이 실행 가능한 테스트만 돈다. YAMNet 로딩이 필요한 테스트는 `-k "yamnet"` 옵션으로 분리.

---

### 8.4 numpy 배열 shape 다루기 기본

오디오 처리에서 자주 등장하는 shape 패턴:

| 의미 | shape 예시 |
|---|---|
| 0.96초 파형 | `(15360,)` — 1D 배열 |
| YAMNet scores (2패치) | `(2, 521)` — 2D 배열 |
| 평균 scores | `(521,)` — 1D 배열 |
| 배치 추론 | `(B, 15360)` — B개 윈도우 묶음 |
| 임베딩 | `(2, 1024)` — 2패치 × 1024차원 |

shape 확인: `arr.shape`
특정 인덱스 추출: `scores[..., [302, 304, 390]]`  — 위험 클래스 인덱스 슬라이싱
패치 평균: `scores.mean(axis=0)` — (num_patches, 521) → (521,)

---

## 9. 학습 추천 경로

### 0주차: 오디오 신호 처리 기초

1. 샘플링 정리, 멜 스펙트로그램 개념 파악
2. librosa 튜토리얼로 WAV 파일 읽고 시각화 실습
3. `python scripts/verify_inference.py`로 프로젝트 실행 확인

**추천 자료**:
- [librosa 공식 문서](https://librosa.org/doc/latest/index.html)
- [Speech and Language Processing (Jurafsky) — Chapter 9](https://web.stanford.edu/~jurafsky/slp3/)

---

### 1~2주차: 딥러닝 / 오디오 분류

1. Sigmoid vs Softmax 차이 실습 (numpy로 직접 계산)
2. Precision / Recall / F1 개념 확인
3. Transfer learning 개요 파악

**추천 자료**:
- [TensorFlow Transfer Learning 공식 튜토리얼](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---

### 3주차: YAMNet 직접 돌려보기

1. `python -m src.cli --input data/sample/test.wav --threshold 0.0 --verbose`로 모든 클래스 score 확인
2. `yamnet_class_map.csv`를 열어 화이트리스트 인덱스와 클래스명 대조
3. 마이크 실시간 분석 (`--input mic`)로 실제 환경음 실험

**추천 자료**:
- [YAMNet TF-Hub 페이지](https://tfhub.dev/google/yamnet/1)
- [AudioSet 공식 사이트](https://research.google.com/audioset/)

---

### 4주차+: 임베디드 통합

1. `docs/embedded-architecture-analysis.md` 정독
2. ESP32 Arduino-ESP32로 I2S 마이크 기초 예제 실행
3. UDP 소켓 테스트 (PC에서 더미 PCM 전송, Jetson에서 수신)

**추천 자료**:
- [ESP-IDF Programming Guide — I2S](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/i2s.html)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

---

## 10. 용어 사전 (Glossary)

| 용어 / 약어 | 설명 |
|---|---|
| **YAMNet** | Google의 오디오 분류 신경망. MobileNetV1 기반, AudioSet 521클래스 사전학습 |
| **AudioSet** | Google이 YouTube 영상에서 구축한 대규모 오디오 데이터셋. 521개 계층적 클래스 |
| **MFCC** | Mel-Frequency Cepstral Coefficients. 멜 스펙트럼에서 추출한 음성 특징 계수. YAMNet은 직접 사용하지 않음 |
| **NS** | Noise Suppression(노이즈 억제). 본 프로젝트에서는 WebRTC NS를 지칭 |
| **VAD** | Voice Activity Detection. 음성 구간 자동 감지 기술 |
| **Debounce** | 일시적 신호 튀김을 무시하고 안정된 신호만 인식하는 기법. 본 프로젝트에서 K/N 다수결 방식 |
| **Hop** | 윈도우 분석 시 다음 윈도우로 이동하는 간격. 본 프로젝트에서 0.48초 |
| **Patch** | YAMNet이 내부적으로 오디오를 분할하는 단위. 0.48초 |
| **Cooldown** | 트리거 발생 후 동일 클래스의 중복 트리거를 억제하는 대기 시간. 본 프로젝트에서 5초 |
| **Sigmoid** | 입력값을 0~1 사이 확률로 변환하는 함수. 다중 레이블 분류에 사용 |
| **F1** | Precision과 Recall의 조화 평균. 불균형 클래스 평가에 적합 |
| **INT8** | 8비트 정수 자료형. TFLite 양자화에서 모델 크기/속도 최적화에 사용 |
| **I2S** | Inter-IC Sound. 오디오 칩 간 디지털 직렬 통신 프로토콜 |
| **MEMS** | Micro-Electro-Mechanical System. 반도체 공정으로 제조한 초소형 마이크 |
| **MQTT** | Message Queuing Telemetry Transport. 경량 발행/구독 메시지 프로토콜. IoT에 많이 쓰임 |
| **MCU** | Micro Controller Unit. 소형 단일칩 컴퓨터. 본 프로젝트에서 ESP32 |
| **SBC** | Single Board Computer. 단일 기판 컴퓨터. 본 프로젝트에서 Jetson Nano |
| **TFLite** | TensorFlow Lite. 모바일/임베디드 장치용 경량 추론 프레임워크 |
| **PCM** | Pulse Code Modulation. 오디오를 디지털 숫자 배열로 저장하는 기본 방식 |
| **DMA** | Direct Memory Access. CPU 개입 없이 주변장치와 메모리 간 데이터 전송을 처리하는 하드웨어 기능 |
| **FAR** | False Alarm Rate. 단위 시간당 오탐 발생 횟수. 목표: ≤ 1회/시간 |
| **SNR** | Signal-to-Noise Ratio. 신호 대 잡음비. 높을수록 신호가 선명 |
| **backbone** | 모델에서 특징을 추출하는 핵심 네트워크 부분. 본 프로젝트에서 YAMNet 전체 |
| **head** | backbone 위에 얹은 작은 분류기. 본 프로젝트에서 M3에 추가할 Dense 2층 |
| **whitelist** | 관심 클래스 목록. 본 프로젝트에서 위험 소리 12종 |
| **JSONL** | JSON Lines. 줄마다 JSON 객체 하나인 텍스트 형식. 스트리밍 로그에 사용 |
| **embedding** | 모델 중간 레이어의 압축된 특징 벡터. YAMNet에서 1024차원 |

---

*문서 버전: v0.1 (2026-05-11). 프로젝트 진행에 따라 갱신 필요.*
*참조: `CLAUDE.md`, `docs/development-plan.md`, `docs/embedded-architecture-analysis.md`*
