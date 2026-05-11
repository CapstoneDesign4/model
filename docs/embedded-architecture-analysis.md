# 임베디드 통합 아키텍처 분석

> 버전: v0.1 (2026-05-11)
> 참조: `CLAUDE.md`, `docs/development-plan.md`, `docs/m1-initial-model-spec.md`, `docs/m2-debounce-spec.md`
> 대상 독자: 기획 에이전트, 모델 개발 에이전트, 임베디드 담당

---

## 1. 시스템 구성도

### 1.1 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                          현장 환경 (실환경)                           │
│                                                                     │
│  ┌──────────────────────────────────────┐                           │
│  │            ESP32 (엣지 노드)          │                           │
│  │                                      │                           │
│  │  [I2S MEMS 마이크]                    │                           │
│  │       │ 16kHz/16bit PCM              │                           │
│  │       ▼                              │                           │
│  │  [DMA 링 버퍼 (0.96s 분량)]           │                           │
│  │       │                              │                           │
│  │       ▼  ── (오디오 청크 업스트림) ──►│── Wi-Fi UDP/TCP ──►       │
│  │  [Wi-Fi 송신부]                       │                           │
│  │                                      │                           │
│  │  ◄── (판정 결과 수신) ──             │◄── Wi-Fi UDP/TCP ──        │
│  │       │                              │                           │
│  │       ▼                              │                           │
│  │  [진동 모터 드라이버]                 │                           │
│  │  [LED/부저 (선택)]                    │                           │
│  └──────────────────────────────────────┘                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Jetson Nano (게이트웨이)                    │   │
│  │                                                              │   │
│  │  ◄── Wi-Fi 수신 ──  오디오 청크 수신 (PCM 32KB/s)            │   │
│  │       │                                                      │   │
│  │       ▼                                                      │   │
│  │  [network_stream.py]  ← 신규 추가 필요                        │   │
│  │       │ float32[15360]                                       │   │
│  │       ▼                                                      │   │
│  │  [preprocess/noise_suppress.py]  (WebRTC NS, M2-NS PR)       │   │
│  │       │                                                      │   │
│  │       ▼                                                      │   │
│  │  [model/yamnet_wrapper.py]  YAMNet 추론 (521-class)          │   │
│  │       │ mean scores (521,)                                   │   │
│  │       ▼                                                      │   │
│  │  [model/danger_filter.py]  화이트리스트 12종 추출              │   │
│  │       │ dict[str, float]                                     │   │
│  │       ▼                                                      │   │
│  │  [postprocess/trigger.py]  Debounce K/N + Cooldown           │   │
│  │       │ list[TriggerEvent]                                   │   │
│  │       ▼                                                      │   │
│  │  [embedded/wifi_sender.py]  판정 결과 JSON 송신  ← 신규 추가  │   │
│  │       │                                                      │   │
│  │  ──► Wi-Fi 송신 ──  ESP32로 판정 결과 전달                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 통신 채널 요약

```
[방향]      [채널]                [내용]                [대역폭/지연]
ESP32 → Jetson   Wi-Fi (UDP or TCP)   PCM 오디오 청크       ~256kbps 상시
Jetson → ESP32   Wi-Fi (UDP or TCP)   판정 결과 JSON        < 1KB/이벤트, 비주기
```

---

## 2. 각 노드 역할 분담

### 2.1 ESP32 역할

| 항목 | 내용 |
|---|---|
| 마이크 인터페이스 | I2S 디지털 MEMS 마이크 (INMP441 또는 SPH0645 계열 권장) |
| 오디오 캡처 사양 | 16kHz / 16bit / Mono. 7,680샘플(0.48초) 단위로 청크 분할 |
| 버퍼 관리 | DMA 기반 더블 버퍼링. 0.48초 분량(7,680 샘플 = 15,360 바이트)씩 전송 |
| 전송 단위 | 0.48초 청크. Jetson 측에서 2개를 합쳐 0.96초 YAMNet 윈도우로 처리 |
| 판정 결과 수신 | Jetson으로부터 JSON 이벤트 수신 |
| 진동 출력 | 수신 이벤트의 `vibration_pattern` 필드에 따라 모터 ON/OFF PWM 제어 |
| 하트비트 처리 | Jetson의 heartbeat 수신 시 연결 상태 플래그 유지. 5초 이상 미수신 시 Wi-Fi 재연결 시도 |
| 전력 고려 | 상시 오디오 전송(Wi-Fi active) 시 ~150~250mA 소비. 배터리 운용 시 딥슬립 모드 별도 설계 필요 |

**마이크 선정 기준 비교**

| 마이크 | 인터페이스 | SNR | 가격 | 추천 |
|---|---|---|---|---|
| INMP441 | I2S | 61dB | 저가 | **1순위** — YAMNet 입력 범위에 적합, 라이브러리 풍부 |
| SPH0645 | I2S | 65dB | 저가 | 2순위 — INMP441 대안 |
| MSM261S | PDM | 62dB | 저가 | PDM → I2S 변환 추가 필요, 권장 비선호 |
| 아날로그 마이크 + ADC | ADC | 다양 | 저가 | ESP32 ADC 노이즈 상대적으로 높음, 비추천 |

### 2.2 Jetson Nano 역할

| 항목 | 내용 |
|---|---|
| 오디오 수신 | 네트워크에서 PCM 청크 수신 후 0.96초 윈도우로 재조립 |
| 전처리 | WebRTC NS 노이즈 감소 (M2-NS PR 기준) |
| YAMNet 추론 | TF-Hub YAMNet frozen backbone, 521-class scores 반환 |
| 위험 클래스 필터 | 화이트리스트 12종 score 추출 |
| 후처리 | Debounce K/N=2/3 + cooldown 5초 |
| 판정 결과 송신 | 트리거 이벤트 발생 시 ESP32로 JSON 전송 |
| 하트비트 송신 | 5초 주기로 heartbeat JSON 전송 |
| 로그 기록 | JSONL 로컬 로그 (`output/run.jsonl`) |
| Jetson GPU 활용 | YAMNet TFLite 또는 TF GPU 추론으로 지연 단축 가능 (M4 TFLite 변환 이후) |

**Jetson Nano 성능 참고치**

| 항목 | 수치 |
|---|---|
| CPU | Cortex-A57 4코어 1.43GHz |
| GPU | 128-core Maxwell |
| RAM | 4GB LPDDR4 |
| YAMNet 추론 시간 (CPU) | 약 20~50ms / 윈도우 (실측 필요) |
| YAMNet 추론 시간 (GPU) | 약 5~15ms / 윈도우 (TFLite GPU delegate) |

---

## 3. 통신 프로토콜 선택지 비교

### 3.1 대역폭 요구사항 계산

```
오디오 스트림 (업스트림):
  16,000 Hz × 16bit × 1ch = 256,000 bps = 256 kbps (raw PCM)
  0.48초 청크 = 7,680 샘플 × 2 bytes = 15,360 bytes ≈ 15KB/청크
  청크 전송 주기: 0.48초마다 1회

판정 결과 (다운스트림):
  이벤트 발생 시에만. JSON 1건 ≈ 200~400 bytes
  하트비트 5초 주기 ≈ 100 bytes/5초
  → 평균 < 1kbps, 피크 시에도 수 kbps 이하
```

### 3.2 오디오 업스트림 프로토콜 비교표

| 프로토콜 | 최대 대역폭 | 실효 지연 | 구현 복잡도 | 연결 안정성 | 추천도 |
|---|---|---|---|---|---|
| Wi-Fi UDP (raw PCM) | 충분 (수십 Mbps) | 낮음 (< 5ms 내부) | 낮음 | 패킷 손실 가능 | **1순위** — 단순하고 낮은 지연. 손실 허용 가능 |
| Wi-Fi TCP | 충분 | 중간 (재전송 지연) | 낮음 | 보장됨 | 2순위 — 손실 불허 시 선택. 재전송 누적 시 지연 위험 |
| WebSocket | 충분 | 중간 | 중간 | 보장됨 | 3순위 — 웹 연동 필요 시. 헤더 오버헤드 있음 |
| MQTT | 충분 | 중간 (브로커 경유) | 높음 | 보장됨 | 오디오 스트리밍 용도 부적합. 브로커 단일 장애점 |
| BLE | 최대 ~2Mbps (BLE 5.0) | 낮음 | 높음 | 중간 (10m 내) | 256kbps 지속 전송 시 BLE 5.0도 빠듯. 오디오 스트리밍 비추천 |
| UART/Serial | 최대 ~3Mbps | 매우 낮음 | 매우 낮음 | 매우 높음 | 유선 케이블 필요. 이동 시 제약 |

**권장: Wi-Fi UDP (오디오 업스트림)**

근거:
- 256kbps raw PCM 전송은 Wi-Fi 2.4GHz에서 여유 있게 수용 가능
- UDP는 TCP 대비 재전송 지연이 없어 실시간 스트리밍에 적합
- 오디오 스트리밍에서 패킷 1~2개 손실은 YAMNet 추론에 미치는 영향이 제한적 (0.48초 윈도우 단위로 처리하므로, 손실된 청크는 직전 값으로 zero-fill하거나 skip)
- ESP32의 Wi-Fi 스택(lwIP)에서 UDP 소켓 API 직접 사용 가능, 구현 단순

### 3.3 판정 결과 다운스트림 프로토콜 비교표

| 프로토콜 | 지연 | 구현 복잡도 | 양방향 여부 | 추천도 |
|---|---|---|---|---|
| Wi-Fi UDP (JSON) | 낮음 | 낮음 | 가능 | **1순위** — 이벤트성 소량 전송에 적합 |
| Wi-Fi TCP (JSON) | 낮음~중간 | 낮음 | 가능 | 1순위 대안 — 유실 없이 전달 보장 필요 시 |
| MQTT | 중간 | 높음 | 가능 | 멀티 구독자(스마트폰 앱 동시 수신) 필요 시 고려 |
| BLE GATT Notify | 낮음 | 높음 | 단방향 | 저전력 필요 시. 오디오 스트리밍과 동시 사용 어려움 |

**권장: Wi-Fi TCP (판정 결과 다운스트림)**

근거:
- 판정 이벤트는 즉각 전달이 중요. TCP 재전송 보장으로 유실 방지
- 판정 결과 데이터 양이 작아 TCP 오버헤드 무시 가능
- 오디오 업스트림(UDP)과 판정 결과 다운스트림(TCP)을 분리하여 각 목적에 맞는 프로토콜 사용

### 3.4 지연 예산 분석 (목표: 위험 소리 → 진동까지 200~500ms)

```
[마이크 캡처]
  오디오 DMA 버퍼링: 0.48초 청크 완성 대기            ≈  480ms (hop 주기)
  ※ 이 480ms는 구조적으로 불가피한 최소 지연

[네트워크 전송: ESP32 → Jetson]
  Wi-Fi UDP 전송 지연:                               ≈  5~20ms

[Jetson 처리]
  WebRTC NS:                                        ≈  5~10ms
  YAMNet 추론 (CPU 기준):                           ≈  20~50ms
  Danger filter + Debounce:                         ≈  < 1ms
  소계:                                             ≈  25~60ms

[네트워크 전송: Jetson → ESP32]
  Wi-Fi TCP 전송 지연:                              ≈  5~20ms

[ESP32 진동 모터 구동]
  수신 파싱 + PWM 설정:                             ≈  < 5ms

합계 (최선):  ≈ 480 + 5 + 25 + 5 + 5   ≈  520ms
합계 (최악):  ≈ 480 + 20 + 60 + 20 + 5  ≈  585ms
```

**결론**: hop 0.48초(480ms) 자체가 지배적 지연 원인. 200ms 목표는 현재 구조에서 달성 불가. 500ms 목표는 최선 케이스에서 근접하나, Debounce N=3(K/N=2/3)이 적용되면 최대 0.96초 추가 지연 발생 가능.

| 지연 목표 | 달성 가능성 | 조건 |
|---|---|---|
| 200ms | 불가 | hop 480ms 구조적 하한 |
| 500ms | 조건부 가능 | --no-debounce 모드 + CPU 추론 최적화 시 |
| 1,000ms (개발계획 KPI) | **가능** | Debounce 2/3 포함 시에도 여유 있음 |

**권장**: 목표를 개발계획 KPI인 end-to-end ≤ 1.0초로 설정 유지. 200~500ms는 추후 hop 크기 축소 또는 TFLite GPU delegate 전환 이후 재평가.

---

## 4. 오디오 전송 포맷 결정

### 4.1 PCM raw vs. 압축(Opus 등) 비교

| 항목 | PCM raw (16bit) | Opus (8~32kbps) |
|---|---|---|
| 대역폭 | 256kbps | 8~32kbps (1/8~1/32 감소) |
| ESP32 인코딩 부담 | 없음 | 높음 (Opus 인코더 RAM 약 20~40KB) |
| Jetson 디코딩 부담 | 없음 | 낮음 |
| 위험음 품질 영향 | 없음 | 있음. 총성·유리 파손 같은 transient 성분 손실 위험 |
| 구현 복잡도 | 매우 낮음 | 높음 (라이브러리 포팅 필요) |
| 권장 | **이 프로젝트에서 권장** | Wi-Fi 환경에서 불필요한 복잡성 |

**권장: PCM raw (16bit signed little-endian)**

근거:
- Wi-Fi 환경에서 256kbps는 충분히 수용 가능
- Opus 등 압축 코덱은 단발성 transient(총성, 유리 파손)를 손상시킬 수 있어 위험음 Recall 저하 위험
- ESP32에서 Opus 인코더 구동 시 메모리/연산 부담이 크고, 캡스톤 범위를 초과
- 노이즈 캔슬링은 Jetson 측에서 처리하므로 압축 전 원본 PCM 전달이 NS 성능에 유리

### 4.2 청크 크기 결정

| 청크 크기 | 전송 주기 | 장점 | 단점 |
|---|---|---|---|
| 0.48초 (7,680샘플) | 480ms마다 | 현재 hop과 일치. Jetson에서 즉시 YAMNet 입력 가능 | 청크 하나 손실 시 0.48초 공백 |
| 0.24초 (3,840샘플) | 240ms마다 | 손실 영향 감소 | Jetson에서 2개 조립 필요, 지연 동일 |
| 0.096초 (1,536샘플) | 96ms마다 | 손실 영향 최소 | 패킷 수 증가, 조립 버퍼 복잡 |
| 0.96초 (15,360샘플) | 960ms마다 | YAMNet 윈도우와 정확히 일치 | 지연 크고 손실 시 1초 공백 |

**권장: 0.48초 청크 (hop 단위)**

근거:
- 현재 `src/audio_io/mic_stream.py`가 hop_samples=7,680 단위로 콜백하는 구조와 정합
- Jetson에서 수신 청크 2개를 링 버퍼에 누적하면 0.96초 YAMNet 윈도우 자동 완성
- 전송 오버헤드 대비 균형이 적절

### 4.3 청크 프레임 헤더 (권장)

각 UDP 패킷에 최소한의 헤더를 포함하여 Jetson 측 순서 보정 및 손실 감지를 가능하게 한다.

```
[헤더 8바이트]
  seq_id   : uint32  (단조 증가, 청크 순서 식별)
  timestamp: uint32  (ESP32 ms 단위 내부 시각, 상대값)
[페이로드: 15,360 bytes (7,680 × 16bit)]
총 패킷 크기: 15,368 bytes
```

---

## 5. 응답 메시지 스키마

### 5.1 위험 이벤트 메시지 (Jetson → ESP32)

```json
{
  "event": "danger",
  "ts": 1746873600.123,
  "class": "glass_shatter",
  "score": 0.87,
  "duration_ms": 960,
  "vibration_pattern": "short_burst_3",
  "severity": "high",
  "seq": 142
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `event` | string | `"danger"` 고정 (heartbeat와 구분) |
| `ts` | float | Jetson 기준 Unix epoch (초) |
| `class` | string | 화이트리스트 enum 값 (예: `glass_shatter`, `screaming`) |
| `score` | float | 트리거 클래스의 YAMNet score (0.0~1.0) |
| `duration_ms` | int | 추론 윈도우 길이, 고정 960 |
| `vibration_pattern` | string | 진동 패턴 식별자. ESP32가 패턴 테이블을 조회해 모터 제어 |
| `severity` | string | `"high"` / `"medium"` / `"low"` — score 구간 기반 |
| `seq` | int | 단조 증가. ESP32 측 중복 이벤트 제거에 활용 |

### 5.2 하트비트 메시지 (Jetson → ESP32, 5초 주기)

```json
{
  "event": "heartbeat",
  "ts": 1746873605.000,
  "seq": 143
}
```

### 5.3 severity 기준

| score 구간 | severity | 진동 강도 |
|---|---|---|
| 0.8 이상 | `"high"` | 강함 |
| 0.5 ~ 0.8 미만 | `"medium"` | 중간 |
| threshold ~ 0.5 미만 | `"low"` | 약함 |

---

## 6. 진동 패턴 설계

### 6.1 위험 클래스별 패턴 차별화 방안

클래스별 패턴 차별화는 사용자에게 어떤 종류의 위험인지 인지시킬 수 있어 유용하다. 단, ESP32 펌웨어 복잡도와 사용자 학습 부담을 고려해야 한다.

**방안 A: 클래스별 차별화 패턴**

| 위험 클래스 | 패턴 식별자 | 패턴 설명 | 근거 |
|---|---|---|---|
| `screaming`, `baby_cry` | `urgent_long` | 1초 ON → 0.5초 OFF, 3회 반복 | 인명 위험, 즉각 대피 필요 |
| `gunshot`, `explosion` | `strong_single` | 0.2초 강한 단발 × 2회 | 충격음 모방, 즉각성 강조 |
| `fire_alarm`, `smoke_alarm` | `short_burst_3` | 0.2초 ON × 3회 (0.1초 간격) | 화재 경보 패턴 모방 |
| `glass_shatter`, `breaking` | `double_pulse` | 0.1초 ON × 2회 (빠른 간격) | 충격음 특성 반영 |
| `siren`, `civil_defense_siren` | `wave_pattern` | 0.5초 ON → 0.5초 OFF, 연속 | 사이렌 리듬 모방 |
| `car_alarm`, `vehicle_horn` | `rhythmic_4` | 0.15초 ON × 4회 | 자동차 경보 리듬 |

**방안 B: severity 기반 단순화 패턴**

| severity | 패턴 | 설명 |
|---|---|---|
| `high` | 강한 연속 진동 1.5초 | 즉각적 강한 알림 |
| `medium` | 0.3초 ON × 3회 | 중간 알림 |
| `low` | 0.2초 단발 | 약한 알림 |

**권장: 방안 B를 기본으로, 방안 A를 선택적으로 지원**

근거:
- MVP 단계에서는 방안 B(3단계 severity)로 구현 부담 최소화
- ESP32 펌웨어에 클래스별 패턴 테이블을 정의해 두고, `vibration_pattern` 필드를 통해 방안 A로 확장 가능하게 설계
- JSON 필드 `vibration_pattern`이 이미 스키마에 포함되어 있으므로 하위 호환성 유지

### 6.2 PWM 제어 사양

| 항목 | 값 |
|---|---|
| 모터 타입 | 코인형 진동 모터 (3V, 100~150mAh) 또는 ERM 모터 |
| PWM 주파수 | 1kHz |
| 강도 제어 | PWM 듀티 사이클: High=100%, Medium=70%, Low=40% |
| 구동 회로 | MOSFET (2N7000 또는 NPN 트랜지스터 + 역방향 다이오드) |

---

## 7. 기존 코드와의 통합 지점

### 7.1 현재 코드 구조와의 차이

현재 `src/cli.py`는 로컬 마이크(`--input mic`) 또는 WAV 파일(`--input FILE`)만 지원한다. 네트워크에서 오디오를 수신하는 경로가 없다.

```
[현재 구조]
  src/cli.py
    --input mic   → src/audio_io/mic_stream.py → YAMNet 추론
    --input FILE  → src/audio_io/file_reader.py → YAMNet 추론

[필요한 변경]
  src/cli.py
    --input mic      → (유지)
    --input FILE     → (유지)
    --input network  → src/audio_io/network_stream.py  [신규]
```

### 7.2 신규 추가 필요 모듈

| 파일 경로 | 역할 | 마일스톤 |
|---|---|---|
| `src/audio_io/network_stream.py` | UDP 소켓 수신, 청크 조립, 0.96초 링 버퍼 관리 | M5-임베디드 |
| `src/embedded/wifi_sender.py` | 판정 결과 JSON을 TCP로 ESP32에 전송, heartbeat | M5-임베디드 |

### 7.3 기존 `uart_sender.py` 재검토

현재 `src/embedded/uart_sender.py`는 M5 UART 플레이스홀더로 정의되어 있다. 이번 시나리오(ESP32 ↔ Jetson Wi-Fi)에서는 UART가 아닌 Wi-Fi 소켓 기반으로 변경된다.

| 항목 | 기존 계획 (M5) | 변경 후 |
|---|---|---|
| 통신 방식 | UART 115200bps (유선) | Wi-Fi TCP/UDP (무선) |
| 대상 파일 | `uart_sender.py` | `wifi_sender.py` (신규) |
| `uart_sender.py` 처리 | 삭제 또는 유선 백업용 유지 | 유선 연결 테스트/폴백용으로 유지 가능 |

### 7.4 `src/cli.py` 변경 필요 사항

```
[추가 CLI 옵션 — network 모드용]
  --input network          : 네트워크 스트림 수신 모드
  --listen-port PORT       : UDP 수신 포트 (기본: 5005)
  --send-host HOST         : 판정 결과 전송 대상 IP (ESP32 주소)
  --send-port PORT         : 판정 결과 전송 TCP 포트 (기본: 5006)

[데이터 흐름 변경]
  기존: mic/file → audio_io → YAMNet → trigger → (콘솔 출력)
  변경: network → audio_io/network_stream → YAMNet → trigger → wifi_sender → ESP32
```

### 7.5 CLAUDE.md 데이터 흐름 다이어그램 갱신 필요

현재 CLAUDE.md의 데이터 흐름:

```
[마이크/파일 입력 16kHz mono]
  → audio_io/ (링 버퍼, 0.96s 윈도우 / 0.48s hop)
  → preprocess/ (노이즈 캔슬링, M2 이후)
  → model/yamnet_wrapper.py (TF-Hub YAMNet, backbone freeze)
  → model/danger_filter.py (12종 화이트리스트 score 추출)
  → postprocess/trigger.py (임계값 + cooldown)
  → embedded/uart_sender.py (UART JSON 알림, M5 이후)
```

M5 임베디드 통합 이후 갱신 필요 내용:

```
[ESP32 오디오 스트림 (UDP, 16kHz PCM)]
  → audio_io/network_stream.py (청크 수신 + 링 버퍼)
  → preprocess/noise_suppress.py (WebRTC NS)
  → model/yamnet_wrapper.py (YAMNet frozen backbone)
  → model/danger_filter.py (12종 화이트리스트 score 추출)
  → postprocess/trigger.py (Debounce K/N + cooldown)
  → embedded/wifi_sender.py (Wi-Fi TCP JSON 판정 결과 → ESP32)
```

---

## 8. 리스크 및 미해결 이슈

### 8.1 리스크 레지스터

| ID | 리스크 | 영향 | 가능성 | 대응책 |
|---|---|---|---|---|
| RE-1 | Wi-Fi 단절 시 오디오 스트림 중단 | 높음 | 중간 | ESP32: 연결 해제 감지 즉시 자동 재연결 시도. Jetson: 수신 타임아웃(2초) 후 재연결 대기 상태 진입 |
| RE-2 | Jetson 추론 지연 누적 (큐 빌드업) | 높음 | 낮음 | 수신 청크를 큐에 쌓되, 큐 길이 > 2(=1초 분량) 초과 시 가장 오래된 항목 드롭. 실시간성 우선 |
| RE-3 | ESP32 타임스탬프 드리프트 | 중간 | 높음 | ESP32 timestamp를 신뢰하지 않고, Jetson 수신 시각 기준으로 YAMNet timestamp 부여. ESP32 timestamp는 디버그 목적으로만 사용 |
| RE-4 | ESP32 Wi-Fi 연속 전송 전력 소비 | 중간 | 높음 | USB 전원 또는 대용량 LiPo(2000mAh 이상) 사용 권장. 배터리 운용 시 딥슬립 + 이벤트 감지 모드 별도 설계 필요 (범위 외) |
| RE-5 | 같은 Wi-Fi AP에서 대역폭 경합 | 중간 | 낮음 | 5GHz 대역 Wi-Fi 사용 권장. 실환경 AP 간섭이 문제 될 경우 AP 전용 구성 |
| RE-6 | UDP 패킷 순서 역전 | 낮음 | 낮음 | seq_id 헤더로 감지. 역전 패킷 발생 시 drop (오디오 연속성보다 실시간성 우선) |
| RE-7 | Jetson 단일 장애점 | 높음 | 낮음 | 프로세스 감시자(systemd) 자동 재시작. 향후 엣지 추론 전환(M4 TFLite) 시 재검토 |
| RE-8 | INMP441 등 I2S 마이크와 ESP32 펌웨어 호환성 | 중간 | 중간 | ESP-IDF I2S 드라이버 사용. Arduino-ESP32 프레임워크도 지원하나 DMA 제어는 ESP-IDF 권장 |

### 8.2 미해결 설계 이슈

| 이슈 | 현황 | 해결 필요 시점 |
|---|---|---|
| Opus 압축 채택 여부 최종 결정 | 미결정. 현재 PCM raw 권장이나, Wi-Fi 환경이 불안정할 경우 재검토 필요 | M5 착수 전 |
| ESP32 펌웨어 개발 언어/프레임워크 | ESP-IDF(C) vs Arduino-ESP32(C++) 미결정 | M5 착수 전 |
| 진동 패턴 방안 A/B 최종 선택 | 미결정 | M5 착수 전 |
| Jetson → ESP32 전송 시 ESP32 IP 주소 관리 | 고정 IP(권장) vs mDNS 동적 탐색 미결정 | M5 착수 전 |
| 다중 ESP32 지원 (1:N 배포) | 현재 단일 ESP32 가정. 다수 착용자 시나리오 미설계 | M6 이후 |

---

## 9. 마일스톤 제안 (임베디드 통합 포함)

### 9.1 기존 마일스톤과의 정합

```
기존: M1(베이스라인) → M2(Debounce + NS) → M3(헤드학습) → M4(TFLite) → M5(임베디드) → M6(현장테스트)
```

현재 M1은 로컬 마이크/파일 기준으로 완료. M2 Debounce PR 완료. M2-NS PR 진행 예정.

### 9.2 임베디드 통합을 위한 M5 세분화 제안

기존 M5 한 단계를 아래와 같이 3개 하위 단계로 쪼갠다.

| ID | 단계명 | 내용 | 산출물 | Exit Criteria |
|---|---|---|---|---|
| M5-a | Serial PoC | Jetson ↔ PC/ESP32 UART로 판정 결과 전송 검증. 기존 `uart_sender.py` 활용 | 동작하는 UART 전송 데모 | Jetson → ESP32 UART JSON 수신 및 LED ON/OFF 확인 |
| M5-b | Wi-Fi 오디오 스트리밍 | ESP32 I2S 마이크 → Wi-Fi UDP → Jetson 수신 파이프라인 구성. `network_stream.py` 구현 | `src/audio_io/network_stream.py` | Jetson에서 수신 PCM으로 YAMNet 추론 결과가 로컬 마이크와 동등 수준임을 확인 |
| M5-c | 진동 통합 | Jetson 판정 결과 Wi-Fi TCP → ESP32 수신 → 진동 모터 ON/OFF | `src/embedded/wifi_sender.py`, ESP32 펌웨어 | 위험음 재생 → 500ms 이내 진동 발생. end-to-end latency ≤ 1.0초 측정 |

### 9.3 전체 마일스톤 흐름 (업데이트)

```
M1 완료 (로컬 베이스라인)
  │
  ├─ M2 Debounce PR ← 현재 위치
  │
  ├─ M2-NS PR (WebRTC NS 통합)
  │
  ├─ M2-AB PR (NS A/B 비교, FAR 측정)
  │
  ├─ M3 (헤드 파인튜닝: FSD50K + UrbanSound8K)
  │
  ├─ M4 (TFLite 변환 + int8 양자화)
  │
  ├─ M5-a (UART PoC)
  ├─ M5-b (Wi-Fi 오디오 스트리밍)
  ├─ M5-c (진동 통합 + end-to-end 지연 측정)
  │
  └─ M6 (현장 테스트 3개 환경)
```

### 9.4 M5-b 이전 병행 가능 작업

ESP32 펌웨어 개발은 M3/M4와 병행 가능하다. Jetson 측 소프트웨어(M5-b)가 완성되기 전에 PC에서 UDP 더미 오디오를 전송하는 테스트 스크립트로 ESP32 Wi-Fi 스택을 먼저 검증할 수 있다.

---

## 10. 결정 필요 사항 (Decision Points)

다음 단계에서 사용자/팀이 결정해야 할 항목을 우선순위 순으로 정리한다.

| 순위 | 결정 항목 | 선택지 | 권장 | 결정 시한 |
|---|---|---|---|---|
| P1 | ESP32 Wi-Fi 통신 방식 최종 확정 | UDP(업스트림) + TCP(다운스트림) vs 양방향 TCP | UDP+TCP 혼합 권장 | M5-b 착수 전 |
| P1 | ESP32 펌웨어 개발 프레임워크 | ESP-IDF(C) vs Arduino-ESP32(C++) | Arduino-ESP32 (개발 속도 우선) | M5-a 착수 전 |
| P1 | 진동 패턴 방안 선택 | 방안 A(클래스별 차별화) vs 방안 B(severity 3단계) | 방안 B 먼저, 방안 A로 확장 | M5-c 착수 전 |
| P2 | ESP32 고정 IP 방식 | 공유기 DHCP 고정 할당 vs ESP32 AP 모드 자체 핫스팟 | DHCP 고정 할당 | M5-b 착수 전 |
| P2 | UART 폴백 유지 여부 | uart_sender.py 유지(유선 백업) vs 삭제 | 유지 권장 (폴백용) | M5-a 착수 전 |
| P3 | 오디오 압축 최종 결정 | PCM raw vs Opus | PCM raw 권장 | M5-b 전 Wi-Fi 환경 평가 후 |
| P3 | 다중 ESP32 지원 범위 | 단일 디바이스 vs 1:N | 현재는 단일. 캡스톤 범위 논의 필요 | M6 전 |

---

## 11. 참고: 관련 문서 및 코드 위치

| 항목 | 경로 |
|---|---|
| 전체 개발 계획 | `docs/development-plan.md` |
| M1 베이스라인 스펙 | `docs/m1-initial-model-spec.md` |
| M2 Debounce 스펙 | `docs/m2-debounce-spec.md` |
| 핵심 결정 요약 | `CLAUDE.md` |
| 현재 CLI 진입점 | `src/cli.py` |
| UART 플레이스홀더 | `src/embedded/uart_sender.py` |
| 노이즈 억제 모듈 (M2-NS 예정) | `src/preprocess/noise_suppress.py` |
| 화이트리스트 설정 | `config/whitelist.yaml` |

---

*문서 버전: v0.1 (2026-05-11 초안). M5-a 착수 전 통신 프로토콜 결정 사항 반영 후 v0.2 갱신 예정.*
