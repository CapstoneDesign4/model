# 실행 가이드

> YAMNet 기반 위험 소리 감지기를 노트북·SBC에서 실행하는 전체 절차.
> 대상 OS: Windows 11 (PowerShell). Linux/macOS도 명령어 일부만 다름.
> M1 베이스라인 + M2 Debounce K/N(2/3) 적용 상태 기준.

---

## 1. 사전 준비

| 항목 | 요구 사항 |
|---|---|
| Python | 3.10 이상 |
| 메모리 | 최소 2GB (TensorFlow 로딩) |
| 디스크 | 약 1.5GB (TF + YAMNet 캐시) |
| 네트워크 | 최초 1회 (TF-Hub에서 YAMNet 다운로드) |
| 마이크 | 마이크 모드 사용 시 OS에서 인식되는 입력 장치 |

---

## 2. 환경 구성 (1회만)

PowerShell을 프로젝트 루트(`C:\Users\user\model`)에서 연 상태로 진행.

### 2.1 가상환경 생성·활성화

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Activate.ps1 실행이 막힐 때 (1회만):**
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 2.2 의존성 설치

```powershell
pip install -r requirements.txt
```

- 5~10분 소요 (TensorFlow 다운로드).
- pip가 TensorFlow 2.16+ 를 끌어오면 명시적으로 다운그레이드:
  ```powershell
  pip install "tensorflow>=2.13,<2.16"
  ```

### 2.3 YAMNet 다운로드 검증

```powershell
python scripts/verify_inference.py
```

- 최초 실행 시 약 200MB 모델을 `~/.cache/tfhub_modules` 에 다운로드.
- `PASS: YAMNet inference OK` 출력이 떠야 다음 단계 진행.

---

## 3. 실행 — 파일 모드

WAV 파일 1개를 윈도우 단위로 분석.

### 3.1 기본 실행

```powershell
python -m src.cli --input data/sample/test.wav --threshold 0.5
```

### 3.2 자주 쓰는 옵션 조합

```powershell
# 매 윈도우 score + debounce 큐 상태 출력
python -m src.cli --input data/sample/test.wav --threshold 0.4 --verbose

# 결과를 JSONL 로그로 저장
python -m src.cli --input data/sample/test.wav --log output/run.jsonl

# 강제 트리거 (동작 확인용 — 모든 윈도우에서 trigger 시도)
python -m src.cli --input data/sample/test.wav --threshold 0.0 --no-debounce
```

---

## 4. 실행 — 마이크 모드

노트북 내장 마이크 또는 USB 마이크에서 실시간 분석.

### 4.1 마이크 장치 확인 (선택)

```powershell
python -c "import sounddevice; print(sounddevice.query_devices())"
```

- 출력의 `>` 표시가 기본 입력 장치.
- 다른 장치를 쓰려면 인덱스를 메모하고 `--device <idx>` 로 지정.

### 4.2 기본 실행

```powershell
python -m src.cli --input mic --threshold 0.4
```

### 4.3 권장 조합

```powershell
# 시연·튜닝 시: verbose + JSONL 로그
python -m src.cli --input mic --threshold 0.4 --verbose --log output/run.jsonl

# 특정 마이크 장치 지정
python -m src.cli --input mic --threshold 0.4 --device 1
```

종료: `Ctrl+C`

---

## 5. M2 Debounce 옵션

M2부터 단발성 false trigger 억제를 위해 K/N 다수결 후처리가 적용됩니다.

### 5.1 동작 요약

```
최근 N=3 윈도우 중 K=2 이상에서 score >= threshold ⇒ trigger 후보
trigger 후보 + cooldown 5초 미경과 아님 ⇒ 실제 emit
```

### 5.2 옵션 표

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--debounce-window N` | 3 (yaml 또는 코드) | 최근 N개 윈도우 보관 |
| `--debounce-k K` | 2 (yaml 또는 코드) | K개 이상 양성 시 트리거. `1 ≤ K ≤ N` |
| `--no-debounce` | off | M1 동작 재현(단일 윈도우 트리거). 비교용 |

우선순위: **CLI 인수 > [config/whitelist.yaml](../config/whitelist.yaml) `debounce` 블록 > 코드 기본값(3,2)**

### 5.3 사용 예

```powershell
# 보수적(false alarm 우선): 3/3 모두 양성이어야 trigger
python -m src.cli --input mic --debounce-window 3 --debounce-k 3

# 민감(놓치지 않는 게 우선): 5/2
python -m src.cli --input mic --debounce-window 5 --debounce-k 2

# M1 동작 재현 (debounce 비활성화)
python -m src.cli --input data/sample/test.wav --no-debounce --verbose
```

### 5.4 verbose 출력 해석

```
[2026-05-10 14:32:01.123] WINDOW scores:
  screaming             : 0.7821  votes=[1,1,0]  sum=2/3  PASS
  glass_shatter         : 0.4502  votes=[0,1,0]  sum=1/3  --
  fire_alarm            : 0.6001  votes=[1,1,1]  sum=3/3  COOLDOWN
```

- `votes=[1,1,0]` — 최근 3 윈도우 양성 이력
- `sum=X/N` — 양성 개수 / 윈도우 크기
- 상태:
  - `PASS` — debounce 통과 + cooldown 미경과 아님 → 이 윈도우에서 `DANGER:` 라인 출력됨
  - `COOLDOWN` — debounce는 통과했으나 5초 cooldown 중이라 발화 억제됨
  - `--` — debounce 미통과

### 5.5 JSONL 로그 스키마

`--log output/run.jsonl` 사용 시 매 윈도우 한 줄:

```json
{
  "timestamp": 1746856321.123,
  "window_duration_ms": 960,
  "scores": {"screaming": 0.78, "glass_shatter": 0.12, "...": "..."},
  "triggered": ["screaming"],
  "top_score": 0.78,
  "debounce_votes": {"screaming": [1,1,0], "fire_alarm": [1,1,1], "...": "..."}
}
```

---

## 6. CLI 옵션 전체 표

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--input` | str | 필수 | `mic` 또는 WAV 파일 경로 |
| `--threshold` | float | 0.5 | 전체 클래스 공통 임계값 오버라이드 |
| `--config` | str | `config/whitelist.yaml` | 화이트리스트 설정 파일 |
| `--hop` | float | 0.48 | hop 길이(초). 실시간 지연 조정 |
| `--verbose` | flag | off | 매 윈도우 score + debounce 상태 출력 |
| `--log` | str | None | 결과를 JSONL로 저장할 경로 |
| `--device` | int | 시스템 기본 | 마이크 장치 인덱스 (mic 모드 한정) |
| `--debounce-window` | int | yaml/3 | debounce 슬라이딩 윈도우 크기 N |
| `--debounce-k` | int | yaml/2 | 트리거 임계 양성 수 K |
| `--no-debounce` | flag | off | debounce 비활성화 (M1 동작) |

`--help` 로도 확인 가능 (TF 로드 없이 즉시 응답):

```powershell
python -m src.cli --help
```

---

## 7. 단위 테스트

```powershell
# 네트워크 불필요 테스트만 (debounce 등)
pytest tests/ -v -k "not yamnet"

# Debounce 단위 테스트만 (29개)
pytest tests/test_debounce_trigger.py -v

# YAMNet 로딩까지 포함 (네트워크 필요)
pytest tests/ -v -k "yamnet"
```

---

## 8. 시연용 음원

위험 클래스 12종은 일상 대화·박수·키보드 소리로는 거의 트리거되지 않습니다.

| 방법 | 비고 |
|---|---|
| YouTube에서 "fire alarm sound", "police siren", "glass breaking sfx" 재생 → 노트북 스피커 → 마이크 | 가장 간단 |
| ESC-50 데이터셋의 `siren`, `glass_breaking`, `fire_alarm` 클립 | 라이선스 CC-BY |
| 직접 비명·고성 (안전한 환경) | 이웃 주의 |

---

## 9. 임계값 튜닝 가이드

| 증상 | 조치 |
|---|---|
| 트리거가 너무 적음 | `--threshold 0.3` 으로 낮추기 |
| 오탐(false positive) 많음 | `--threshold 0.5` 이상으로 올리기 또는 `--debounce-k 3 --debounce-window 3` |
| 단발성 피크에 자주 반응 | M2 debounce 기본값(2/3) 유지. 더 보수적이면 3/3 |
| 환경별로 적정값 다름 | 카페·거리·실내 각각 측정 권장. M3에서 환경 프로파일 도입 예정 |

---

## 10. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: tensorflow` | 가상환경 미활성화. `.venv\Scripts\Activate.ps1` 다시 실행 |
| `OSError: PortAudio library not found` | `pip install --force-reinstall sounddevice`, Linux는 `sudo apt install libportaudio2` |
| 마이크 입력이 0에 가까운 무음 | OS 사운드 설정에서 입력 장치/볼륨 확인. `sounddevice.query_devices()` 로 기본 장치 재확인 |
| YAMNet 다운로드 실패 | 사내망/방화벽이면 `TFHUB_CACHE_DIR` 환경변수로 캐시 디렉터리 지정 후 모델 사전 배치 |
| `sounddevice status: input overflow` 다수 | 노트북 부하 큼. 다른 무거운 프로세스 종료 또는 `--hop` 값 키우기 |
| `[ERROR] --debounce-k (3) 은 --debounce-window (2) 이하` | `K ≤ N` 만족하도록 인수 조정 |
| `--help` 가 느리거나 멈춤 | TF가 import되어 그럼. M2부터 import 지연됐으니 즉시 응답해야 정상. 멈추면 venv 손상 의심 |

---

## 11. 알려진 제한

- **노이즈 캔슬링 미적용** — 배경 소음이 큰 환경에서는 오탐/미탐 가능. M2 후속 PR에서 WebRTC NS 통합 예정.
- **임베디드 알림 미연동** — 트리거는 stdout/JSONL에만. UART 송신은 M5에서.
- **마이크 게인 자동 조절 없음** — OS 사운드 설정에서 입력 레벨 조정.
- **클래스별 개별 debounce 미지원** — M2는 글로벌 N/K만. 클래스별 오버라이드는 M3 이후.

---

## 12. 참고 문서

- [docs/development-plan.md](development-plan.md) — 전체 개발 계획·마일스톤
- [docs/m1-initial-model-spec.md](m1-initial-model-spec.md) — M1 베이스라인 상세 스펙
- [docs/m2-debounce-spec.md](m2-debounce-spec.md) — M2 Debounce K/N 상세 스펙
- [docs/mic-quickstart.md](mic-quickstart.md) — 마이크 시연 빠른 시작(M1 기준)

---

*문서 버전: v0.1 (2026-05-10). M2 NS 통합 PR 머지 시 §9·§11 갱신 예정.*
