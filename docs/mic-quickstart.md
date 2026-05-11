# 노트북 마이크 실시간 분석 가이드 (M1)

이 문서는 현재 M1 베이스라인 상태에서 노트북 내장 마이크로 위험 소리 감지를 시연·검증하는 절차를 정리합니다.

## 1. 현재 구현 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| `src/audio_io/mic_stream.py` | ✅ 완료 | sounddevice 콜백 + 0.96s 윈도우 |
| `src/cli.py --input mic` | ✅ 완료 | 마이크 모드 진입점 |
| YAMNet 추론 + 화이트리스트 12종 | ✅ 완료 | `config/whitelist.yaml` |
| 후처리(임계값 + 5s cooldown) | ✅ 완료 | M1 단순 버전 |
| 노이즈 캔슬링 전처리 | ⏳ M2 | 현재는 패스스루 |
| 임베디드 UART 송신 | ⏳ M5 | 플레이스홀더 |
| `.venv` 가상환경 | ❌ 미생성 | 사용자 환경에서 직접 생성 필요 |
| 의존성 설치 | ❌ 미수행 | `pip install -r requirements.txt` |
| YAMNet 모델 캐시 | ❌ 미수행 | 최초 실행 시 ~200MB 다운로드 |

## 2. 사전 준비

- Python 3.11 (Windows)
- 인터넷 연결 (TF-Hub에서 YAMNet 1회 다운로드)
- 동작 가능한 입력 마이크 (노트북 내장 또는 USB 마이크)

## 3. 실행 절차

PowerShell에서 프로젝트 루트(`C:\CapstoneDesign\model`)에 위치한 상태로 진행합니다.

### 3.1 Python 3.11 가상환경 생성·활성화

Python 3.11 설치 여부를 먼저 확인합니다.

```powershell
py -3.11 --version
```

이미 Python 3.12로 `.venv`를 만들었다면 삭제 후 다시 생성합니다.

```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
```

Python 3.11로 새 가상환경을 생성합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

`python --version` 출력이 `Python 3.11.x`인지 확인합니다.

> Activate.ps1 실행이 막히면 1회만:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3.2 의존성 설치

```powershell
pip install -r requirements.txt
```

- 5~10분 소요 (TensorFlow 용량 큼)
- Python 3.11에서 TensorFlow 2.13~2.15 범위가 설치됩니다. Python 3.12 가상환경에서는 이 범위의 TensorFlow가 설치되지 않습니다.

### 3.3 환경 검증 (YAMNet 최초 다운로드 포함)

```powershell
python scripts/verify_inference.py
```

`PASS` 출력이 떠야 다음 단계 진행 가능. 실패 시 TF/PortAudio 설치 확인.

### 3.4 마이크 장치 확인

```powershell
python -c "import sounddevice; print(sounddevice.query_devices())"
```

출력 예시:
```
> 0 마이크 (Realtek Audio), MME (2 in, 0 out)
  1 스피커 (Realtek Audio), MME (0 in, 2 out)
  ...
```

`>` 표시가 기본 입력 장치. 기본 장치가 원하는 마이크가 아니면 인덱스를 메모해두고 CLI에 `--device <idx>`를 전달합니다.

### 3.5 실시간 분석 실행

```powershell
python -m src.cli --input mic --threshold 0.4 --verbose --log output/run.jsonl
```

옵션 설명:
- `--input mic` — 마이크 입력 모드
- `--threshold 0.4` — 클래스 score 임계값 (기본 0.5보다 낮춰서 감도↑)
- `--verbose` — 윈도우별 score 디버그 출력
- `--log output/run.jsonl` — 트리거 이벤트를 JSONL로 저장

종료: `Ctrl+C`

## 4. 시연·검증 팁

### 4.1 위험 소리 트리거가 잘 발생하는 음원

위험 클래스 12종은 **일상 대화·박수·키보드 소리로는 거의 트리거되지 않습니다.** 실제 검증은 다음 방식으로:

| 방법 | 비고 |
|---|---|
| YouTube에서 "fire alarm sound", "police siren", "glass breaking sfx" 재생 후 노트북 스피커 → 노트북 마이크 | 가장 빠르고 안전 |
| ESC-50 데이터셋의 `siren`, `glass_breaking`, `fire_alarm` 클립 재생 | 라이선스: CC-BY |
| 직접 비명·고성 (안전한 환경에서) | 이웃 주의 |

### 4.2 임계값 튜닝

- 트리거가 너무 적음 → `--threshold 0.3`
- 오탐(false positive) 많음 → `--threshold 0.5` 이상
- 환경별로 적정값 다름. 카페·거리·실내 각각 측정 권장.

### 4.3 cooldown

같은 클래스가 5초 내 재트리거되지 않습니다. (`config/whitelist.yaml`의 `cooldown_sec`)

## 5. 알려진 제한 (M1)

- **노이즈 캔슬링 미적용**: 배경 소음이 큰 환경에서는 오탐/미탐 발생 가능. M2에서 WebRTC NS 통합 예정.
- **debounce 미적용**: 단일 윈도우만으로 트리거. 일시적 피크에 의한 false positive 가능. M2에서 K/N 다수결 추가 예정.
- **임베디드 알림 미연동**: 트리거는 stdout과 JSONL 로그로만 출력. UART 송신은 M5에서 구현.
- **마이크 게인 자동 조절 없음**: 너무 작거나 크면 OS 사운드 설정에서 입력 레벨 조정 필요.

## 6. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: tensorflow` | 가상환경 미활성화. `.venv\Scripts\Activate.ps1` 다시 실행 |
| `OSError: PortAudio library not found` | `pip install --force-reinstall sounddevice` 또는 시스템 PortAudio 설치 |
| 마이크 입력이 0에 가까운 무음 | OS 사운드 설정에서 입력 장치/볼륨 확인. `sounddevice.query_devices()`로 기본 장치 확인 |
| YAMNet 다운로드 실패 | 사내망/방화벽이면 `TFHUB_CACHE_DIR` 환경변수로 수동 캐시 디렉터리 지정 후 모델 사전 배치 |
| `sounddevice status: input overflow` 경고 다수 | 노트북 부하 큼. 다른 무거운 프로세스 종료 또는 `blocksize` 조정(코드 수정 필요) |

## 7. 다음 단계

- ESC-50/UrbanSound8K 클립을 `data/sample/`에 두고 파일 모드(`--input data/sample/xxx.wav`)로 정량 검증
- M2: WebRTC NS 전처리 + debounce 후처리 통합
- M3: YAMNet embedding + 경량 분류 헤드 파인튜닝
- M5: UART JSON 라인 송신으로 임베디드 모듈 연동
