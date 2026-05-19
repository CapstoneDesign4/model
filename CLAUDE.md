# CLAUDE.md

이 파일은 이 저장소에서 작업할 때 Claude Code(claude.ai/code)에게 지침을 제공합니다.

## 프로젝트 개요

이 저장소는 캡스톤 디자인 프로젝트의 모델 컴포넌트를 포함합니다. YAMNet을 이용해 실제 환경의 소리를 분석하고, 그 중에서 위험한 소리를 감지하여 임베디드 모델에 알려주는 모델을 제작하는 것이 목표입니다.

### 핵심 요구사항

- **YAMNet 기반 소리 분석**: 사전 학습된 YAMNet 모델을 활용해 오디오 입력을 분류합니다.
- **위험 소리 클래스 한정**: YAMNet의 전체 클래스 중 위험 상황과 관련된 클래스(예: 비명, 유리 깨짐, 차량 경적, 화재 경보, 총소리 등)만으로 범위를 좁혀 분류합니다.
- **위험 소리 감지 → 임베디드 알림**: 여러 소리가 섞인 환경에서도 위험 소리만 식별해 임베디드 모듈로 신호를 전달합니다.
- **노이즈 캔슬링**: 배경 소음 속에서 위험 소리만 정확히 감지할 수 있도록 노이즈 캔슬링/노이즈 감소 전처리를 추가합니다.

## 시작하기

```powershell
# Python 3.11 가상환경 생성 및 활성화
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version

# 의존성 설치
pip install -r requirements.txt

# YAMNet 로딩 검증 (최초 실행 시 모델 다운로드 발생)
python scripts/verify_inference.py

# WAV 파일로 검증 (샘플 파일이 있을 경우)
python scripts/verify_inference.py --file data/sample/test.wav
```

## 명령어

```powershell
# WAV 파일 위험 소리 분석
python -m src.cli --input data/sample/test.wav --threshold 0.5 --verbose

# 마이크 실시간 분석
python -m src.cli --input mic --threshold 0.4 --log output/run.jsonl

# 임계값 0으로 강제 트리거 (동작 확인용)
python -m src.cli --input data/sample/test.wav --threshold 0.0

# 단위 테스트 (네트워크 불필요 테스트만)
pytest tests/ -v

# 단위 테스트 (YAMNet 로딩 포함, 네트워크 필요)
pytest tests/ -v -k "yamnet"
```

### Docker 명령어 (Python/TF 환경 불필요)

```powershell
# 이미지 빌드 (첫 빌드: 10~15분 소요)
docker build -t yamnet-danger:latest .

# WAV 파일 분석
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  yamnet-danger:latest --input data/sample/test.wav --threshold 0.5 --verbose

# pytest 실행
docker run --rm --entrypoint pytest yamnet-danger:latest tests/ -v
```

## 아키텍처

진입점: `src/cli.py` (`python -m src.cli`)

데이터 흐름:
```
[마이크/파일 입력 16kHz mono]
  → audio_io/ (링 버퍼, 0.96s 윈도우 / 0.48s hop)
  → preprocess/ (노이즈 캔슬링, M2 이후)
  → model/yamnet_wrapper.py (TF-Hub YAMNet, backbone freeze)
  → model/danger_filter.py (12종 화이트리스트 score 추출)
  → postprocess/trigger.py (임계값 + cooldown)
  → embedded/uart_sender.py (UART JSON 알림, M5 이후)
```

주요 설계 결정:
- YAMNet backbone 전체 freeze. M3에서 경량 헤드(Dense 2층, sigmoid)만 학습.
- 출력은 argmax 단일 라벨이 아닌 클래스별 독립 sigmoid (multi-label).
- 화이트리스트 12종 인덱스: 11, 20, 302, 304, 390, 391, 393, 394, 420, 421, 435, 437, 464.
- glass(435) + shatter(437)는 max()로 통합하여 `glass_shatter` 단일 이벤트 처리.
- 후처리: 단일 임계값(M1) → debounce K/N 다수결(M2) → 환경별 프로파일(M3).

## M2 Debounce 결정 사항 (2026-05-10 확정)

- **Debounce K/N**: 클래스별 독립 슬라이딩 윈도우 N=3, 트리거 임계 K=2 (2/3 다수결).
- **상태 자료구조**: `DebounceState` (deque maxlen=N) 를 기존 `Trigger` 클래스 내부에 추가. 외부 인터페이스 최대한 유지.
- **cooldown 위치**: debounce 통과 후 적용 (순서: debounce → cooldown → emit).
- **config/whitelist.yaml 변경**: 최상단에 글로벌 `debounce: {window: 3, k: 2}` 블록 추가. 블록 없으면 기본값 적용 (하위 호환).
- **CLI 신규 옵션**: `--debounce-window` (int, 기본 3), `--debounce-k` (int, 기본 2), `--no-debounce` (flag).
- **JSONL 로그**: `debounce_votes` 필드 상시 추가 (--log 사용 시).
- **verbose 출력**: `--verbose` 시 클래스별 votes 이력 `[a,b,c]` 및 sum/N 표시.
- **이번 PR 비범위**: WebRTC NS 통합, NS on/off A/B 스크립트 (별도 후속 PR).
- **단위 테스트**: `tests/test_debounce_trigger.py`, TC-1~TC-5 (YAMNet 로드 없이 실행 가능).

상세 스펙: `docs/m1-initial-model-spec.md` (M1), `docs/m2-debounce-spec.md` (M2 Debounce)
전체 계획: `docs/development-plan.md`
M1 개선 노트: `docs/m1-improvement-notes.md`

## M1 현황 및 M2 착수 전 우선 과제

> 2026-05-11 기준. 마이크 실시간 분석 동작 확인 완료. 조용한 환경에서 전 클래스 0.0000~0.0025 수준으로 false positive 없음 (정상).

**M2 전에 반드시 처리해야 할 P1 항목:**

1. **평가 데이터셋 수집 및 검증**: ESC-50/FSD50K 위험 클래스 WAV를 `data/sample/`에 배치하고 `--threshold 0.0 --verbose`로 실제 YAMNet 점수 기록. M1 Exit Criteria(F1 ≥ 0.6) 판정 전제 조건.
2. **클래스별 임계값 근거 확보**: 단일 임계값 0.4가 모든 클래스에 적절한지 미검증. 위험 소리 입력 시 점수 분포 측정 필요.
3. **YAMNet 인덱스 최종 확인**: `development-plan.md`와 `m1-initial-model-spec.md`의 인덱스 표 불일치 — `yamnet_class_map.csv` 직접 조회로 `config/whitelist.yaml` 13개 인덱스 확정 필요.

상세 분석: `docs/m1-improvement-notes.md`

## Docker 도입 계획 (2026-05-14 결정)

- **파일 모드 한정 지원**: `--input <wav>` 및 `pytest`는 Docker로 단일 명령 실행 가능.
- **마이크 모드 비목표**: macOS/Windows는 venv 워크플로 유지. Linux는 `/dev/snd` 바인드로 실험적 가능.
- **베이스 이미지**: `python:3.11-slim` (glibc 호환성 + 최소 크기).
- **YAMNet 빌드 타임 캐시**: `ENV TFHUB_CACHE_DIR=/opt/tfhub_cache` + `RUN python scripts/verify_inference.py`로 첫 실행 시 네트워크 불필요.
- **볼륨**: `data/sample`(ro), `output`(rw), `config`(ro, 선택).
- **docker-compose.yml**: 현 단계 불필요, MQTT 등 사이드카 추가 시 재검토.

상세: `docs/docker-plan.md`
