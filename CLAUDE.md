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
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\Activate.ps1

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

상세 스펙: `docs/m1-initial-model-spec.md`
전체 계획: `docs/development-plan.md`
