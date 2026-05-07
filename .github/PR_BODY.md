## 요약
- YAMNet 기반 위험 소리 감지 M1 베이스라인 구현 (12종 화이트리스트, 임계값 + cooldown)
- 노트북 마이크 실시간 분석 퀵스타트 가이드 추가
- 브랜치/커밋/PR을 위임할 `pr-manager` 서브에이전트 정의 추가

## 변경 내용

### 코드 (M1 베이스라인)
- `src/model/yamnet_wrapper.py` — TF-Hub YAMNet 로딩 및 추론 래퍼
- `src/model/danger_filter.py` — 화이트리스트 12종 score 추출 (Glass/Shatter는 max로 통합)
- `src/postprocess/trigger.py` — 클래스별 임계값 + 5초 cooldown
- `src/audio_io/file_reader.py`, `mic_stream.py` — 0.96s 윈도우 / 0.48s hop 슬라이딩
- `src/cli.py` — `--input file|mic`, `--threshold`, `--log` 옵션 + JSONL 로그
- `src/preprocess/`, `src/embedded/` — M2/M5 플레이스홀더 (TODO 주석)

### 설정·도구
- `requirements.txt` — TensorFlow 2.13~2.16, tensorflow-hub, librosa, sounddevice 등
- `config/whitelist.yaml` — 12종 클래스 인덱스/임계값/cooldown
- `.gitignore` — `.venv/`, `__pycache__/`, `output/`, `data/raw/` 등
- `scripts/verify_inference.py` — YAMNet 로드 + 더미 추론 환경 검증
- `tests/test_yamnet_load.py` — pytest (네트워크 없으면 skip)

### 문서
- `CLAUDE.md` — 시작하기/명령어/아키텍처 섹션 갱신
- `docs/m1-initial-model-spec.md` — M1 상세 스펙 (입력/모델/출력/CLI/체크리스트)
- `docs/mic-quickstart.md` — 노트북 마이크 실시간 분석 절차 + 트러블슈팅

### 에이전트
- `.claude/agents/pr-manager.md` — PR 매니저 서브에이전트

## 검증

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verify_inference.py     # PASS 필요
pytest tests/ -v                       # 단위 테스트
python -m src.cli --input mic --threshold 0.4 --verbose
```

- `verify_inference.py`로 환경 검증 후 마이크 모드 실시간 동작 가능
- 시연용으로 YouTube의 사이렌·화재경보·유리 깨짐 SFX를 재생해 트리거 확인 권장

## 관련 문서
- `docs/development-plan.md` — 전체 개발 계획·마일스톤
- `docs/m1-initial-model-spec.md` — M1 상세 스펙
- `docs/mic-quickstart.md` — 마이크 실행 가이드

## 다음 단계 (M2 이후)
- 노이즈 캔슬링 전처리 통합 (WebRTC NS)
- K/N 다수결 debounce 후처리
- ESC-50/UrbanSound8K 정량 평가 파이프라인
- M3: YAMNet embedding + 경량 분류 헤드 파인튜닝
- M5: UART JSON 라인 임베디드 통신

🤖 Generated with [Claude Code](https://claude.com/claude-code)
