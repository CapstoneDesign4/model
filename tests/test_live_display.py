"""LiveDisplay 헬퍼 단위 테스트.

YAMNet 로딩 없이 실행 가능. ANSI 모드와 fallback 모드를 모두 검증한다.

실행: pytest tests/test_live_display.py -v
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest

from src.cli import LiveDisplay, _ansi_supported, _fmt_ts


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처 / 유틸
# ─────────────────────────────────────────────────────────────────────────────

ALL_KEYS = [
    "screaming", "baby_cry", "glass_shatter", "breaking",
    "gunshot", "explosion", "fire_alarm", "smoke_alarm",
    "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
]

TIMESTAMP = 1_700_000_000.0  # 고정 에포크 (2023-11-14 계열, 테스트 결정론)


def _make_scores(high: list[str] | None = None) -> dict[str, float]:
    high = high or []
    return {k: (0.8 if k in high else 0.05) for k in ALL_KEYS}


def _make_mock_trigger(debounce_enabled: bool = True) -> MagicMock:
    """Trigger 를 모방하는 Mock. get_debounce_state() 가 DebounceState-like 객체를 반환."""
    from src.postprocess.trigger import DebounceState

    mock_trigger = MagicMock()

    def _get_state(key: str) -> DebounceState:
        state = DebounceState(window=3, k=2)
        state.push(1)
        return state

    mock_trigger.get_debounce_state.side_effect = _get_state
    return mock_trigger


def _make_trigger_event(key: str = "screaming", score: float = 0.9) -> object:
    """TriggerEvent 를 모방하는 간단한 네임드튜플 대체 객체."""
    from src.postprocess.trigger import TriggerEvent
    return TriggerEvent(
        key=key,
        display_name=key,
        score=score,
        timestamp=TIMESTAMP,
        debounce_votes=[1, 1],
    )


# ─────────────────────────────────────────────────────────────────────────────
# _ansi_supported 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestAnsiSupported:
    """_ansi_supported() 의 판정 로직을 검증한다."""

    def test_returns_false_when_not_tty(self):
        """stdout 이 TTY 가 아니면 False 를 반환해야 한다."""
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            result = _ansi_supported()
        assert result is False

    def test_returns_false_when_no_color_set(self):
        """NO_COLOR 환경변수가 설정되면 TTY 여도 False 를 반환해야 한다."""
        with patch("sys.stdout") as mock_stdout, \
             patch.dict(os.environ, {"NO_COLOR": "1"}):
            mock_stdout.isatty.return_value = True
            result = _ansi_supported()
        assert result is False

    def test_returns_true_when_tty_and_no_no_color(self):
        """TTY 이고 NO_COLOR 미설정이면 True 를 반환해야 한다."""
        env_without_no_color = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        with patch("sys.stdout") as mock_stdout, \
             patch.dict(os.environ, env_without_no_color, clear=True), \
             patch("os.system"):  # Windows os.system("") 방어
            mock_stdout.isatty.return_value = True
            result = _ansi_supported()
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# fallback 모드 (non-TTY) 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveDisplayFallback:
    """isatty=False 일 때 LiveDisplay 가 기존 print 방식으로 fallback 해야 한다."""

    @pytest.fixture
    def display_non_verbose(self):
        """non-verbose fallback LiveDisplay."""
        with patch("src.cli._ansi_supported", return_value=False):
            d = LiveDisplay(verbose=False)
        assert d._ansi is False
        return d

    @pytest.fixture
    def display_verbose(self):
        """verbose fallback LiveDisplay."""
        with patch("src.cli._ansi_supported", return_value=False):
            d = LiveDisplay(verbose=True)
        assert d._ansi is False
        return d

    def test_update_no_danger_prints_header_and_all_classes(self, display_non_verbose, capsys):
        """fallback non-verbose: update_no_danger 가 헤더 + 12종 클래스를 출력해야 한다."""
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()
        display_non_verbose.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
        captured = capsys.readouterr()
        # 헤더 검증
        assert "WINDOW scores" in captured.out
        # 전체 12종 클래스가 출력되었는지 검증
        for key in ALL_KEYS:
            assert key in captured.out

    def test_update_no_danger_prints_scores(self, display_non_verbose, capsys):
        """fallback non-verbose: update_no_danger 출력에 점수가 포함되어야 한다."""
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()
        display_non_verbose.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
        captured = capsys.readouterr()
        assert "0.8" in captured.out

    def test_update_no_danger_does_not_include_debounce_info(self, display_non_verbose, capsys):
        """fallback non-verbose: debounce_enabled=False 이면 votes 정보가 없어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        display_non_verbose.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
        captured = capsys.readouterr()
        assert "votes=" not in captured.out

    def test_update_no_danger_all_12_classes_output(self, display_non_verbose, capsys):
        """fallback non-verbose: 12종 클래스 모두 한 번씩 출력되어야 한다 (fallback TTY=False 케이스)."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        display_non_verbose.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
        captured = capsys.readouterr()
        # 헤더 포함 총 13줄 이상 출력되어야 함
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) >= 1 + len(ALL_KEYS)

    def test_update_verbose_prints_header_and_classes(self, display_verbose, capsys):
        """fallback verbose: update_verbose 가 헤더 + 클래스 줄을 출력해야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        display_verbose.update_verbose(TIMESTAMP, scores, trigger, debounce_enabled=True)
        captured = capsys.readouterr()
        assert "WINDOW scores" in captured.out
        assert "screaming" in captured.out

    def test_update_verbose_includes_debounce_info(self, display_verbose, capsys):
        """fallback verbose: debounce_enabled=True 이면 votes 정보가 포함되어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        display_verbose.update_verbose(TIMESTAMP, scores, trigger, debounce_enabled=True)
        captured = capsys.readouterr()
        assert "votes=" in captured.out
        assert "sum=" in captured.out

    def test_emit_danger_prints_danger_line(self, display_non_verbose, capsys):
        """fallback: emit_danger 가 DANGER 줄을 출력해야 한다."""
        event = _make_trigger_event()
        display_non_verbose.emit_danger(event)
        captured = capsys.readouterr()
        assert "DANGER" in captured.out
        assert "screaming" in captured.out

    def test_finalize_no_error(self, display_non_verbose):
        """fallback: finalize() 가 예외 없이 실행되어야 한다."""
        display_non_verbose.finalize()  # 예외 없으면 통과

    def test_fallback_does_not_write_ansi_sequences(self, display_non_verbose, capsys):
        """fallback 모드에서는 ESC 문자가 출력에 포함되지 않아야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        display_non_verbose.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
        captured = capsys.readouterr()
        assert "\x1b" not in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# ANSI 모드 테스트 (stdout 을 StringIO 로 대체)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveDisplayAnsi:
    """ANSI 모드에서 올바른 escape sequence 가 stdout 에 기록되는지 검증한다."""

    @pytest.fixture
    def buf_display_non_verbose(self):
        """non-verbose ANSI LiveDisplay + 캡처용 버퍼."""
        buf = io.StringIO()
        with patch("src.cli._ansi_supported", return_value=True):
            d = LiveDisplay(verbose=False)
        assert d._ansi is True
        return d, buf

    @pytest.fixture
    def buf_display_verbose(self):
        """verbose ANSI LiveDisplay + 캡처용 버퍼."""
        buf = io.StringIO()
        with patch("src.cli._ansi_supported", return_value=True):
            d = LiveDisplay(verbose=True)
        assert d._ansi is True
        return d, buf

    # ---------- non-verbose ----------

    def test_update_no_danger_writes_move_up_on_second_call(self, buf_display_non_verbose):
        """ANSI non-verbose: 두 번째 update_no_danger 호출에서 \\x1b[<N>A escape 가 포함되어야 한다."""
        display, _buf = buf_display_non_verbose
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
            written = mock_out.getvalue()

        # 두 번째 호출: 이전 블록 위치로 이동하는 MOVE_UP escape 포함
        assert "\x1b[" in written
        assert "A" in written

    def test_update_no_danger_contains_all_classes(self, buf_display_non_verbose):
        """ANSI non-verbose: 출력에 12종 클래스 이름과 점수가 모두 포함되어야 한다."""
        display, _buf = buf_display_non_verbose
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
            written = mock_out.getvalue()

        for key in ALL_KEYS:
            assert key in written
        assert "WINDOW scores" in written

    def test_update_no_danger_no_debounce_info(self, buf_display_non_verbose):
        """ANSI non-verbose: debounce_enabled=False 이면 votes 정보가 없어야 한다."""
        display, _buf = buf_display_non_verbose
        scores = _make_scores()
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)
            written = mock_out.getvalue()

        assert "votes=" not in written

    def test_block_lines_set_after_first_update_no_danger(self, buf_display_non_verbose):
        """ANSI non-verbose: 첫 update_no_danger 호출 후 _block_lines 가 헤더+12 = 13 이어야 한다."""
        display, _buf = buf_display_non_verbose
        scores = _make_scores()
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update_no_danger(TIMESTAMP, scores, trigger, debounce_enabled=False)

        assert display._block_lines == 1 + len(ALL_KEYS)

    def test_emit_danger_non_verbose_resets_block_lines(self, buf_display_non_verbose):
        """ANSI non-verbose: emit_danger 후 _block_lines 가 0으로 리셋되어야 한다."""
        display, _buf = buf_display_non_verbose
        display._block_lines = 13
        event = _make_trigger_event()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.emit_danger(event)

        assert display._block_lines == 0

    def test_emit_danger_non_verbose_writes_danger_line(self, buf_display_non_verbose):
        """ANSI non-verbose: emit_danger 가 DANGER 키워드를 포함한 줄을 출력해야 한다."""
        display, _buf = buf_display_non_verbose
        display._block_lines = 13
        event = _make_trigger_event(key="screaming", score=0.9)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.emit_danger(event)
            written = mock_out.getvalue()

        assert "DANGER" in written
        assert "screaming" in written

    def test_emit_danger_no_block_yet_just_prints(self, buf_display_non_verbose):
        """ANSI non-verbose: 블록 미출력(_block_lines=0) 상태에서 emit_danger 는 단순 인쇄."""
        display, _buf = buf_display_non_verbose
        assert display._block_lines == 0
        event = _make_trigger_event()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.emit_danger(event)
            written = mock_out.getvalue()

        assert "DANGER" in written
        # 블록 지우기 escape (\x1b[2K 반복) 는 없어야 함
        assert "\x1b[2K" not in written

    # ---------- verbose ----------

    def test_verbose_first_call_sets_block_lines(self, buf_display_verbose):
        """ANSI verbose: 첫 update_verbose 호출 후 _block_lines 가 줄 수를 기억해야 한다."""
        display, _buf = buf_display_verbose
        scores = _make_scores()
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update_verbose(TIMESTAMP, scores, trigger, debounce_enabled=True)

        # 헤더 1줄 + 클래스 12줄 = 13줄
        assert display._block_lines == 1 + len(ALL_KEYS)

    def test_verbose_second_call_moves_cursor_up(self, buf_display_verbose):
        """ANSI verbose: 두 번째 호출에서 \\x1b[<N>A escape 가 포함되어야 한다."""
        display, _buf = buf_display_verbose
        scores = _make_scores()
        trigger = _make_mock_trigger()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update_verbose(TIMESTAMP, scores, trigger, debounce_enabled=True)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out2:
            display.update_verbose(TIMESTAMP, scores, trigger, debounce_enabled=True)
            written2 = mock_out2.getvalue()

        # 이전 블록 줄 수만큼 위로 이동하는 escape 포함
        assert "\x1b[" in written2
        assert "A" in written2  # MOVE_UP 의 A 방향

    def test_verbose_emit_danger_resets_block_lines(self, buf_display_verbose):
        """ANSI verbose: emit_danger 후 _block_lines 가 0으로 리셋되어야 한다."""
        display, _buf = buf_display_verbose
        # 블록이 이미 출력된 상태 가정
        display._block_lines = 13
        event = _make_trigger_event()

        with patch("sys.stdout", new_callable=io.StringIO):
            display.emit_danger(event)

        assert display._block_lines == 0

    def test_verbose_emit_danger_writes_danger_line(self, buf_display_verbose):
        """ANSI verbose: emit_danger 가 DANGER 키워드를 포함한 줄을 출력해야 한다."""
        display, _buf = buf_display_verbose
        display._block_lines = 13
        event = _make_trigger_event(key="gunshot", score=0.95)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.emit_danger(event)
            written = mock_out.getvalue()

        assert "DANGER" in written
        assert "gunshot" in written

    def test_verbose_emit_danger_no_block_yet_just_prints(self, buf_display_verbose):
        """ANSI verbose: 블록이 미출력 상태(_block_lines=0)에서 emit_danger 는 단순 인쇄."""
        display, _buf = buf_display_verbose
        assert display._block_lines == 0
        event = _make_trigger_event()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.emit_danger(event)
            written = mock_out.getvalue()

        assert "DANGER" in written
        # 블록 지우기 escape (\x1b[2K 반복) 는 없어야 함
        assert "\x1b[2K" not in written


# ─────────────────────────────────────────────────────────────────────────────
# _build_block_lines 정적 메서드 (순수 문자열 생성)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBlockLines:
    """_build_block_lines 가 올바른 형식의 블록을 생성하는지 검증한다."""

    def test_contains_header(self):
        """헤더 줄이 블록 첫 줄에 있어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=False)
        assert "WINDOW scores" in lines[0]

    def test_contains_all_class_keys(self):
        """12종 클래스가 모두 블록에 포함되어야 한다."""
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=False)
        block_text = "\n".join(lines)
        for key in ALL_KEYS:
            assert key in block_text

    def test_total_line_count(self):
        """헤더 1줄 + 클래스 12줄 = 13줄이어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=False)
        assert len(lines) == 1 + len(ALL_KEYS)

    def test_non_verbose_no_debounce_info(self):
        """debounce_enabled=False 이면 votes 정보가 없어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=False)
        block_text = "\n".join(lines)
        assert "votes=" not in block_text

    def test_verbose_includes_debounce_info(self):
        """debounce_enabled=True 이면 votes 정보가 포함되어야 한다."""
        scores = _make_scores()
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=True)
        block_text = "\n".join(lines)
        assert "votes=" in block_text
        assert "sum=" in block_text

    def test_score_format(self):
        """클래스 줄에 4자리 소수점 점수가 포함되어야 한다."""
        scores = _make_scores(["screaming"])
        trigger = _make_mock_trigger()
        lines = LiveDisplay._build_block_lines(TIMESTAMP, scores, trigger, debounce_enabled=False)
        screaming_line = next(ln for ln in lines if "screaming" in ln)
        assert "0.8000" in screaming_line


# ─────────────────────────────────────────────────────────────────────────────
# finalize 동작
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalize:
    """finalize() 가 커서를 올바르게 처리하는지 검증한다."""

    def test_finalize_ansi_with_block(self):
        """ANSI: 블록이 있으면(_block_lines > 0) finalize 가 개행을 써야 한다."""
        with patch("src.cli._ansi_supported", return_value=True):
            display = LiveDisplay(verbose=False)
        display._block_lines = 13

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.finalize()
            written = mock_out.getvalue()

        assert "\n" in written

    def test_finalize_ansi_verbose_with_block(self):
        """ANSI verbose: 블록이 있으면 finalize 가 개행을 써야 한다."""
        with patch("src.cli._ansi_supported", return_value=True):
            display = LiveDisplay(verbose=True)
        display._block_lines = 5

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.finalize()
            written = mock_out.getvalue()

        assert "\n" in written

    def test_finalize_ansi_no_block_no_output(self):
        """ANSI: 블록이 없으면(_block_lines == 0) finalize 가 아무것도 출력하지 않아야 한다."""
        with patch("src.cli._ansi_supported", return_value=True):
            display = LiveDisplay(verbose=False)
        assert display._block_lines == 0

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            display.finalize()
            written = mock_out.getvalue()

        assert written == ""

    def test_finalize_fallback_no_error(self):
        """fallback 모드: finalize() 가 아무것도 하지 않고 예외 없이 종료되어야 한다."""
        with patch("src.cli._ansi_supported", return_value=False):
            display = LiveDisplay(verbose=False)
        display.finalize()  # 예외 없으면 통과
