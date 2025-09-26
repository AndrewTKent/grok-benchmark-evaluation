from terminal_bench.terminal.tmux_session import TmuxSession

def safe_read_pane(session: TmuxSession) -> str:
    try: return session.read_pane()
    except Exception:
        try: return session.capture_pane()
        except Exception: return ""

def send_with_enter(session: TmuxSession, text: str) -> None:
    try:
        session.send_keys((text or "") + "\n")
    except TypeError:
        try:
            session.send_keys(text or "")
            session.send_keys("\n")
        except Exception:
            session.send_keys(text or "")
            session.send_keys("\r")
