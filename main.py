import os
import sys
import time
import shutil
import platform
import subprocess
import threading
import itertools


# ──────────────────────────────────────────────────────────────────────────
#  COLORS  (force-enabled — including legacy Windows cmd via WinAPI)
# ──────────────────────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    WHITE = "\033[97m"

    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"


def enable_windows_ansi():
    """Force-enable ANSI escape processing on Windows cmd.exe / PowerShell."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


enable_windows_ansi()

# Only disable color if explicitly requested or output is being piped to a file.
USE_COLOR = not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def c(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


def width() -> int:
    return min(shutil.get_terminal_size((70, 20)).columns, 64)


def hr(char: str = "─", color: str = C.DIM) -> str:
    return c(char * width(), color)


def center(text: str) -> str:
    return text.center(width())


# ──────────────────────────────────────────────────────────────────────────
#  PROJECT METADATA
# ──────────────────────────────────────────────────────────────────────────
PROJECT_NAME = "LEXIS PLATFORM"
TAGLINE = "AI-Powered Document Intelligence"
OWNER = "Ruturaj Mankapure"
GITHUB = "https://github.com/ruturaj-018"
VERSION = "v1.0.0"

VENV_DIR = "venv"
APP_FILE = "app.py"

# ──────────────────────────────────────────────────────────────────────────
#  DISPLAY TIMING  — set how long the intro should take to unfold
# ──────────────────────────────────────────────────────────────────────────
# Change this single number to switch the default: 60 = 1 minute, 120 = 2 minutes.
DEFAULT_INTRO_SECONDS = 4
# Total seconds the banner/credits/features/system-info intro should take.
# Set to 0 for instant (no delay). Examples: 60 = 1 minute, 120 = 2 minutes.
#
# Override without editing this file:
#   python main.py --intro 60        (1 minute)
#   python main.py --intro 120       (2 minutes)
#   set LEXIS_INTRO_SECONDS=90  &&  python main.py     (Windows)
#   LEXIS_INTRO_SECONDS=90 python main.py              (macOS/Linux)
def _resolve_intro_duration() -> float:
    if "--intro" in sys.argv:
        idx = sys.argv.index("--intro")
        if idx + 1 < len(sys.argv):
            try:
                return float(sys.argv[idx + 1])
            except ValueError:
                pass
    env_val = os.environ.get("LEXIS_INTRO_SECONDS")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return DEFAULT_INTRO_SECONDS


INTRO_DURATION_SECONDS = _resolve_intro_duration()

# Internally we spread INTRO_DURATION_SECONDS across this many "beats"
# (banner lines + credit lines + feature rows + system info rows).
def _line_delay(num_lines: int) -> float:
    if INTRO_DURATION_SECONDS <= 0 or num_lines <= 0:
        return 0.0
    return INTRO_DURATION_SECONDS / num_lines


def slow_print(text: str, delay: float):
    """Print a line, then pause. Pause is skipped if delay is 0."""
    print(text)
    if delay > 0:
        time.sleep(delay)


def venv_paths():
    """Return (python_exe, streamlit_exe, pip_exe) inside the venv, OS-aware."""
    if platform.system() == "Windows":
        bin_dir = os.path.join(VENV_DIR, "Scripts")
        py = os.path.join(bin_dir, "python.exe")
        st = os.path.join(bin_dir, "streamlit.exe")
        pip = os.path.join(bin_dir, "pip.exe")
    else:
        bin_dir = os.path.join(VENV_DIR, "bin")
        py = os.path.join(bin_dir, "python")
        st = os.path.join(bin_dir, "streamlit")
        pip = os.path.join(bin_dir, "pip")
    return py, st, pip


# ──────────────────────────────────────────────────────────────────────────
#  SPINNER / LOADER
# ──────────────────────────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str, color: str = C.CYAN):
        self.message = message
        self.color = color
        self._stop_event = threading.Event()
        self._thread = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r  {c(frame, self.color, C.BOLD)} {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)

    def start(self):
        if not USE_COLOR:
            print(f"  ... {self.message}")
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, success: bool = True, final_message: str = None):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        icon = c("✓", C.GREEN, C.BOLD) if success else c("✗", C.RED, C.BOLD)
        msg = final_message or self.message
        sys.stdout.write(f"\r  {icon} {msg}" + " " * 12 + "\n")
        sys.stdout.flush()


def loading_bar(label: str, duration: float = 0.8, color: str = C.BRIGHT_CYAN):
    bar_width = 28
    steps = 24
    for i in range(steps + 1):
        filled = int(bar_width * i / steps)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = int(100 * i / steps)
        sys.stdout.write(f"\r  {label}  {c(bar, color)} {pct:>3}%")
        sys.stdout.flush()
        time.sleep(duration / steps)
    print()


# ──────────────────────────────────────────────────────────────────────────
#  UI SECTIONS
# ──────────────────────────────────────────────────────────────────────────
def print_banner():
    w = width()
    lines = [
        "",
        c("╔" + "═" * w + "╗", C.BRIGHT_CYAN),
        c("║", C.BRIGHT_CYAN) + c(center(PROJECT_NAME), C.BOLD, C.WHITE) + c("║", C.BRIGHT_CYAN),
        c("║", C.BRIGHT_CYAN) + c(center(TAGLINE), C.DIM, C.CYAN) + c("║", C.BRIGHT_CYAN),
        c("║", C.BRIGHT_CYAN) + center("").rjust(w) + c("║", C.BRIGHT_CYAN),
        c("║", C.BRIGHT_CYAN) + c(center(VERSION), C.ITALIC, C.DIM) + c("║", C.BRIGHT_CYAN),
        c("╚" + "═" * w + "╝", C.BRIGHT_CYAN),
        "",
    ]
    delay = _line_delay(len(lines))
    for line in lines:
        slow_print(line, delay)


def print_credits():
    lines = [
        c("  CREATED BY", C.BOLD, C.BRIGHT_MAGENTA),
        " " + hr(),
        f"  {c('Author', C.DIM):<14}{c(OWNER, C.BRIGHT_YELLOW, C.BOLD)}",
        f"  {c('GitHub', C.DIM):<14}{c(GITHUB, C.BRIGHT_BLUE, C.UNDERLINE)}",
        "",
    ]
    delay = _line_delay(len(lines))
    for line in lines:
        slow_print(line, delay)


def print_features():
    features = [
        ("PDF Analysis", "Extract and analyze PDF documents"),
        ("DOCX Analysis", "Parse and process Word documents"),
        ("Semantic Search", "Vector-based contextual retrieval"),
        ("RAG Pipeline", "Retrieval-augmented generation"),
        ("Multi-Provider AI", "OpenAI, Anthropic, and more"),
        ("Ollama Local Models", "Run models fully offline"),
    ]
    header_lines = [c("  FEATURES", C.BOLD, C.BRIGHT_MAGENTA), " " + hr()]
    delay = _line_delay(len(header_lines) + len(features) + 1)
    for line in header_lines:
        slow_print(line, delay)
    for name, desc in features:
        check = c("✓", C.GREEN, C.BOLD)
        slow_print(f"  {check} {c(name, C.WHITE, C.BOLD):<28} {c(desc, C.DIM)}", delay)
    slow_print("", delay)


def print_system_info():
    info = [
        ("Python", platform.python_version()),
        ("Platform", platform.system()),
        ("Working dir", os.getcwd()),
    ]
    header_lines = [c("  SYSTEM", C.BOLD, C.BRIGHT_MAGENTA), " " + hr()]
    delay = _line_delay(len(header_lines) + len(info) + 1)
    for line in header_lines:
        slow_print(line, delay)
    for label, value in info:
        slow_print(f"  {c(label + ':', C.DIM):<22}{c(value, C.YELLOW)}", delay)
    slow_print("", delay)


# ──────────────────────────────────────────────────────────────────────────
#  VENV LIFECYCLE  (create → install → launch, no shell "activate" needed)
# ──────────────────────────────────────────────────────────────────────────
def ensure_venv() -> bool:
    """Create the venv if missing. Returns True on success."""
    py, _, _ = venv_paths()

    if os.path.exists(py):
        print(f"  {c('✓', C.GREEN, C.BOLD)} Virtual environment found ({VENV_DIR}/)")
        return True

    spinner = Spinner(f"Creating virtual environment in '{VENV_DIR}' ...", color=C.MAGENTA).start()
    try:
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        spinner.stop(final_message=f"Virtual environment created ({VENV_DIR}/)")
        return True
    except subprocess.CalledProcessError:
        spinner.stop(success=False, final_message="Failed to create virtual environment")
        return False


def ensure_streamlit() -> bool:
    """Install streamlit into the venv if not already present."""
    _, st, pip = venv_paths()

    if os.path.exists(st):
        print(f"  {c('✓', C.GREEN, C.BOLD)} Streamlit already installed in venv")
        return True

    spinner = Spinner("Installing Streamlit into venv (first run) ...", color=C.BLUE).start()
    try:
        subprocess.run([pip, "install", "--quiet", "streamlit"], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        spinner.stop(final_message="Streamlit installed")
        return True
    except subprocess.CalledProcessError:
        spinner.stop(success=False, final_message="Failed to install Streamlit")
        print(c(f"\n  Try manually: {pip} install streamlit\n", C.YELLOW))
        return False


def check_app_file() -> bool:
    if os.path.exists(APP_FILE):
        print(f"  {c('✓', C.GREEN, C.BOLD)} {APP_FILE} found")
        return True
    print(f"  {c('✗', C.RED, C.BOLD)} {APP_FILE} not found in current directory")
    return False


# ──────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    os.system("cls" if platform.system() == "Windows" else "clear")

    print_banner()
    print_credits()
    print_features()
    print_system_info()

    print(c("  STARTUP SEQUENCE", C.BOLD, C.BRIGHT_MAGENTA))
    print(" ", hr())

    if not check_app_file():
        print(c(f"\n  Aborting: cannot launch without {APP_FILE}\n", C.RED))
        sys.exit(1)

    if not ensure_venv():
        sys.exit(1)

    if not ensure_streamlit():
        sys.exit(1)

    print()
    print(c("  LAUNCHING", C.BOLD, C.BRIGHT_MAGENTA))
    print(" ", hr())
    loading_bar("Booting Lexis", duration=0.9)
    print()
    print(c(center("🚀  Launching Streamlit application  🚀"), C.BOLD, C.BRIGHT_GREEN))
    print(hr("═", C.BRIGHT_CYAN))
    print()

    # Run streamlit directly from the venv's interpreter — equivalent to
    # activating the venv and running `streamlit run app.py`, but works
    # cross-shell without needing a separate "activate" step.
    _, streamlit_exe, _ = venv_paths()
    subprocess.run([streamlit_exe, "run", APP_FILE])


if __name__ == "__main__":
    main()