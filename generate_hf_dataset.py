import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

main = import_module("file_agent.hf_cli").main


if __name__ == "__main__":
    raise SystemExit(main())
