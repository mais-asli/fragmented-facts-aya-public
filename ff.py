"""Run the project without an editable install: python ff.py --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fragmented_facts.cli import main

if __name__ == "__main__":
    main()
