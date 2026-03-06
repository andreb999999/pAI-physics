"""
Entry point for the consortium multi-agent research system.

All run logic lives in consortium/runner.py.
All LaTeX prereq logic lives in consortium/prereqs.py.
"""

import sys
from consortium.runner import main

if __name__ == "__main__":
    sys.exit(main())
