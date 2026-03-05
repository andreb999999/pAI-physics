"""
Entry point for the freephdlabor multi-agent research system.

All run logic lives in freephdlabor/runner.py.
All LaTeX prereq logic lives in freephdlabor/prereqs.py.
"""

import sys
from freephdlabor.runner import main

if __name__ == "__main__":
    sys.exit(main())
