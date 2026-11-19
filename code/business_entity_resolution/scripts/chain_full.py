"""Entry point for scripts/run.sh (which expects a Python file): execs the bash chain so the memory
watchdog covers the whole process tree.  Usage: run.sh chain scripts/chain_full.py [from-step]"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.execvp("bash", ["bash", os.path.join(here, "chain_full.sh")] + sys.argv[1:])
