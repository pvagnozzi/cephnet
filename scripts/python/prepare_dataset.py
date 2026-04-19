# prepare_dataset.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Entry-point for dataset preparation — delegates to cephnet CLI

import sys

from cephnet.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["prepare", *sys.argv[1:]], standalone_mode=True))
