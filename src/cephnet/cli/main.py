# main.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Root Click group for the cephnet CLI

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="cephnet")
def main() -> None:
    """🧠 cephnet — Cephalometric landmark detection training subsystem."""


from cephnet.cli.commands import prepare, train, validate, verify  # noqa: E402

main.add_command(prepare)
main.add_command(train)
main.add_command(validate)
main.add_command(verify)
