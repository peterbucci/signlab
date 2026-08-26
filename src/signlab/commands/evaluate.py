"""Evaluation command group."""

from signlab.commands._group import create_group

app = create_group(help_text="Evaluate checkpoints on locked clips and continuous replay sessions.")
