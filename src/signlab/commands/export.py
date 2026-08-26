"""Model-export command group."""

from signlab.commands._group import create_group

app = create_group(help_text="Export candidate checkpoints into immutable portable bundles.")
