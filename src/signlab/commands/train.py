"""Training command group."""

from signlab.commands._group import create_group

app = create_group(help_text="Run reproducible training experiments from validated configurations.")
