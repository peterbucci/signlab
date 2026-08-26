"""Dataset command group."""

from signlab.commands._group import create_group

app = create_group(
    help_text="Capture, import, validate, version, and split consent-approved datasets."
)
