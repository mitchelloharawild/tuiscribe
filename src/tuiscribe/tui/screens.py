"""Modal screens: source picker, note input, help."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from ..capture.devices import Device, DeviceError, list_devices
from ..config import CHANNEL_LABELS

MODAL_CSS = """
{name} {{ align: center middle; }}
{name} > Vertical {{
    width: 80%; max-width: 100; height: auto; max-height: 80%;
    border: round $accent; background: $panel; padding: 1 2;
}}
{name} Label {{ margin-bottom: 1; text-style: bold; }}
{name} .hint {{ color: $text-muted; margin-top: 1; }}
"""


class SourcePicker(ModalScreen[Device | None]):
    """Pick a PipeWire node for a channel. Returns the device, or None if cancelled."""

    DEFAULT_CSS = MODAL_CSS.format(name="SourcePicker")
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, channel: str, current: Device) -> None:
        super().__init__()
        self.channel = channel
        self.current = current
        self.kind = "source" if channel == "mic" else "sink"
        self.devices: list[Device] = []
        self.error = ""
        try:
            self.devices = [d for d in list_devices() if d.kind == self.kind]
        except DeviceError as e:
            self.error = str(e)

    def compose(self) -> ComposeResult:
        what = "microphone" if self.channel == "mic" else "output to capture (its monitor)"
        with Vertical():
            yield Label(f"{CHANNEL_LABELS[self.channel]}: choose {what}")
            if self.error:
                yield Static(Text(self.error, "red"))
            options = []
            for d in self.devices:
                t = Text()
                t.append("● " if d.name == self.current.name else "  ", "green")
                t.append(d.description, "bold" if d.is_default else "")
                t.append(f"  {d.api or ''} {d.state or ''}", "dim")
                if d.is_default:
                    t.append("  default", "cyan")
                if d.looks_like_speakers:
                    t.append("  speakers", "yellow")
                options.append(Option(t, id=str(d.id)))
            yield OptionList(*options, id="devices")
            yield Static("enter: switch · esc: cancel · the other channel keeps recording", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        for i, d in enumerate(self.devices):
            if d.name == self.current.name:
                ol.highlighted = i
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        for d in self.devices:
            if str(d.id) == event.option.id:
                self.dismiss(d)
                return
        self.dismiss(None)


class NoteInput(ModalScreen[str | None]):
    DEFAULT_CSS = MODAL_CSS.format(name="NoteInput")
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Insert a note into the transcript")
            yield Input(placeholder="e.g. ## Action items   or   Decision: ship on Friday")
            yield Static("Markdown is kept as typed · enter: insert · esc: cancel", classes="hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


HELP = [
    ("m / s", "switch the mic / system-audio source"),
    ("M / S", "mute or unmute mic / system (muted audio is not transcribed)"),
    ("space", "pause or resume transcription (meters keep running)"),
    ("n", "insert a note or heading into the transcript"),
    ("?", "this help"),
    ("q / ctrl+c", "stop: transcribe remaining speech, write footer, exit"),
]


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = MODAL_CSS.format(name="HelpScreen")
    BINDINGS = [Binding("escape,question_mark,q", "dismiss(None)", "close")]

    def compose(self) -> ComposeResult:
        t = Text()
        for key, desc in HELP:
            t.append(f"{key:<12}", "bold cyan")
            t.append(desc + "\n")
        t.append(
            "\nThe .md file only receives finished utterances; dimmed lines in the preview are "
            "live guesses that are replaced when the utterance ends. Mic = You, system audio = Remote.",
            "dim",
        )
        with Vertical():
            yield Label("tuiscribe: keys")
            yield Static(t)
            yield Static("esc: close", classes="hint")
