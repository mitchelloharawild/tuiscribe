"""Modal screens: source picker, speaker labels and names, note input, help."""

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
from ..diarize import LiveSpeaker
from ..diarize.cluster import SEED_THRESHOLD

NAME_SPEAKERS = "__names__"  # DiarizePicker result: open the SpeakerNamer

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


class DiarizePicker(ModalScreen[str | None]):
    """Switch speaker labels on or off for one input. Returns the channel, or None."""

    DEFAULT_CSS = MODAL_CSS.format(name="DiarizePicker")
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, channels: list[str], enabled: set[str], saved_audio: set[str], speakers: int = 0) -> None:
        super().__init__()
        self.channels = channels
        self.enabled = enabled
        self.saved_audio = saved_audio
        self.speakers = speakers  # heard so far on the enabled inputs

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Speaker labels: tell apart voices within an input")
            options = []
            for ch in self.channels:
                on = ch in self.enabled
                t = Text()
                t.append("● on   " if on else "○ off  ", "green" if on else "dim")
                t.append(CHANNEL_LABELS[ch], "bold")
                t.append("  Local 1, Local 2, ..." if ch == "mic" else "  Remote 1, Remote 2, ...", "dim")
                if ch not in self.saved_audio:
                    t.append("  audio not saved: from now only", "yellow")
                options.append(Option(t, id=ch))
            if self.speakers:
                t = Text("✎      ")
                t.append("Name speakers…", "bold")
                t.append(f"  {self.speakers} heard so far", "dim")
                options.append(Option(t, id=NAME_SPEAKERS))
            yield OptionList(*options)
            yield Static(
                "enter: switch on/off · esc: cancel\n"
                "Inputs that are on when you stop are labelled for the whole session from the saved audio.",
                classes="hint",
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class SpeakerNamer(ModalScreen[LiveSpeaker | None]):
    """Pick a live speaker to name. Returns it, or None."""

    DEFAULT_CSS = MODAL_CSS.format(name="SpeakerNamer")
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, speakers: list[LiveSpeaker], saving: bool) -> None:
        super().__init__()
        self.speakers = speakers
        self.saving = saving  # confirmed voices are saved at stop

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Name speakers")
            options = []
            for i, sp in enumerate(self.speakers):
                t = Text()
                t.append("✓ " if sp.confirmed else "  ", "green")
                t.append(f"{CHANNEL_LABELS[sp.channel]:<7}", "dim")
                t.append(f"{sp.label:<16}", "bold")
                t.append(f"{sp.seconds:>5.0f} s speech", "dim")
                if sp.suggestion and not sp.confirmed:
                    name, score = sp.suggestion
                    if name == sp.label:
                        t.append(f"  saved voice ({score:.2f}): confirm?", "cyan")
                    elif score >= SEED_THRESHOLD:
                        t.append(f"  sounds a bit like {name} ({score:.2f})", "dim")
                options.append(Option(t, id=str(i)))
            yield OptionList(*options)
            saved = "Named speakers' voices are saved when you stop, so later sessions know them. " if self.saving else ""
            yield Static(
                f"enter: name · esc: cancel\n{saved}Only name someone you are sure of.",
                classes="hint",
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.speakers[int(event.option.id)])


class SpeakerNameInput(ModalScreen[str | None]):
    """Type a speaker's name. Returns the name, "" to forget it, or None if cancelled."""

    DEFAULT_CSS = MODAL_CSS.format(name="SpeakerNameInput")
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, speaker: LiveSpeaker, known: list[str]) -> None:
        super().__init__()
        self.speaker = speaker
        self.known = known

    def compose(self) -> ComposeResult:
        sp = self.speaker
        guess = sp.label if sp.confirmed or (sp.suggestion and sp.suggestion[0] == sp.label) else ""
        with Vertical():
            yield Label(f"Who is {sp.label} ({CHANNEL_LABELS[sp.channel]})?")
            yield Input(value=guess, placeholder="their name")
            hint = "enter: save · empty: forget the name · esc: cancel"
            if self.known:
                hint += "\nSaved voices: " + ", ".join(self.known)
            yield Static(hint, classes="hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())


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
    ("d", "speaker labels on/off per input (e.g. several people on one mic); name speakers"),
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
