from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Label, Select
from rich.text import Text


class Slider(Widget, can_focus=True):
    """A minimal horizontal slider.

    Supports left/right (and up/down) arrows, home/end, and mouse
    click + drag. Emits a Slider.Changed message when its value
    changes.
    """

    value: reactive[float] = reactive(0.0)

    class Changed(Message):
        """Posted when the slider's value changes."""

        def __init__(self, slider: "Slider", value: float) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

        @property
        def control(self) -> "Slider":
            return self.slider

    def __init__(
        self,
        *,
        min: float = 0.0,
        max: float = 1.0,
        step: float = 0.01,
        value: float = 0.0,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.min = min
        self.max = max
        self.step = step
        self._grabbed = False
        self.value = self._clamp_to_step(value)

    # -- value helpers ---------------------------------------------------

    def _clamp_to_step(self, value: float) -> float:
        value = max(self.min, min(self.max, value))
        if self.step:
            steps = round((value - self.min) / self.step)
            value = self.min + steps * self.step
            value = max(self.min, min(self.max, value))
        return round(value, 6)

    def _set_value(self, value: float) -> None:
        value = self._clamp_to_step(value)
        if value != self.value:
            self.value = value

    def _set_from_x(self, x: int) -> None:
        width = max(self.size.width, 1)
        fraction = 0.0 if width <= 1 else x / (width - 1)
        fraction = max(0.0, min(1.0, fraction))
        self._set_value(self.min + fraction * (self.max - self.min))

    def watch_value(self, value: float) -> None:
        self.refresh()
        if self.is_mounted:
            self.post_message(self.Changed(self, value))

    # -- rendering -------------------------------------------------------

    def render(self) -> Text:
        width = max(self.size.width, 1)
        span = self.max - self.min
        fraction = 0.0 if span <= 0 else (self.value - self.min) / span
        fraction = max(0.0, min(1.0, fraction))
        pos = int(round(fraction * (width - 1))) if width > 1 else 0

        bar = Text()
        for i in range(width):
            if i == pos:
                bar.append("●", style="bold white")
            elif i < pos:
                bar.append("━", style="rgb(78,191,113)")
            else:
                bar.append("━", style="grey37")
        return bar

    # -- input -----------------------------------------------------------

    def on_key(self, event) -> None:
        if event.key in ("left", "down"):
            self._set_value(self.value - self.step)
            event.stop()
        elif event.key in ("right", "up"):
            self._set_value(self.value + self.step)
            event.stop()
        elif event.key == "home":
            self._set_value(self.min)
            event.stop()
        elif event.key == "end":
            self._set_value(self.max)
            event.stop()

    def on_mouse_down(self, event) -> None:
        self._grabbed = True
        self.capture_mouse()
        self._set_from_x(event.x)
        event.stop()

    def on_mouse_move(self, event) -> None:
        if self._grabbed:
            self._set_from_x(event.x)
            event.stop()

    def on_mouse_up(self, event) -> None:
        if self._grabbed:
            self._grabbed = False
            self.release_mouse()
            event.stop()


class CreateAttackScreen(ModalScreen[dict | None]):
    """Modal form for configuring a new attack.

    Dropdown options are injected via attacks / defenses so the
    UI never hardcodes capabilities. On submit the collected config dict
    is returned through dismiss; wiring it to the API is left to the
    caller. Cancelling returns None.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, attacks: list[str], defenses: list[str]) -> None:
        super().__init__()
        self.attacks = attacks
        self.defenses = defenses

    def compose(self) -> ComposeResult:
        with Vertical(id="create-attack-dialog"):
            yield Label("Create Attack", id="create-attack-title")
            with VerticalScroll(id="create-attack-form"):
                yield Label("Attack type *")
                yield Select(
                    [(a, a) for a in self.attacks],
                    id="attack_type",
                    prompt="Select attack",
                )

                yield Label("Target model *")
                yield Input(
                    placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    id="target_model",
                )

                yield Label("Poison rate *  0.10", id="poison_rate_label")
                yield Slider(min=0.0, max=1.0, step=0.01, value=0.1, id="poison_rate")

                yield Label("Dataset name *")
                yield Input(placeholder="e.g. imdb", id="dataset_name")

                yield Label("Text column *")
                yield Input(placeholder="e.g. text", id="text_column")

                yield Label("Target column *")
                yield Input(placeholder="e.g. label", id="target_column")

                yield Label("Samples (blank = full dataset)")
                yield Input(value=None, id="samples", type="integer")

                yield Label("Defense type *")
                yield Select(
                    [(d, d) for d in self.defenses],
                    id="defense_type",
                    prompt="Select defense"
                )

                with Collapsible(title="Advanced", id="advanced"):
                    yield Label("Notes")
                    yield Input(id="notes")

                    yield Label("Seed")
                    yield Input(value="0", id="seed", type="integer")

                    yield Label("Epochs")
                    yield Input(value="3", id="epochs", type="integer")

                    yield Label("Device")
                    yield Input(value="cpu", id="device")

                    yield Label("Test size")
                    yield Input(value="0.1", id="test_size")

                    yield Label("Learning rate")
                    yield Input(value="2e-5", id="learning_rate")

                    yield Label("Output dir")
                    yield Input(value="./training_output", id="output_dir")

            with Horizontal(id="create-attack-buttons"):
                yield Button("Cancel", id="cancel", variant="error")
                yield Button("Create", id="submit", variant="success")

    def on_mount(self) -> None:
        # Default the defense dropdown to "none" when it is available.
        if "none" in self.defenses:
            self.query_one("#defense_type", Select).value = "none"

    @on(Slider.Changed, "#poison_rate")
    def _update_poison_label(self, event: Slider.Changed) -> None:
        self.query_one("#poison_rate_label", Label).update(
            f"Poison rate *  {event.value:.2f}"
        )

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#submit")
    def _on_submit(self, event: Button.Pressed) -> None:
        event.stop()
        missing = self._missing_required()
        if missing:
            self.app.push_screen(ErrorDialog(missing))
            return
        self.dismiss(self._collect())

    def action_cancel(self) -> None:
        self.dismiss(None)

    # Required fields (the ones marked "*" in the form) mapped to their
    # human-readable labels for the error popup.
    REQUIRED_FIELDS = {
        "attack_type": "Attack type",
        "target_model": "Target model",
        "dataset_name": "Dataset name",
        "text_column": "Text column",
        "target_column": "Target column",
        "defense_type": "Defense type"
    }

    def _missing_required(self) -> list[str]:
        """Return the labels of any required fields that are empty/unset."""
        config = self._collect()
        missing = []
        for key, label in self.REQUIRED_FIELDS.items():
            value = config[key]
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(label)
        return missing

    def _collect(self) -> dict:
        """Gather the current form state into a config dict (UI state only)."""

        def text(widget_id: str) -> str:
            return self.query_one(widget_id, Input).value

        def integer(widget_id: str) -> int | None:
            raw = self.query_one(widget_id, Input).value.strip()
            return int(raw) if raw else None

        def selected(widget_id: str):
            widget = self.query_one(widget_id, Select)
            return None if widget.is_blank() else widget.value

        return {
            "attack_type": selected("#attack_type"),
            "target_model": text("#target_model"),
            "poison_rate": self.query_one("#poison_rate", Slider).value,
            "dataset_name": text("#dataset_name"),
            "text_column": text("#text_column"),
            "target_column": text("#target_column"),
            "defense_type": selected("#defense_type"),
            "notes": text("#notes"),
            "seed": integer("#seed"),
            "epochs": integer("#epochs"),
            "samples": integer("#samples"),
            "device": text("#device"),
            "test_size": text("#test_size"),
            "learning_rate": text("#learning_rate"),
            "output_dir": text("#output_dir"),
        }


class ErrorDialog(ModalScreen[None]):
    """Small popup that lists missing required fields."""

    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, missing: list[str]) -> None:
        super().__init__()
        self.missing = missing

    def compose(self) -> ComposeResult:
        with Vertical(id="error-dialog"):
            yield Label("Missing required fields", id="error-title")
            yield Label(
                "Please fill in the following before creating the attack:",
                id="error-subtitle",
            )
            yield Label(
                "\n".join(f"• {field}" for field in self.missing),
                id="error-list",
            )
            yield Button("OK", id="error-ok", variant="error")

    @on(Button.Pressed, "#error-ok")
    def _on_ok(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()
