from textual.app import App, ComposeResult, on
from textual.widgets import Header, Footer, DataTable, Button, Static, Label
from textual.containers import Horizontal, Vertical
import httpx

from create_attack import CreateAttackScreen

API_BASE = "http://localhost:8000"


class RunsTable(Static):
    """Widget that displays a DataTable containing data on all runs."""

    COLUMNS = (
        "ID",
        "Attack",
        "Poison Rate",
        "Initial Acc.",
        "Post-Attack Acc.",
        "Post-Defense Acc.",
        "Degradation",
        "Status",
    )

    def compose(self) -> ComposeResult:
        yield Label("Runs", classes="section-title")
        yield DataTable(id="runs-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(*self.COLUMNS)
        self.refresh_runs()

    def craft_row(self, run: dict) -> list:
        """
        Crafts a row corresponding to the format of the runs table

        Parses a RunResult model to extract required data for the
        runs table.

        Args:
            run (dict): A dict representing a RunResult model

        Returns:
            list: A list formatted as a row in the runs table column
            with values parsed from the given model instance.
        """
        return [
            run["run_id"],
            run["config"]["attack_type"],
            run["config"]["poison_rate"],
            run["baseline_metrics"]["accuracy"] if run["baseline_metrics"] else "-",
            run["post_attack_metrics"]["accuracy"] if run["post_attack_metrics"] else "-",
            run["post_defense_metrics"]["accuracy"] if run["post_defense_metrics"] else "-",
            run.get("degradation", "-"),
            run["status"],
        ]

    def refresh_runs(self) -> None:
        """
        Refreshes the runs table

        Sends a GET request to the API at /runs to obtain a list
        of available runs. Then parses each run using the craft_row
        function and then adds each row to the table.
        """
        table = self.query_one(DataTable)
        table.clear()
        try:
            runs = httpx.get(f"{API_BASE}/runs").json()
            for run in runs:
                table.add_row(*self.craft_row(run))
        except httpx.RequestError as e:
            table.add_row("ERROR", str(e), "", "", "", "", "", "")


class NegevApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "Negev"
    SUB_TITLE = "LLM Data Poisoning Attack Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(classes="sidebar"):
                yield Label("Controls", classes="section-title")
                yield Button("Health Check", id="get_health", variant="primary")
                yield Label("status: —", id="health_label", classes="status")
                yield Button("Create Attack", id="create_attack", variant="success")
            with Vertical(classes="main-pane"):
                yield RunsTable()
        yield Footer()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        self.process_button_function(event.button.id)

    def process_button_function(self, button_id: str) -> None:
        match button_id:
            case "get_health":
                label = self.query_one("#health_label", Label)
                label.update("status: requesting...")
                response = self.get_response("health")
                label.update(f"status: {response}")

            case "create_attack":
                self.open_create_attack()

    def open_create_attack(self) -> None:
        """Open the Create Attack modal, populating dropdowns from /capabilities."""
        try:
            caps = self.get_response("capabilities")
            attacks = caps.get("attacks", [])
            defenses = caps.get("defenses", [])
        except Exception:
            attacks, defenses = [], []
        self.push_screen(
            CreateAttackScreen(attacks, defenses), self.on_attack_submitted
        )

    def on_attack_submitted(self, config: dict | None) -> None:
        """Callback for the Create Attack modal"""
        if config is None:
            return
    
        try:
            resp = httpx.post(f"{API_BASE}/runs", json=config)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self.notify(f"Config Error: {e.response.json()}", severity="error")
        except httpx.RequestError as e:
            self.notify(f"Network Error: {e}", severity="error")

        self.action_refresh()

    def get_response(self, param: str):
        return httpx.get(f"{API_BASE}/{param}").json()

    def action_refresh(self) -> None:
        self.query_one(RunsTable).refresh_runs()


if __name__ == "__main__":
    NegevApp().run()
