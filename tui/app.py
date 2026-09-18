from textual.app import App, ComposeResult, on
from textual.widgets import Header, Footer, DataTable, Button, Static, Label
from textual.containers import Horizontal, Vertical
from textual import work
import httpx

from create_attack import CreateAttackScreen

API_BASE = "http://localhost:8000"


class NegevClient:
    """Centralized async HTTP client for talking with the Negev API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url)


    async def get(self, path: str) -> dict:
        response = await self._client.get(path)
        response.raise_for_status()
        return response.json()


    async def post(self, path: str, json: dict) -> dict:
        response = await self._client.post(path, json=json)
        response.raise_for_status()
        return response.json()


    async def aclose(self) -> None:
        await self._client.aclose()


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


    def __init__(self) -> None:
        super().__init__()
        self._notified_failures: set[str] = set()


    def compose(self) -> ComposeResult:
        yield Label("Runs", classes="section-title")
        yield DataTable(id="runs-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(*self.COLUMNS)
        self.refresh_runs()
        self.set_interval(5, self.refresh_runs)

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

    @work(exclusive=True)
    async def refresh_runs(self) -> None:
        """
        Refreshes the runs table

        Sends a GET request to the API at /runs to obtain a list
        of available runs. Then parses each run using the craft_row
        function and then adds each row to the table.
        """
        table = self.query_one(DataTable)
        try:
            runs = await self.app.client.get("/runs")
            table.clear()
            for run in runs:
                table.add_row(*self.craft_row(run))
                if run['status'] == "failed" and run['run_id'] not in self._notified_failures:
                    self._notified_failures.add(run['run_id'])
                    self.notify(f"Run {run['run_id']} failed: {run.get('error_message', 'unknown error')}", severity="error", timeout=5)

        except httpx.HTTPError as e:
            self.notify(f"Error connecting to server: {e}", severity="error")


class NegevApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "Negev"
    SUB_TITLE = "LLM Data Poisoning Attack Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]


    def __init__(self) -> None:
        super().__init__()
        self.client = NegevClient(API_BASE)

    
    async def on_unmount(self) -> None:
        await self.client.aclose()


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


    @on(Button.Pressed, "#get_health")
    def _health_pressed(self) -> None:
        self.run_health_check()


    @on(Button.Pressed, "#create_attack")
    def _create_attack_pressed(self) -> None:
        self.open_create_attack()


    @work(exclusive=True)
    async def run_health_check(self) -> None:
        label = self.query_one("#health_label", Label)
        label.update("status: requesting...")
        try:
            response = await self.client.get("/health")
            label.update(f"status: {response}")
        except httpx.HTTPError as e:
            label.update(f"status: error ({e})")


    @work(exclusive=True)
    async def open_create_attack(self) -> None:
        """Open the Create Attack modal, populating dropdowns from /capabilities."""
        try:
            caps = await self.client.get("/capabilities")
            attacks = caps.get("attacks", [])
            defenses = caps.get("defenses", [])
        except httpx.HTTPError:
            attacks, defenses = [], []
        config = await self.push_screen_wait(CreateAttackScreen(attacks, defenses))
        await self.on_attack_submitted(config)


    async def on_attack_submitted(self, config: dict | None) -> None:
        """Callback for the Create Attack modal"""
        if config is None:
            return
    
        try:
            result = await self.client.post("/runs", json=config)
            self.notify(f"Run {result['run_id']} started", severity="information")
        except httpx.HTTPStatusError as e:
            self.notify(f"Config Error: {e.response.json()}", severity="error")
            return
        except httpx.RequestError as e:
            self.notify(f"Network Error: {e}", severity="error")
            return
        
        self.action_refresh()

    def action_refresh(self) -> None:
        self.query_one(RunsTable).refresh_runs()


if __name__ == "__main__":
    NegevApp().run()