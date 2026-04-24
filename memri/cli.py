"""memri CLI — init, status, serve, mcp-server, dashboard, ingest."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option("0.2.0", prog_name="memri")
def main():
    """memri — observational memory for coding agents."""


# ─────────────────────────────── init ──────────────────────────────────


@main.command()
@click.option("--claude-code", "claude_code", is_flag=True, default=False,
              help="Configure memri in Claude Code settings.json")
@click.option("--all-agents", "all_agents", is_flag=True, default=False,
              help="Configure memri for all detected coding agents")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be done without making changes")
def init(claude_code: bool, all_agents: bool, dry_run: bool):
    """Initialize memri: create database, detect agents, configure MCP."""
    from .config import MemriConfig, MEMRI_DIR
    from .ingestion.auto_detect import detect_installed_agents, configure_claude_code

    console.print("\n[bold]memri init[/bold]\n")

    # Create config dir and DB
    MEMRI_DIR.mkdir(parents=True, exist_ok=True)
    config = MemriConfig.load()
    if not dry_run:
        config.save()

    from .storage.sqlite_store import SQLiteStore
    store = SQLiteStore(config.db_path)
    console.print(f"[green]ok[/green] Database: {config.db_path}")

    # Detect agents
    agents = detect_installed_agents()
    if agents:
        console.print(f"[green]ok[/green] Detected agents: {', '.join(agents)}")
    else:
        console.print("[yellow]warn[/yellow] No coding agents detected (Claude Code, Cursor, Codex)")

    # Configure agents
    if claude_code or all_agents or ("claude-code" in agents and not dry_run):
        success, msg = configure_claude_code(dry_run=dry_run)
        icon = "[green]ok[/green]" if success else "[red]err[/red]"
        console.print(f"{icon} Claude Code: {msg}")

    # Auto-detect best available provider if none configured
    from pathlib import Path
    claude_creds = Path.home() / ".claude" / ".credentials.json"

    if not config.llm_api_key and config.llm_provider not in ("passive", "claude-code-auth"):
        if claude_creds.exists():
            # Claude Code credentials found — use them automatically
            if not dry_run:
                config.llm_provider = "claude-code-auth"
                config.llm_model = "claude-haiku-4-5-20251001"
                config.save()
            console.print(
                "[green]ok[/green] LLM: using your Claude subscription "
                "(claude-code-auth — no API key needed)"
            )
        else:
            console.print(
                "\n[yellow]No API key or Claude login found.[/yellow]\n\n"
                "  [bold]Option 1 — Use your Claude subscription[/bold] (no API key)\n"
                "    Already have Claude Code? Just run: [cyan]claude[/cyan]  to log in once.\n"
                "    memri will automatically use those credentials.\n\n"
                "  [bold]Option 2 — Free Gemini API[/bold] (takes 1 min)\n"
                "    → [link=https://aistudio.google.com/apikey]https://aistudio.google.com/apikey[/link]\n"
                "    Add [cyan]GEMINI_API_KEY=...[/cyan] to [cyan]~/.memri/.env[/cyan]\n\n"
                "  [bold]Option 3 — Local model via Ollama[/bold] (fully private)\n"
                "    → [link=https://ollama.ai]https://ollama.ai[/link]  then: ollama pull llama3\n"
                "    Set [cyan]llm_provider: openai-compatible[/cyan] in [cyan]~/.memri/config.json[/cyan]\n\n"
                "  [bold]Option 4 — Passive mode[/bold] (no LLM, zero setup)\n"
                "    Set [cyan]llm_provider: passive[/cyan] in [cyan]~/.memri/config.json[/cyan]\n"
            )
    else:
        console.print(
            "\n[bold]Ready.[/bold] Start the MCP server: [cyan]memri mcp-server[/cyan]\n"
        )


# ────────────────────────────── status ─────────────────────────────────


@main.command()
def status():
    """Show memory stats: sessions, tokens saved, costs."""
    from .config import MemriConfig
    from .core.memory import MemriMemory

    config = MemriConfig.load()
    memory = MemriMemory(config)
    stats = memory.get_stats()

    table = Table(title="Memri Memory Status", show_header=False, box=None,
                  padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Threads", str(stats["threads"]))
    table.add_row("Messages", f"{stats['messages']:,}")
    table.add_row("Observation blocks", str(stats["observations"]))
    table.add_row("Tokens saved", f"{stats['tokens_saved']:,}")
    table.add_row("Cost saved (est.)", f"${stats['cost_saved_usd']:.4f}")
    table.add_row("Memri LLM cost", f"${stats['llm_cost_usd']:.4f}")
    net = stats["cost_saved_usd"] - stats["llm_cost_usd"]
    table.add_row("Net savings", f"${net:.4f}")
    if stats.get("oldest_memory"):
        table.add_row("Oldest memory", stats["oldest_memory"][:10])

    console.print(table)


# ──────────────────────────── mcp-server ───────────────────────────────


@main.command("mcp-server")
def mcp_server():
    """Start the MCP server (called by Claude Code / Cursor)."""
    from .mcp.server import run
    run()


# ─────────────────────────────── watch ─────────────────────────────────


@main.command()
@click.option("--path", "extra_paths", multiple=True,
              help="Additional paths to watch (repeatable)")
@click.option("--auto-observe", is_flag=True, default=False,
              help="Run observer automatically when a thread exceeds the threshold")
def watch(extra_paths: tuple, auto_observe: bool):
    """Watch for new coding-agent sessions and ingest them in real time.

    Monitors ~/.claude/projects/ (and Cursor/Codex dirs if present).
    Runs until Ctrl+C.
    """
    from .config import MemriConfig
    from .core.memory import MemriMemory
    from .ingestion.watcher import SessionWatcher, default_watch_paths

    config = MemriConfig.load()
    memory = MemriMemory(config)

    paths = default_watch_paths() + [Path(p) for p in extra_paths]
    if not paths:
        console.print("[red]err[/red] No agent session directories found to watch.")
        console.print("     Pass --path to specify a directory manually.")
        return

    console.print("\n[bold]memri watch[/bold]  (Ctrl+C to stop)\n")
    for p in paths:
        console.print(f"[green]ok[/green] Watching {p}")
    console.print()

    total_ingested = 0

    def on_ingest(path: str, added: int):
        nonlocal total_ingested
        total_ingested += added
        short = Path(path).name
        console.print(f"[green]+{added}[/green] {short}  ({total_ingested:,} total)")

    async def _run():
        watcher = SessionWatcher(memory, paths, on_ingest=on_ingest)

        if auto_observe:
            async def _observer_loop():
                while True:
                    await asyncio.sleep(60)
                    await memory.observe_all()
            asyncio.create_task(_observer_loop())

        await watcher.run_forever()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print(f"\n[dim]Stopped. {total_ingested:,} messages ingested this session.[/dim]")


# ────────────────────────────── dashboard ──────────────────────────────


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8050, show_default=True)
def dashboard(host: str, port: int):
    """Open the local web dashboard."""
    from .dashboard.app import run
    console.print(f"[green]Dashboard running at[/green] http://{host}:{port}")
    run(host=host, port=port)


# ──────────────────────────── ingest ───────────────────────────────────


@main.command()
@click.option("--agent", default="claude-code",
              type=click.Choice(["claude-code", "cursor", "codex"]),
              show_default=True)
@click.option("--path", "sessions_path", default=None,
              help="Custom path to sessions directory")
def ingest(agent: str, sessions_path: Optional[str]):
    """Ingest existing coding agent sessions into memory."""
    from .config import MemriConfig
    from .core.memory import MemriMemory

    config = MemriConfig.load()
    memory = MemriMemory(config)

    async def _run():
        if agent == "claude-code":
            from .ingestion.claude_code import ingest_all_sessions
            path = Path(sessions_path) if sessions_path else None
            results = await ingest_all_sessions(memory, path)
        elif agent == "cursor":
            from .ingestion.cursor import ingest_all_sessions
            path = Path(sessions_path) if sessions_path else None
            results = await ingest_all_sessions(memory, path)
        else:
            from .ingestion.codex import ingest_all_sessions
            path = Path(sessions_path) if sessions_path else None
            results = await ingest_all_sessions(memory, path)
        return results

    results = asyncio.run(_run())
    total = sum(results.values())
    console.print(
        f"[green]ok[/green] Ingested {total:,} messages from {len(results)} session(s)"
    )


# ────────────────────────────── observe ────────────────────────────────


@main.command()
@click.option("--thread", default=None, help="Only observe a specific thread ID")
def observe(thread: Optional[str]):
    """Run the Observer on all threads with enough unobserved tokens.

    Compresses raw messages into dense observation blocks.
    Safe to run multiple times — skips threads already under the threshold.
    """
    from .config import MemriConfig
    from .core.memory import MemriMemory

    config = MemriConfig.load()
    memory = MemriMemory(config)

    threads = memory.store.list_threads()
    if thread:
        threads = [t for t in threads if t.id == thread]

    total_queued = 0
    for t in threads:
        unobserved = memory.store.get_messages(t.id, unobserved_only=True)
        tok = sum(m.token_count for m in unobserved)
        if tok >= config.observe_threshold:
            total_queued += tok

    if not total_queued:
        console.print("[dim]No threads exceed the observe threshold. Nothing to do.[/dim]")
        return

    console.print(f"Processing {total_queued:,} unobserved tokens across {len(threads)} thread(s)...\n")

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    import asyncio

    cycles_total = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Observing...", total=len(threads))

        async def _run():
            nonlocal cycles_total
            for t in threads:
                progress.update(task, description=f"[cyan]{t.id[:8]}[/cyan]")
                cycles = 0
                while True:
                    unobserved = memory.store.get_messages(t.id, unobserved_only=True)
                    tok = sum(m.token_count for m in unobserved)
                    if tok < config.observe_threshold:
                        break
                    await memory._run_observer(t.id, unobserved)
                    cycles += 1
                cycles_total += cycles
                progress.advance(task)

        asyncio.run(_run())

    stats = memory.get_stats()
    console.print(f"[green]ok[/green] {cycles_total} observation cycle(s) completed")
    console.print(f"   Observation blocks: {stats['observations']}")
    console.print(f"   Tokens saved: {stats['tokens_saved']:,}")
    console.print(f"   Cost saved (est.): ${stats['cost_saved_usd']:.4f}")
    console.print(f"   Memri cost: ${stats['llm_cost_usd']:.4f}")


# ─────────────────────────── embed ─────────────────────────────────────


@main.command()
def embed():
    """Build semantic search index from all observation blocks.

    Requires: pip install 'memri[embeddings]'
    """
    from .config import MemriConfig
    from .core.memory import MemriMemory

    config = MemriConfig.load()
    memory = MemriMemory(config)

    if not memory.embedder.available:
        console.print("[red]err[/red] sentence-transformers not installed.")
        console.print("     Run: pip install 'memri[embeddings]'")
        return

    obs_count = len(memory.store.get_all_observations())
    if not obs_count:
        console.print("[dim]No observation blocks to embed. Run `memri observe` first.[/dim]")
        return

    console.print(f"Embedding {obs_count} observation block(s)...")

    count = [0]

    def cb(tid, n):
        count[0] = n
        console.print(f"  [{n}/{obs_count}] {tid[:8]}", end="\r")

    total = memory.build_embeddings(progress_cb=cb)
    console.print(f"\n[green]ok[/green] Built embeddings for {total} observation block(s)")


# ─────────────────────────── config ────────────────────────────────────


@main.command("config")
@click.option("--show", is_flag=True, default=False, help="Show current config")
@click.option("--set", "set_kv", nargs=2, metavar="KEY VALUE", multiple=True,
              help="Set a config key (repeatable)")
def config_cmd(show: bool, set_kv: tuple):
    """View or update memri configuration."""
    from .config import MemriConfig

    config = MemriConfig.load()

    if set_kv:
        for key, value in set_kv:
            if hasattr(config, key):
                # Attempt type coercion
                current = getattr(config, key)
                if isinstance(current, int):
                    value = int(value)
                elif isinstance(current, bool):
                    value = value.lower() in ("1", "true", "yes")
                setattr(config, key, value)
                console.print(f"[green]ok[/green] {key} = {value}")
            else:
                console.print(f"[red]err[/red] Unknown config key: {key}")
        config.save()

    if show or not set_kv:
        import dataclasses
        data = dataclasses.asdict(config)
        data.pop("llm_api_key", None)  # Never display API keys
        console.print_json(json.dumps(data, indent=2))
