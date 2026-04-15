"""
LLM Comparison Study
Automates: prompt delivery, output collection, tournament judging.
Uses Poe's API via fastapi-poe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml
import fastapi_poe as fp
from dotenv import load_dotenv

STUDY_DIR = Path(__file__).parent
load_dotenv(STUDY_DIR / ".env")
load_dotenv(STUDY_DIR.parents[2] / ".env")  # project root .env fallback

API_KEY = os.getenv("POE_API_KEY")


STATE_FILE = STUDY_DIR / "judge_state.json"


def load_config():
    with open(STUDY_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_matchups": [], "rounds": [], "current_winner": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_prompt():
    """Extract the prompt from prompt.md (content inside the fenced code block)."""
    text = (STUDY_DIR / "prompt.md").read_text()
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    return "\n".join(lines[1:]).strip()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def split_thinking(text: str) -> tuple[str, str]:
    """Split model output into (thinking_trace, clean_output).

    Handles three formats:
    1. Blockquote thinking: *Thinking...* followed by > blockquotes
    2. Inline thinking ticks: Thinking... (Ns elapsed) repeated inline
    3. No thinking: returns ("", original_text)
    """
    lines = text.split("\n")

    # Format 1: *Thinking...* header then > blockquotes
    # Output starts at the first non-blockquote, non-empty line after the header
    if lines and lines[0].startswith("*Thinking"):
        for i in range(1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(">"):
                return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
        return text.strip(), ""

    # Format 2: inline "Thinking... (Ns elapsed)" ticks
    match = re.match(
        r"((?:Thinking\.{3} \(\d+s elapsed\))+)(.*)",
        text, re.DOTALL,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return "", text.strip()


def word_count(text: str) -> int:
    return len(text.split())



def fetch_latest_usage(bot_name: str) -> dict:
    """Fetch the most recent usage entry for a bot from Poe's Usage API."""
    try:
        resp = httpx.get(
            "https://api.poe.com/usage/points_history",
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={"limit": 5},
        )
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("bot_name", "").lower() == bot_name.lower():
                breakdown = entry.get("cost_breakdown_in_points", {})
                # Token counts are embedded in strings like "1433 points (3581 tokens)"
                def _parse_tokens(s):
                    m = re.search(r"\((\d+) tokens\)", s or "")
                    return int(m.group(1)) if m else 0
                return {
                    "cost_usd": entry.get("cost_usd", ""),
                    "cost_points": entry.get("cost_points", 0),
                    "tokens_output": _parse_tokens(breakdown.get("Output", "")),
                    "tokens_input": _parse_tokens(breakdown.get("Input", "")),
                }
    except Exception as e:
        print(f"  (Usage API: {e})")
    return {}


def query_model(bot_name: str, prompt: str, input_path: Path | None,
                parameters: dict | None) -> dict:
    """Send prompt + optional file to a Poe bot, return output and metadata."""
    attachments = []
    if input_path and input_path.exists():
        attachment = fp.upload_file_sync(
            open(input_path, "rb"),
            api_key=API_KEY
        )
        attachments.append(attachment)

    message = fp.ProtocolMessage(
        role="user",
        content=prompt,
        attachments=attachments,
        parameters=parameters or {},
    )

    start = time.time()
    chunks = []
    for partial in fp.get_bot_response_sync(
        messages=[message],
        bot_name=bot_name,
        api_key=API_KEY,
    ):
        chunks.append(partial.text)
    elapsed = time.time() - start

    # Fetch cost data from Usage API
    cost_data = fetch_latest_usage(bot_name)

    return {
        "output": "".join(chunks),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost": cost_data,
    }


def save_output(model_config: dict, result: dict, run_num: int = None) -> Path:
    """Save model output to outputs/ directory."""
    output_dir = STUDY_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    slug = slugify(model_config["name"])
    suffix = f"-run{run_num}" if run_num else ""
    path = output_dir / f"{slug}{suffix}.md"

    params_str = ""
    if model_config.get("parameters"):
        params_str = ", ".join(f"{k}: {v}" for k, v in model_config["parameters"].items())

    run_label = f" (Run {run_num})" if run_num else ""

    thinking, clean_output = split_thinking(result["output"])
    thinking_wc = word_count(thinking) if thinking else 0
    output_wc = word_count(clean_output)

    cost = result.get("cost", {})

    content = f"""# Model — {model_config["name"]}{run_label}

## Metadata

- **Bot:** {model_config["bot_name"]}
- **Parameters:** {params_str or "default"}
- **Timestamp:** {result["timestamp"]}
- **Response time:** {result["elapsed_seconds"]}s
- **Tokens (output):** {cost.get("tokens_output", "—")}
- **Word count (output):** {output_wc}
- **Word count (thinking):** {thinking_wc}
- **Cost (USD):** {cost.get("cost_usd", "—")}
- **Points (Poe):** {cost.get("cost_points", "—")}

## Output

{result["output"]}

## Comments

<!-- Your observations (optional) -->
"""
    path.write_text(content)
    return path


def run_judge(output_a: str, name_a: str, output_b: str, name_b: str,
              judge_config: dict, force_swap: bool | None = None) -> dict:
    """Run a pairwise judge comparison with randomized position.

    force_swap: None = random, True = swap, False = original order.
    """
    swap = force_swap if force_swap is not None else (random.random() < 0.5)
    if not swap:
        first, first_name = output_a, name_a
        second, second_name = output_b, name_b
        swapped = False
    else:
        first, first_name = output_b, name_b
        second, second_name = output_a, name_a
        swapped = True

    judge_prompt = f"""Both outputs below were generated from the same input using the same prompt.

Can you please evaluate the two below AI outputs?

Which model wins? Why? What does each do better or worse than the other? (table)

## Model A

```
{first}
```

## Model B

```
{second}
```"""

    result = query_model(
        bot_name=judge_config["bot_name"],
        prompt=judge_prompt,
        input_path=None,
        parameters=judge_config.get("parameters"),
    )

    return {
        "verdict": result["output"],
        "position_a": first_name,
        "position_b": second_name,
        "swapped": swapped,
        "timestamp": result["timestamp"],
        "judge_model": judge_config.get("name", judge_config["bot_name"]),
        "cost": result.get("cost", {}),
    }


def save_judge_output(matchup_num: int, name_a: str, name_b: str,
                      runs: list, winner: str = "—"):
    """Save judge verdicts for a matchup to outputs/judge/."""
    judge_dir = STUDY_DIR / "outputs" / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    slug_a = slugify(name_a)
    slug_b = slugify(name_b)
    path = judge_dir / f"matchup-{matchup_num}-{slug_a}-vs-{slug_b}.md"

    run_sections = []
    for j, r in enumerate(runs, 1):
        jc = r.get("judge", {}).get("cost", {})
        run_label = f"## Run {j}\n\n" if len(runs) > 1 else ""
        run_sections.append(f"""{run_label}**Position order** ({"swapped" if r["judge"]["swapped"] else "original"}):
- A = {r["judge"]["position_a"]}
- B = {r["judge"]["position_b"]}

**Cost:** ${float(jc.get('cost_usd', 0) or 0):.3f} ({int(jc.get('cost_points', 0) or 0):,} pts) | Tokens (output): {jc.get('tokens_output', '—')}

### Judge verdict

> {r["judge"]["verdict"].replace(chr(10), chr(10) + "> ")}
""")

    content = f"""# Matchup {matchup_num} — {name_a} vs. {name_b}

**Judge:** {runs[0]["judge"].get("judge_model", "unknown")}
**Winner:** {winner}

{chr(10).join(run_sections)}
"""
    path.write_text(content)
    return path


def update_study(config: dict, model_results: dict, rounds: list):
    """Update study.md with results."""
    gen_runs = config.get("runs_per_model", 1)
    judge_runs = config.get("judge_runs_per_matchup", 1)

    # Build models table
    model_rows = []
    for i, m in enumerate(config["models"], 1):
        slug = slugify(m["name"])
        params = ""
        if m.get("parameters"):
            params = ", ".join(f"{k}={v}" for k, v in m["parameters"].items())
        results = model_results.get(m["name"], [])
        if results:
            avg_time = round(sum(r["elapsed_seconds"] for r in results) / len(results), 1)
            time_str = f"{avg_time}s" + (f" (avg of {len(results)})" if len(results) > 1 else "")
        else:
            # Backfill from existing output file
            output_path = STUDY_DIR / "outputs" / f"{slug}.md"
            if output_path.exists():
                file_text = output_path.read_text()
                match = re.search(r"Response time:\*\* (.+?)s", file_text)
                time_str = f"{match.group(1)}s" if match else ""
            else:
                file_text = ""
                time_str = ""
        # Backfill metrics from output file
        output_path = STUDY_DIR / "outputs" / f"{slug}.md"
        if output_path.exists():
            file_text = output_path.read_text()
            def _extract(pattern, default="—"):
                m2 = re.search(pattern, file_text)
                return m2.group(1) if m2 else default
            tokens_out = _extract(r"Tokens \(output\):\*\* (.+)")
            output_wc = _extract(r"Word count \(output\):\*\* (\d+)")
            thinking_wc = _extract(r"Word count \(thinking\):\*\* (\d+)")
            cost_usd = _extract(r"Cost \(USD\):\*\* (.+)")
            cost_points = _extract(r"Points \(Poe\):\*\* (.+)")
        else:
            tokens_out = output_wc = thinking_wc = cost_usd = cost_points = "—"
        if gen_runs > 1:
            links = ", ".join(f"[run {r}](outputs/{slug}-run{r}.md)" for r in range(1, gen_runs + 1))
        else:
            links = f"[output](outputs/{slug}.md)"
        model_rows.append(f"| {i} | {m['name']} | {params} | {links} | {time_str} | {tokens_out} | {output_wc} | {thinking_wc} | {cost_usd} | {cost_points} |")

    # Group rounds into matchups
    matchups = []
    for r in rounds:
        key = (r["name_a"], r["name_b"])
        if not matchups or matchups[-1]["key"] != key:
            matchups.append({"key": key, "runs": [r], "winner": r.get("matchup_winner", "—")})
        else:
            matchups[-1]["runs"].append(r)
            if r.get("matchup_winner"):
                matchups[-1]["winner"] = r["matchup_winner"]

    # Build tournament progress table
    bracket_rows = []
    for i, matchup in enumerate(matchups, 1):
        name_a, name_b = matchup["key"]
        slug_a = slugify(name_a)
        slug_b = slugify(name_b)
        judge_file = f"matchup-{i}-{slug_a}-vs-{slug_b}.md"
        winner = matchup.get("winner", "—")
        bracket_rows.append(
            f"| {i} | {name_a} | {name_b} | [details](outputs/judge/{judge_file}) | {winner} |"
        )

    # Build judge cost rows
    judge_cost_rows = []
    total_judge_usd = 0.0
    total_judge_points = 0
    for i, matchup in enumerate(matchups, 1):
        for j, r in enumerate(matchup["runs"], 1):
            jc = r.get("judge", {}).get("cost", {})
            usd = float(jc.get("cost_usd", 0) or 0)
            pts = int(jc.get("cost_points", 0) or 0)
            tokens = jc.get("tokens_output", "—")
            total_judge_usd += usd
            total_judge_points += pts
            run_label = f" / Run {j}" if len(matchup["runs"]) > 1 else ""
            judge_cost_rows.append(
                f"| M{i}{run_label} | {matchup['key'][0]} | {matchup['key'][1]} | {r['judge'].get('judge_model', '—')} "
                f"| {tokens} | ${usd:.3f} | {pts:,} |"
            )

    # Compute totals
    # Sum generation costs from output files
    total_gen_usd = 0.0
    total_gen_points = 0
    for m in config["models"]:
        slug = slugify(m["name"])
        output_path = STUDY_DIR / "outputs" / f"{slug}.md"
        if output_path.exists():
            ft = output_path.read_text()
            usd_m = re.search(r"Cost \(USD\):\*\* (.+)", ft)
            pts_m = re.search(r"Points \(Poe\):\*\* (.+)", ft)
            if usd_m:
                try: total_gen_usd += float(usd_m.group(1))
                except ValueError: pass
            if pts_m:
                try: total_gen_points += int(pts_m.group(1))
                except ValueError: pass

    total_usd = total_gen_usd + total_judge_usd
    total_points = total_gen_points + total_judge_points
    cost_totals = (
        f"**Total judge cost:** ${total_judge_usd:.3f} ({total_judge_points:,} pts)\n"
        f"**Total generation cost:** ${total_gen_usd:.3f} ({total_gen_points:,} pts)\n"
        f"**Total study cost:** ${total_usd:.3f} ({total_points:,} pts)"
    ) if rounds else "**Total judge cost:** —\n**Total study cost:** —"

    gen_note = f"\n  - **Generation runs per model:** {gen_runs}" if gen_runs > 1 else ""
    judge_note = f"\n  - **Judge runs per matchup:** {judge_runs}" if judge_runs > 1 else ""
    content = f"""# LLM Comparison Study

**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Author:** (your name)
**Platform:** Poe (API)

## Methodology

- **Task:** Fixed prompt sent to all models (see [prompt.md](prompt.md))
- **Input:** Same input and prompt across all models (see [prompt.md](prompt.md))
- **Evaluation:** Tournament-style pairwise comparison
  - **Judge LLM:** {config["judge"]["bot_name"]} ({", ".join(f"{k}={v}" for k, v in config["judge"].get("parameters", {}).items())}) — see [judge-prompt.md](judge-prompt.md)
  - **Human reviewer:** qualitative comments, no numerical scale
  - **Position bias mitigation:** Randomized A/B placement per round{gen_note}{judge_note}
- **Advancement:** The better output (judge verdict or human call) advances to face the next model

## Models Tested

| # | Model | Parameters | Output | Response Time | Tokens (output) | Words (output) | Words (thinking) | Cost (USD) | Points (Poe) |
|---|-------|------------|--------|---------------|-----------------|----------------|------------------|------------|--------------|
{chr(10).join(model_rows)}

## Tournament Bracket

| Matchup | Defender | Challenger | Verdict | Winner |
|---------|----------|------------|---------|--------|
{chr(10).join(bracket_rows) if bracket_rows else "<!-- Will be populated after judging -->"}

## Judge Costs

| Matchup | Defender | Challenger | Judge | Tokens (output) | Cost (USD) | Points (Poe) |
|---------|----------|------------|-------|-----------------|------------|--------------|
{chr(10).join(judge_cost_rows) if judge_cost_rows else "<!-- Will be populated after judging -->"}

{cost_totals}
"""
    (STUDY_DIR / "study.md").write_text(content)


def cmd_generate(args):
    """Generate outputs from all models (or a subset via --only)."""
    config = load_config()
    prompt = load_prompt()
    input_file = STUDY_DIR / config.get("input_file", "input.txt")
    runs = config.get("runs_per_model", 1)

    if not input_file.exists():
        print(f"Warning: input file not found at {input_file}")
        if input("Continue without file attachment? [y/N] ").lower() != "y":
            return

    # Filter models if --only is specified
    models = config["models"]
    if args.only:
        only_names = {n.strip().lower() for n in args.only.split(",")}
        models = [m for m in models if m["name"].lower() in only_names]
        if not models:
            print(f"No models matched --only filter. Available names:")
            for m in config["models"]:
                print(f"  - {m['name']}")
            return

    # Check for existing outputs
    existing = []
    new = []
    for m in models:
        slug = slugify(m["name"])
        path = STUDY_DIR / "outputs" / f"{slug}.md"
        if path.exists():
            existing.append(m)
        else:
            new.append(m)

    if existing and not args.force:
        print(f"\nExisting outputs found for {len(existing)} model(s):")
        for m in existing:
            print(f"  - {m['name']}")
        print(f"\nNew models to run: {len(new)}")
        print(f"\nOptions:")
        print(f"  1) Skip existing, run new only")
        print(f"  2) Overwrite all (re-query existing + new)")
        print(f"  3) Abort")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice == "2":
            models = existing + new
        elif choice == "1":
            models = new
        else:
            print("Aborted.")
            return
    elif args.force:
        models = existing + new

    if not models:
        print("No new models to run.")
        update_study(config, {}, [])
        print("study.md refreshed.")
        return

    model_results = {}
    failed = []
    for m in models:
        results = []
        for run in range(1, runs + 1):
            run_label = f" (run {run}/{runs})" if runs > 1 else ""
            print(f"\n--- Querying {m['name']}{run_label} ---")
            try:
                result = query_model(
                    bot_name=m["bot_name"],
                    prompt=prompt,
                    input_path=input_file if input_file.exists() else None,
                    parameters=m.get("parameters"),
                )
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(m["name"])
                break
            print(f"Done in {result['elapsed_seconds']}s")

            run_num = run if runs > 1 else None
            path = save_output(m, result, run_num=run_num)
            results.append(result)
            print(f"Saved to {path.name}")

        if results:
            model_results[m["name"]] = results

    update_study(config, model_results, [])
    print(f"\nDone — {sum(len(r) for r in model_results.values())} outputs generated.")
    if failed:
        print(f"Failed: {', '.join(failed)} — check parameters and retry with --only")


def cmd_judge(args):
    """Run tournament-style judging. Resumes from saved state if available."""
    config = load_config()
    output_dir = STUDY_DIR / "outputs"
    gen_runs = config.get("runs_per_model", 1)
    runs = config.get("judge_runs_per_matchup", 1)

    # Load existing outputs (use first run or single output)
    outputs = {}
    for m in config["models"]:
        slug = slugify(m["name"])
        if gen_runs > 1:
            path = output_dir / f"{slug}-run1.md"
        else:
            path = output_dir / f"{slug}.md"
        if not path.exists():
            print(f"Missing output for {m['name']} — run 'generate' first.")
            return
        text = path.read_text()
        match = re.search(r"## Output\n\n(.*?)(?=\n## Comments)", text, re.DOTALL)
        raw_output = match.group(1).strip() if match else text
        _, clean_output = split_thinking(raw_output)
        outputs[m["name"]] = clean_output

    # Load or initialize state
    state = load_state()
    rounds = state.get("rounds", [])
    completed = set(tuple(x) for x in state.get("completed_matchups", []))
    models = config["models"]

    if state.get("current_winner") and state["current_winner"] in outputs:
        current_winner_name = state["current_winner"]
        print(f"Resuming tournament. Current champion: {current_winner_name}")
    else:
        current_winner_name = models[0]["name"]
    current_winner_output = outputs[current_winner_name]

    for m in models[1:]:
        challenger_name = m["name"]
        matchup_key = (current_winner_name, challenger_name)

        if matchup_key in completed:
            print(f"Skipping completed matchup: {current_winner_name} vs. {challenger_name}")
            continue

        challenger_output = outputs[challenger_name]

        # Build a balanced swap schedule
        if runs > 1:
            swap_schedule = [False] * (runs // 2) + [True] * (runs // 2)
            if runs % 2 == 1:
                swap_schedule.append(random.choice([True, False]))
            random.shuffle(swap_schedule)

        matchup_num = len(completed) + 1
        for run_idx in range(runs):
            run_label = f" (run {run_idx+1}/{runs})" if runs > 1 else ""
            force_swap = swap_schedule[run_idx] if runs > 1 else None

            print(f"\n--- Matchup {matchup_num}: {current_winner_name} vs. {challenger_name}{run_label} ---")
            try:
                judge_result = run_judge(
                    current_winner_output, current_winner_name,
                    challenger_output, challenger_name,
                    config["judge"],
                    force_swap=force_swap,
                )
            except Exception as e:
                print(f"\nERROR: {e}")
                print("Saving progress...")
                state["rounds"] = rounds
                state["current_winner"] = current_winner_name
                save_state(state)
                update_study(config, {}, rounds)
                print(f"Saved. Re-run 'judge' to retry this matchup.")
                return

            rounds.append({
                "name_a": current_winner_name,
                "name_b": challenger_name,
                "judge": judge_result,
                "run": run_idx + 1 if runs > 1 else None,
            })
            # Save after every judge call
            state["rounds"] = rounds
            state["current_winner"] = current_winner_name
            save_state(state)
            # Save judge file incrementally (winner TBD until matchup ends)
            matchup_runs = [r for r in rounds if r["name_a"] == current_winner_name and r["name_b"] == challenger_name]
            save_judge_output(matchup_num, current_winner_name, challenger_name, matchup_runs)
            update_study(config, {}, rounds)
            print(f"Position A: {judge_result['position_a']}")
            print(f"Position B: {judge_result['position_b']}")
            print(f"\nJudge says:\n{judge_result['verdict'][:500]}...")

        # Collect this matchup's runs for saving
        matchup_runs = [r for r in rounds if r["name_a"] == current_winner_name and r["name_b"] == challenger_name]

        # Ask for winner once after all runs for this matchup
        print(f"\nWho wins this matchup?")
        print(f"  1) {current_winner_name}")
        print(f"  2) {challenger_name}")
        print(f"  Enter) Pause tournament (verdicts are saved)")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted. Progress saved.")
            state["rounds"] = rounds
            state["current_winner"] = current_winner_name
            save_state(state)
            update_study(config, {}, rounds)
            return
        if choice == "":
            state["rounds"] = rounds
            state["current_winner"] = current_winner_name
            save_state(state)
            update_study(config, {}, rounds)
            print(f"\nTournament paused. Re-run 'judge' to continue.")
            return
        elif choice == "2":
            current_winner_name = challenger_name
            current_winner_output = challenger_output

        # Save winner into round data and judge file
        winner = current_winner_name
        for r in matchup_runs:
            r["matchup_winner"] = winner
        save_judge_output(matchup_num, matchup_runs[0]["name_a"], matchup_runs[0]["name_b"], matchup_runs, winner)

        # Mark matchup as completed
        completed.add(matchup_key)
        state["completed_matchups"] = [list(x) for x in completed]
        state["rounds"] = rounds
        state["current_winner"] = current_winner_name
        save_state(state)
        update_study(config, {}, rounds)

    update_study(config, {}, rounds)
    # Clean up state file — tournament is done
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print(f"\nTournament complete — {len(completed)} matchups. Review study.md for results.")


def main():
    if not API_KEY:
        print("Error: POE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="LLM Comparison Study Runner")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate outputs from all models")
    gen.add_argument("--only", help="Comma-separated model names to run (skip others)")
    gen.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    sub.add_parser("judge", help="Run tournament judging on existing outputs")
    allp = sub.add_parser("all", help="Generate outputs then run judging")
    allp.add_argument("--only", help="Comma-separated model names to generate")
    allp.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    allp.add_argument("--skip-cost", action="store_true", help="Skip cost input prompts")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "judge":
        cmd_judge(args)
    elif args.command == "all":
        cmd_generate(args)
        cmd_judge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
