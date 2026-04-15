# LLM Comparison Study — Blueprint

A framework for comparing LLM outputs on a fixed task using tournament-style pairwise evaluation with an LLM judge and human review. Built on [Poe's API](https://creator.poe.com/docs/external-applications/external-application-guide), which provides access to models from OpenAI, Anthropic, Google, and others through a single API key.

## How it works

1. **Generate**: The same prompt + input file is sent to each model via Poe's API. Outputs are saved to `outputs/`, one file per model. Cost and token data is fetched automatically from Poe's Usage API.
2. **Judge**: Models are compared pairwise in a sequential tournament. The first model faces the second; the winner faces the third; and so on. An LLM judge evaluates each pair blindly (it only sees "Model A" / "Model B"). Full judge verdicts are saved to `outputs/judge/`, one file per matchup.
3. **Decide**: After each matchup, the user reads the judge verdicts and picks the winner. The winner advances. The tournament can be paused and resumed at any time.
4. **Document**: `study.md` serves as a dashboard — model table with metrics and costs, tournament progress table with links to verdict files, judge cost breakdown, and total study cost.

## Quick start

```bash
# Clone and set up
git clone <repo-url>
cd llm-comparison-blueprint
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add a Poe API key (create one at poe.com/api/keys)
echo "POE_API_KEY=your_key_here" > .env
```

Then:
1. Write the task prompt in `prompt.md`
2. Place the input file in the study directory (use a `_gitignore` suffix to keep it out of version control)
3. Configure models in `config.yaml`
4. Run the study:

```bash
source .venv/bin/activate
python run_study.py generate    # send prompt to all models, collect outputs
python run_study.py judge       # run the tournament (interactive)
```

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Models to test, their parameters, judge config, run counts |
| `prompt.md` | The exact prompt sent to all models |
| `judge-prompt.md` | The judge evaluation prompt (for documentation — the script builds it) |
| `study.md` | Dashboard: model table, tournament progress, cost tracking |
| `outputs/` | One markdown file per model output, with metadata and word counts |
| `outputs/judge/` | One markdown file per matchup, with full judge verdicts and cost data |
| `judge_state.json` | Tournament progress for pause/resume (gitignored, auto-deleted when complete) |
| `run_study.py` | The automation script |
| `.env` | Poe API key (gitignored) |
| `requirements.txt` | Python dependencies |

## Usage

```bash
source .venv/bin/activate

# Generate outputs from all models
python run_study.py generate

# Generate only specific models
python run_study.py generate --only "GPT-5.4 (medium)"

# Force regenerate (prompts before overwriting)
python run_study.py generate --force

# Run the tournament judge (interactive)
python run_study.py judge

# Generate then judge in sequence
python run_study.py all
```

### Pausing and resuming the tournament

The user can pause the tournament at any winner prompt by pressing Enter. Progress is saved to `judge_state.json`. Re-running `python run_study.py judge` resumes from where it left off. Completed matchups are skipped, and the current champion is restored.

If the script crashes or is interrupted (Ctrl+C), all completed judge verdicts are already saved — both to `outputs/judge/` and to `study.md`. No data is lost.

## Configuration

`config.yaml` controls everything:

```yaml
prompt_file: prompt.md          # prompt sent to all models
input_file: input.txt           # input file (rename as needed)
runs_per_model: 1               # how many times to query each model
judge_runs_per_matchup: 2       # how many times the judge evaluates each pair

judge:
  name: Claude-Opus-4.6 (medium)
  bot_name: claude-opus-4.6
  parameters:
    output_effort: medium

models:
  - name: GPT-5.4 (medium)
    bot_name: gpt-5.4           # Poe bot handle (case-sensitive)
    parameters:                 # bot-specific — check Poe UI for available options
      reasoning_effort: medium
      enable_web_search: false
```

**Parameters are bot-specific on Poe.** The user should check each bot's settings in the Poe UI for available parameter names and values. Common ones: `reasoning_effort`, `output_effort`, `thinking_level`, `enable_web_search`, `verbosity`.

**Model order matters.** The first model in the list is the initial defender. Subsequent models challenge the current champion in order. Consider seeding weaker/cheaper models first so stronger ones face the reigning champion.

## Adding new models later

1. Add the model to `config.yaml`
2. Run `python run_study.py generate` — only the new model runs (existing outputs are preserved)
3. Run `python run_study.py judge` — the full tournament re-runs with all models

The judge model can be updated in `config.yaml` at any time. Each matchup records which judge evaluated it, so results from different judge versions remain traceable.

## Cost tracking

- **Generation costs** are fetched automatically from Poe's Usage API after each model query
- **Judge costs** are tracked per matchup run during the tournament
- **Totals** (generation + judging) are computed at the bottom of `study.md`
- The Usage API is read-only and free — it does not consume points
- Cost data (USD, Poe points, output tokens) is stored both in individual output files and in `study.md`

## Thinking traces

Some models emit thinking/reasoning traces before their output. The script:
- **Keeps them** in the saved output files (for review)
- **Strips them** before sending to the judge (so it evaluates only the actual output)
- **Counts words separately**: "Words (output)" vs "Words (thinking)" in the study table

## Position bias

LLM judges tend to favor whichever output they read first. Mitigations:
- With `judge_runs_per_matchup: 1` — position is randomized per matchup
- With `judge_runs_per_matchup: 2+` — a balanced schedule is built: half the runs use original order, half use swapped order, then the schedule is shuffled. For odd counts, the extra run is randomly assigned. The user sees all verdicts before picking a winner.

## Limitations

- **Tournament format**: A model that would beat the champion might be eliminated early by a different opponent. Consider running multiple tournaments with different seedings if results seem order-dependent.
- **Single judge**: The evaluation relies on one LLM judge. Different judges may have different preferences.
- **Single prompt**: Results are specific to the task and prompt used. A model that wins on one task may not win on another.
- **Poe platform**: Costs and available models depend on Poe's pricing and catalog. Parameter names are bot-specific and may change.

## License

MIT
