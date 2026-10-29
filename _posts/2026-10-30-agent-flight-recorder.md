---
layout: post
title: "My agent broke because of one prompt change, so I built it a flight recorder"
date: 2026-10-30
categories: agents testing
---

Every agent developer knows this night. Yesterday the agent was fine — every task handled, every tool called in the right order. Today you changed one line in the prompt. One line. Reordered a sentence, tightened an instruction. Now the agent calls the wrong tool three steps in, or answers with something confidently, plausibly wrong.

So you rerun it. That costs money, and you get a *different* wrong answer. Rerun again: different again. Ten reruns later you've spent real money, learned nothing, and your logs are print soup — twenty `print` statements interleaved with SDK progress spinners, and the one line you needed belonged to a run that already scrolled away.

Here is the actual problem: **an agent run is a long chain of model calls and tool calls, and you cannot diff what you did not record.** Rerunning is not reproduction. The model is nondeterministic, and every attempt to see "the same" failure again is a fresh, paid, slightly different run.

## The idea: treat a run like a flight

Airplanes carry flight recorders because you cannot rerun a crashed flight. You pull the recorder and you know exactly what happened.

That's [tapedeck](https://github.com/EAnge1o/tapedeck), a Python library I built: a flight recorder for LLM agents.

- **Record once.** A live agent run — every model call, every tool call, every usage record — traced to a versioned JSONL file.
- **Replay forever.** Offline, zero API calls, byte-deterministic outputs.
- **Diff two runs.** One command prints the first step where behavior drifted: which step, which call, from what value to what value.

It has zero dependencies: pure Python standard library — even the test suite is `unittest`. Traces are local files only; nothing leaves the machine. They can go into git, be `cat`-ed, be `rsync`-ed.

One clarification that comes up immediately: platforms like LangSmith and Langfuse are complementary, not competitors. They are dashboards — built to watch trends across many runs. tapedeck is a recorder — built to answer "what is the complete call sequence of this one incident, and where exactly does it differ from last Friday's good run?" For that question, local files beat a web snapshot.

## How it works

Your agent code talks to a tape instead of calling the SDK directly. The entire integration surface is one function:

```python
import tapedeck

def run_agent(tape, message):                      # takes a tape instead of calling openai itself
    reply = tape.call("model", "deepseek.chat", chat, messages)   # fn = your real call
    return tape.call("tool", "postprocess", lambda: reply.strip())

with tapedeck.record("chat-v3", redact=tapedeck.Redactor()) as tape:   # REC: auto-saves on exit
    run_agent(tape, "hello")

with tapedeck.play("chat-v3") as tape:                                 # PLAY: free regression test
    run_agent(tape, "hello")                                           # validates step count on exit
```

`tape.call(kind, name, fn, *args)` does the one interesting thing in the library: while recording, it executes `fn` and writes the result to the trace. While replaying, it **never calls `fn` at all** — it serves the recorded value. That's why replay costs $0.00 and cannot fail on a network error: there is no network.

The integration rule of thumb: model calls and side-effectful tool calls go through the tape; pure functions don't. tapedeck doesn't replace unit tests — it covers the integration layer. The fact that your whole agent once behaved end-to-end correctly is a fact worth freezing as a regression assertion.

A recorded run is a directory with two plain files:

```
.tapedeck/runs/20261026-151919-demo-stable/
    meta.json     # name, time, host, step count, whether redacted
    trace.jsonl   # one step per line: kind/name/input/output/error/usage/wall_ms
```

and each step is one JSON line:

```json
{"v": 1, "i": 0, "kind": "model", "name": "model.plan",
 "input": [], "output": {"tool": "get_price", "args": "coffee"},
 "usage": {"prompt_tokens": 118, "completion_tokens": 24, "model": "deepseek-chat"},
 "wall_ms": 0.142}
```

JSONL rather than SQLite or a cloud store is deliberate. `grep`, `jq`, `git diff` and `cat` all work directly on a crash scene, and a run directory can sit in git as review material. The `v` field is validated on read and unknown versions are rejected, so format upgrades stay explicit instead of silent.

Two properties the recorder guarantees:

- **Tracing never breaks the run.** Anything unserializable on the recording path is downgraded to a `repr` string or a warning event, and if the agent crashes, every step before the crash is already on disk. The recorder must not become a new failure point.
- **Replay is strict by default.** `play()` raises `ReplayDrift` on a missing step, an extra step, or changed inputs — for a regression test, silent success is the worst possible outcome. (`lenient=True` exists as an explicit escape hatch for human debugging; it is not for CI.)

## Finding the first fork

Here is the repo's demo agent — a small coffee-pricing assistant — recorded twice. Between the two recordings, someone changed the discount policy.

```bash
pip install -e .                      # or just drop the tapedeck/ folder into your project
python examples/demo_agent.py --mode record --scenario stable
python examples/demo_agent.py --mode record --scenario drift   # someone changed the discount policy
python examples/demo_agent.py --mode replay                    # offline replay, $0.00
python -m tapedeck diff demo-stable demo-drift
```

The output below is pasted from those commands, not hand-written:

```
=== replay ===
Your coffee is $3.50; with the 10% member discount you pay $6.30.

[replayed '20261026-151919-demo-stable': served 4 steps, 0 live calls, $0.00 spent]

=== diff demo-stable demo-drift ===
run A: demo-stable (4 steps)
run B: demo-drift (4 steps)
step 2 (tool:compute_total) output:
  A: 6.3
  B: 5.6
  context (steps 0..1):
    step 0 (model:model.plan) input: []
    step 1 (tool:get_price) input: []
run A: $0.0001
run B: $0.0001
```

The replay served four steps with zero live calls, and the recorded final answer comes back verbatim. The diff walked both traces in lockstep and reported the first divergence: step 2, `tool:compute_total`, output `6.3` became `5.6` — with the matching earlier steps shown as context and the usage/cost of both runs side by side.

The design choice behind this: agent debugging wants the **first fork point**, not an alignment report. The diff aligns runs by position, checking `kind:name` and input depth, rather than fuzzy content-matching one run against the other — content alignment produces an expensive edit-distance report, when what you actually need is where the two runs parted ways, because from the first fork onward everything downstream is unexplainable anyway. Position matching is also what lets the replay error message be precise: "step 2, this call, the input changed from X to Y."

There's also a small CLI (`tapedeck ls` for runs with steps/tokens/cost, `tapedeck report RUN` for per-step detail, `tapedeck prune --keep 20`).

## CI: exit codes do the talking

`tapedeck diff` was built to sit in CI, and its exit codes are the whole interface:

- `0` — the two runs are identical
- `1` — output drift
- `2` — structural drift (steps added or removed)

So the regression gate for an agent is a replay plus a diff. `with tapedeck.play(ref) as tape: run_agent(...)` is a golden replay test in one line, and a diff step whose exit code gates the build answers "did this prompt or model change alter behavior — and at which step?"

Two deliberate defaults before you put this in a pipeline. Redaction is off unless you opt in with `Redactor()` — because running diff and input matching over redacted text would treat two different emails, both masked to `<EMAIL>`, as equal, and wrong data is worse than no data. The default patterns cover emails, API keys, phone numbers, ID numbers and bank cards. And cost is computed by pure multiplication over a price table you own: usage goes into the trace verbatim as the provider returned it, and an unlisted model reports `None` rather than 0, so nothing ever "looks free."

## Honest boundaries

Three things tapedeck cannot do, stated up front:

- **Changes hidden in closures are invisible to replay matching.** Replay matches on call-site inputs. If drift hides in a variable a closure captured — never passing through a parameter — input matching won't flag it. This is exactly why `diff` (comparing two *real* recordings) exists, and why the docs recommend passing behavior-critical parameters explicitly.
- **Deterministic replay holds only up to the I/O layer.** If your agent reads an unseeded RNG or the wall clock, replay may diverge from the recording. tapedeck doesn't guess at seeding for you — this is the shared boundary of every record/replay tool (VCR.py, respx, and friends).
- **No automatic monkeypatching in v0.1.** Auto-intercepting the OpenAI/DeepSeek clients requires per-SDK adapters. v0.1 chooses the explicit `tape.call` — three lines of intrusion — in exchange for zero dependencies and zero magic.

## What's shipping next

On the immediate roadmap: OpenAI/DeepSeek SDK auto-interception adapters, packaged as optional extras so the core stays zero-dependency; a CLI for lenient replay (switch back to live from the first drift point and keep running); tamper-evident trace signing via a hash chain; and a pytest fixture.

## Try it

Repo: <https://github.com/EAnge1o/tapedeck> — MIT, Python 3.10+, `pip install -e .` or literally drop the `tapedeck/` folder into your project. The project is early, which is another way of saying the roadmap is still very much shaped by feedback.

If the tape metaphor is useful to you beyond agents: the sister project [mocktape](https://github.com/EAnge1o/mocktape) applies record-once/replay-offline to HTTP traffic. It records real API responses into readable YAML cassettes for offline frontend/integration testing (it's an npm package, with nearest-tape-match diagnostics when a replay hits a 404). mocktape handles HTTP traffic, tapedeck handles agent decisions; neither knows the other exists, but they often end up in the same test suite.