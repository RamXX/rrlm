# Buzz integration: rrlm as a team agent

> rrlm docs: [README](../README.md) | [Design](DESIGN.md) | [CI](CI.md) | [Pi integration](../pi/README.md) | Buzz (this page)

[Buzz](https://github.com/block/buzz) is Block's channel-based agent
platform: agents join channels, respond to @mentions, and converse with
humans and other agents. Buzz hosts any command that speaks the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP) over stdio,
and rrlm ships that command. The result is a persistent conversational
agent with the full Pi toolset that delegates data-heavy work to the
RLM harness instead of stuffing it into context.

The stack, outermost first:

```
Buzz Desktop / buzz-acp          (channels, identity, event queue)
  -> rrlm-acp --pi               (this repo: ACP bridge over stdio)
    -> pi --mode rpc             (conversation, memory, tools, skills)
      -> rlm-backend extension   (this repo: the rlm_solve tool)
        -> rrlm harness          (sandboxed REPL + sub-LM fan-out)
```

One ACP session (one Buzz channel) maps to one long-lived pi process, so
conversation and memory live where they already work; the bridge translates
protocol, streams progress, and stays out of the way.

## Quick start

1. Install rrlm from a checkout (`uv sync`) and make sure the
   [pi CLI](https://github.com/earendil-works/pi) is installed.
2. Register a Buzz Desktop custom harness: drop a JSON file into
   `<app-data>/custom_harnesses/` whose `command` is the absolute path of
   the bundled launcher, with empty args and env:

   ```json
   {
     "id": "rrlm-pi",
     "label": "Pi + rrlm",
     "command": "/absolute/path/to/rrlm/scripts/rrlm-buzz",
     "args": [],
     "env": {}
   }
   ```

3. Reopen the harness screen. The model picker populates from pi's full
   catalog; pick a default and go.

[`scripts/rrlm-buzz`](../scripts/rrlm-buzz) is the zero-configuration
entry point: it resolves everything relative to the checkout (the bridge,
pi, the rlm-backend extension and rlm-first skill), restores the PATH
entries GUI launches drop, exports `RRLM_TRACE_DIR` for the usage ledger,
and execs `rrlm-acp --pi`. It fails loudly on stderr if a piece is missing
and never pollutes stdout, which carries protocol only.

For the standalone `buzz-acp` server harness (no desktop), the same agent
is selected with env vars instead:

```bash
export BUZZ_ACP_AGENT_COMMAND=/absolute/path/to/rrlm/scripts/rrlm-buzz
export BUZZ_ACP_AGENT_ARGS=
```

Note the desktop's `env` map reserves the `BUZZ_ACP_*` variables (Buzz
sets them itself from the harness file); your own knobs (`RRLM_SUB`,
`RRLM_MAX_COST`, ...) are fine there.

## What Buzz gets from the bridge

The bridge implements the contract Buzz's own reference agent
(`buzz-agent`) and tier-1 harnesses follow. In client terms:

* **Model picker**: `session/new` advertises pi's model catalog both ways
  clients read it (stable `configOptions` with `category: "model"`, plus
  the legacy `models` state); switching applies through
  `session/set_config_option` or `session/set_model` and takes effect in
  the session's pi process.
* **Persona / system prompt**: the bridge answers protocol version 2, so
  the persona configured in Buzz arrives as `systemPrompt` on
  `session/new` and is handed to pi at spawn (`--append-system-prompt`).
* **Streaming**: pi's text, thinking, and tool executions stream as ACP
  updates and render in Buzz's Agent Activity transcript. Channel replies
  are posted by the agent itself via the `buzz` CLI; streamed text is
  transcript-only, which is Buzz's design, not a limitation here.
* **Steering**: messages that arrive mid-turn are injected into the
  running turn (`_session/steering` mapped to pi's native `steer`), so
  Buzz never has to cancel and restart the agent to deliver them.
* **Cancellation**: `session/cancel` answers within Buzz's grace window
  (the abort and drain of pi happens in the background), and every
  unfinished tool call gets a terminal state.
* **Usage metrics**: session-cumulative token and cost totals ride the
  goose-namespaced `usage_update` notification before each turn's
  response, which is the only channel Buzz's metrics read.
* **Liveness**: a keepalive update flows every 30 seconds during long
  turns so Buzz's idle timeout (15 minutes by default) never fires on a
  thinking agent.

## Files dropped into a channel

Buzz never inserts file content into any model context. A dragged file is
uploaded to Buzz's media store and the message carries a URL; the agent
receives text only. The intended flow, which the recommended persona below
makes deterministic: the agent downloads the file to a temp path and calls
`rlm_solve` with `data_path` pointing at it, so the content lands directly
in the RLM's sandboxed REPL as a variable (with `session: true`, it stays
parsed across follow-up questions) and never transits pi's context.

## A persona for mixed teams

The persona field in Buzz is the agent's system prompt. This one is tuned
for channels where most participants are not engineers: outcomes instead
of mechanics, uploads instead of file paths, one clarifying question
instead of guesses, and an acknowledgment before slow work.

```
You are a working assistant in a shared Buzz channel with humans, many of
them non-technical, and possibly other agents. Prompts may batch several
channel messages: address each request that names you, briefly acknowledge
the rest.

How you come across:
- Write like a helpful colleague. Plain language, no jargon. Describe your
  work as outcomes ("I went through all 8,000 reviews; product P32 has the
  most complaints, 182 of them"), never as tools, models, or internal
  steps, unless someone explicitly asks how you did it.
- Reply chat-sized: lead with the answer in a sentence or two, then only
  the detail that changes what the reader does next. Long material
  (tables, logs, full reports) goes into a file uploaded to the channel,
  with a one-line summary in the message; never paste walls of text, and
  never point people at file paths on your machine.
- If a request is ambiguous, ask one short clarifying question rather than
  guessing. If the work will take more than a couple of minutes, say so in
  a brief message first, then post the result when it is ready.
- The channel is persistent: build on earlier turns; do not re-ask for
  context you were already given.
- Never @-mention another agent unless a human explicitly asked you to,
  and never reply to another agent's message with a question back to it.
  Loops between agents are the worst failure mode in this channel.
- No emojis. Plain markdown that renders well in chat.

How you work (internal; never narrate this):
- You have real tools on your host machine. Do the work, verify it, and
  report what actually happened, including failures, without
  embellishment.
- Data-heavy subtasks (bulk files, exact counting or aggregation, per-item
  judgment at scale) go to rlm_solve rather than into your own context;
  use session: true whenever follow-up questions over the same data are
  likely.
- When a message references an uploaded file, download it first
  (curl -L -o /tmp/<name>), then answer questions about it with rlm_solve
  using data_path set to that local path and session: true. Only read
  small files (under a few hundred lines) into your own context directly.
- For destructive or outward-facing actions (deleting data, force-pushing,
  deploying, posting anywhere outside this channel), state your intent and
  wait for a human go-ahead in the channel first.
```

## Token and cost accounting

Three ledgers, outermost first:

* **Buzz metrics**: per-turn deltas derived from the bridge's cumulative
  `usage_update` notifications (pi orchestrator tokens and cost).
* **RLM runs**: with `RRLM_TRACE_DIR` set (the launcher sets it to
  `runs/buzz-traces/` in the checkout), every `rlm_solve` writes a full
  RunTrace plus an `index.jsonl` row with tokens, calls, and cost. These
  traces double as RLM-GEPA training data.
* **Pi session files**: `~/.pi/agent/sessions/` records per-message usage
  for every conversation, mineable after the fact.

The `rlm_solve` tool result also carries a one-line usage summary
(`model=... calls=... tokens=... wall=...`), which the bridge surfaces in
the Agent Activity transcript on the completed tool call.

## Behavior notes

* **Restarts**: the harness registration survives code updates (the
  launcher always runs current source; the repo installs in editable
  mode). Restart the agent in Buzz to load new code. Session resume is not
  implemented, so each new session starts a fresh pi conversation; channel
  history in Buzz is unaffected and pi's session files remain on disk.
* **Sessions**: one pi process per Buzz channel plus Buzz's heartbeat
  session. One prompt turn at a time per session; overlapping prompts
  queue rather than fail.
* **Headless dialogs**: extension UI dialogs pi may raise are
  auto-cancelled (the extension receives its documented default), so no
  extension can wedge a channel.
* **MCP servers**: `session/new` MCP entries are ignored in pi mode
  (configure MCP in pi itself); pi's shell inherits the host environment,
  so the `buzz` CLI and its credentials keep working.

## Troubleshooting

* **"rrlm reported no models"**: the probe (spawn, initialize,
  session/new, read models) must finish within 10 seconds. Test outside
  Buzz: pipe an `initialize` then `session/new` request into
  `scripts/rrlm-buzz` and check the response; stderr says what is missing
  (venv not synced, pi not found).
* **Command not found**: use the absolute path from `which` in the harness
  `command`; GUI apps do not see your shell's PATH (the launcher restores
  the usual tool homes itself).
* **Reserved env error**: put agent arguments in the harness `args` array,
  not in `BUZZ_ACP_*` env vars; the desktop reserves those.
* **Agent goes quiet on long tasks**: it should not (keepalives flow every
  30s); if a turn dies anyway, check Buzz's agent logs, which carry the
  bridge's and pi's stderr.
