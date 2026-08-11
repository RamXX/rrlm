// rlm-backend: registers an `rlm_solve` tool that delegates data-heavy subtasks
// to the rrlm RLM-first harness (predict-rlm). The large data payload goes into
// the harness REPL, never into Pi's context window, which is the entire point.
//
// Two execution modes:
//   * one-shot (default): each call runs `rrlm-solve` as its own subprocess.
//   * session (`session: true`): calls run inside ONE persistent `rrlm-session`
//     subprocess holding a live REPL namespace, so variables and parsed data
//     from one call are available to the next. The conversation (Pi) supplies
//     the dialogue; the session supplies the computed state.
//
// Models come from your Pi config: by default the harness orchestrates with the
// SAME model Pi is currently using (read from the tool's execution context), and
// resolves credentials/endpoints from ~/.pi/agent/, local, OpenRouter, OpenAI,
// Anthropic, z.ai, etc. Override per the env table below. Switching Pi to a
// different model restarts the session (fresh namespace) so the session's
// orchestrator always matches the conversation's model.
//
// Install (one of):
//   - the install script (rrlm is installed from source, not a package index):
//     `curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash`
//     then point pi at this extension; or
//   - run from a checkout with RRLM_DIR set (uses `uv run` in that project).
//
// Env knobs (rrlm reads these itself; they are inherited by child processes,
// so setting them in Pi's environment is enough):
//   RRLM_MAIN      orchestrator model ref (Pi 'provider/model'); default: Pi's current model
//   RRLM_SUB       leaf model ref for predict() fan-out; default: same as main
//   RRLM_BACKEND   sandbox backend: 'supervisor' (default), 'jspi', or 'sbx'
//   RRLM_WEB       '1' to give the agent live web retrieval (needs the rrlm 'web' extra)
//   RRLM_TIMEOUT   hard wall-clock ceiling in seconds for one rlm_solve call
//   RRLM_MAX_COST  soft USD ceiling per call (cost-reporting providers only)
//   RRLM_DIR       project checkout to run via `uv run` (dev mode); unset = installed CLIs

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const RRLM_DIR = process.env.RRLM_DIR;
const WEB = /^(1|true|yes|on)$/i.test(process.env.RRLM_WEB ?? "");
// Generous ceiling for one call; the Python side enforces RRLM_TIMEOUT itself,
// this only prevents an unbounded hang if the child stops responding entirely.
const CALL_TIMEOUT_MS = 3_600_000;

// Best-effort: turn Pi's current Model object into an rrlm model reference
// (provider/id). Defensive about the provider field shape across pi versions.
function modelRef(model: unknown): string | undefined {
  if (!model || typeof model !== "object") return undefined;
  const m = model as { id?: unknown; provider?: unknown };
  const id = typeof m.id === "string" ? m.id : undefined;
  let provider: string | undefined;
  if (typeof m.provider === "string") provider = m.provider;
  else if (m.provider && typeof m.provider === "object") {
    const p = m.provider as { name?: unknown; id?: unknown };
    provider = (typeof p.name === "string" && p.name) || (typeof p.id === "string" && p.id) || undefined;
  }
  if (!id) return undefined;
  return provider ? `${provider}/${id}` : id;
}

// ---------------------------------------------------------------------------
// Persistent session: one rrlm-session child per extension instance, speaking
// line-delimited JSON (see src/rrlm/session_server.py for the protocol).
// ---------------------------------------------------------------------------

type Pending = { resolve: (response: SessionResponse) => void; reject: (err: Error) => void };
type SessionResponse = { id: unknown; result?: Record<string, unknown>; error?: string };

let sessionChild: ChildProcessWithoutNullStreams | null = null;
let sessionModelKey = "";
let sessionBuffer = "";
let nextRequestId = 1;
const pending = new Map<number, Pending>();

function killSession(reason: string): void {
  const child = sessionChild;
  sessionChild = null;
  sessionBuffer = "";
  for (const waiter of pending.values()) waiter.reject(new Error(`rlm session ${reason}`));
  pending.clear();
  if (child && child.exitCode === null) child.kill();
}

function ensureSession(mainRef?: string, subRef?: string): ChildProcessWithoutNullStreams {
  const key = `${mainRef ?? ""}|${subRef ?? ""}`;
  if (sessionChild && sessionModelKey !== key) {
    // The conversation's model changed; a session's orchestrator is fixed at
    // start, so restart with the new model (namespace starts fresh).
    killSession("restarted for a model change");
  }
  if (sessionChild) return sessionChild;

  const sessionArgs = [
    ...(mainRef ? ["--main", mainRef] : []),
    ...(subRef ? ["--sub", subRef] : []),
  ];
  const [command, args, options] = RRLM_DIR
    ? ["uv", ["run", "--", "rrlm-session", ...sessionArgs], { cwd: RRLM_DIR }]
    : ["rrlm-session", sessionArgs, {}];
  // stderr inherits so rrlm warnings/logs land in Pi's log, never in protocol.
  const child = spawn(command as string, args as string[], {
    ...(options as object),
    stdio: ["pipe", "pipe", "inherit"],
  });
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    sessionBuffer += chunk;
    let newline: number;
    while ((newline = sessionBuffer.indexOf("\n")) >= 0) {
      const line = sessionBuffer.slice(0, newline).trim();
      sessionBuffer = sessionBuffer.slice(newline + 1);
      if (!line) continue;
      let response: SessionResponse;
      try {
        response = JSON.parse(line);
      } catch {
        continue; // not a protocol line; ignore rather than desynchronize
      }
      const waiter = typeof response.id === "number" ? pending.get(response.id) : undefined;
      if (waiter) {
        pending.delete(response.id as number);
        waiter.resolve(response);
      }
    }
  });
  child.on("exit", () => {
    if (sessionChild === child) killSession("exited");
  });
  child.on("error", () => {
    if (sessionChild === child) killSession("failed to start (is rrlm-session on PATH?)");
  });
  sessionChild = child;
  sessionModelKey = key;
  return child;
}

function sessionRequest(
  child: ChildProcessWithoutNullStreams,
  payload: Record<string, unknown>,
  signal: AbortSignal | undefined,
): Promise<SessionResponse> {
  const id = nextRequestId++;
  return new Promise<SessionResponse>((resolve, reject) => {
    // An abort kills the whole session: the Python side is mid-run and has no
    // cancel channel, so losing the namespace is the correct cost of a cancel.
    const onAbort = () => killSession("aborted");
    const timer = setTimeout(() => killSession("timed out"), CALL_TIMEOUT_MS);
    pending.set(id, {
      resolve: (response) => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        resolve(response);
      },
      reject: (err) => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        reject(err);
      },
    });
    signal?.addEventListener("abort", onAbort, { once: true });
    child.stdin.write(JSON.stringify({ id, ...payload }) + "\n");
  });
}

// ---------------------------------------------------------------------------

type SolvePayload = {
  answer?: unknown;
  error?: string | null;
  wall_clock_s?: number;
  spawn_stats?: Record<string, unknown>;
  usage?: Record<string, number>;
  config?: Record<string, string>;
};

function formatResult(payload: SolvePayload) {
  const u = payload.usage ?? {};
  const summary =
    `model=${payload.config?.main_model ?? "?"} ` +
    `calls=${u.calls ?? 0} ` +
    `tokens=${u.prompt_tokens ?? 0}+${u.completion_tokens ?? 0} ` +
    `wall=${payload.wall_clock_s}s ` +
    `subs=${JSON.stringify(payload.spawn_stats ?? {})}`;
  const answer = payload.answer;
  const text = typeof answer === "string" ? answer : JSON.stringify(answer);
  return {
    content: [{ type: "text" as const, text: text || "(no answer)" }],
    details: { summary, ...payload },
  };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "rlm_solve",
    label: "RLM Solve",
    description:
      "Delegate a data-heavy subtask to the RLM-first harness. The data is " +
      "loaded into a sandboxed REPL (NOT this conversation's context); the " +
      "harness writes code to probe it, fans out cheap sub-LM calls only for " +
      "irreducible semantic judgment, verifies, and returns the answer. Use " +
      "when the data is large, exact aggregation/search over many items is " +
      "required, or per-item semantic judgment is needed at scale. For small " +
      "data you can read directly, do NOT use this, read it yourself. " +
      "With session=true, computed state (variables, parsed data) persists " +
      "across rlm_solve calls in this conversation, so follow-up questions " +
      "can build on earlier work instead of re-parsing the data." +
      (WEB
        ? " WEB ACCESS IS ENABLED: this harness can also research the LIVE WEB " +
          "(it has web_search and fetch tools and will search, fetch the source, " +
          "extract, and verify). Use it for factual or current-events questions " +
          "you cannot answer with certainty from memory, or whenever a cited " +
          "source URL is required, even with NO data payload. Prefer delegating " +
          "such lookups here over answering them from memory."
        : ""),
    promptGuidelines:
      "Pass the FULL data via `data` (or `data_path` for a file on disk); never " +
      "pre-summarize it. Keep `instruction` specific and answerable from the data. " +
      "When you expect follow-up questions over the same data, set `session: true` " +
      "on every related call and tell the harness what to KEEP (e.g. 'parse the " +
      "ledger into `entries`'), then reference those names in later instructions. " +
      "Session calls after the first can omit the data entirely." +
      (WEB
        ? " For a live web lookup, leave data empty and put the question in " +
          "`instruction` (ask it to cite the source URL)."
        : ""),
    parameters: Type.Object({
      instruction: Type.String({
        description: "What to accomplish, answerable from the data alone.",
      }),
      data: Type.Optional(
        Type.String({ description: "The data payload (inline). Use data_path for files." }),
      ),
      data_path: Type.Optional(
        Type.String({ description: "Absolute path to a data file, instead of inline data." }),
      ),
      session: Type.Optional(
        Type.Boolean({
          description:
            "Run inside the conversation's persistent REPL session: variables and " +
            "parsed data from earlier session calls stay available; name them in the " +
            "instruction. Use for follow-up questions over the same data.",
        }),
      ),
      reset_session: Type.Optional(
        Type.Boolean({
          description: "Clear the persistent session's computed state before this call.",
        }),
      ),
    }),
    // Signature-robust across pi versions: the trailing args (signal/onUpdate/ctx)
    // have shifted between releases, so detect the AbortSignal and the execution
    // context (which carries `.model`) by shape rather than by position.
    async execute(_id, params, ...rest) {
      const signal = rest.find(
        (a): a is AbortSignal => !!a && typeof a === "object" && "aborted" in a,
      );
      const ctx = rest.find(
        (a): a is { model?: unknown } =>
          !!a && typeof a === "object" && ("model" in a || "modelRegistry" in a),
      );

      // Orchestrator model: explicit override, else Pi's current model, else let
      // rrlm fall back to Pi's configured default (~/.pi/config.json).
      const mainRef = process.env.RRLM_MAIN ?? modelRef(ctx?.model);
      const subRef = process.env.RRLM_SUB;

      // Stage inline data to a temp file so huge payloads never hit argv or
      // protocol-line limits; both modes read the file themselves.
      let dataFile: string | null = null;
      let tmpDir: string | null = null;
      if (params.data_path) {
        dataFile = params.data_path;
      } else if (params.data) {
        tmpDir = await mkdtemp(join(tmpdir(), "rlm-solve-"));
        dataFile = join(tmpDir, "data.txt");
        await writeFile(dataFile, params.data);
      }

      try {
        if (params.session) {
          const child = ensureSession(mainRef, subRef);
          if (params.reset_session) {
            await sessionRequest(child, { op: "reset" }, signal);
          }
          const response = await sessionRequest(
            child,
            {
              op: "solve",
              instruction: params.instruction,
              ...(dataFile ? { data_file: dataFile } : {}),
            },
            signal,
          );
          if (response.error) {
            return {
              content: [{ type: "text", text: `rlm_solve (session) failed: ${response.error}` }],
              details: response,
              isError: true,
            };
          }
          return formatResult(response.result as SolvePayload);
        }

        // One-shot mode: unchanged - each call is its own rrlm-solve process.
        const solveArgs = [
          "--instruction", params.instruction,
          "--data", dataFile ? `@${dataFile}` : "",
          ...(mainRef ? ["--main", mainRef] : []),
          ...(subRef ? ["--sub", subRef] : []),
          "--json",
        ];
        const [command, args, options] = RRLM_DIR
          ? ["uv", ["run", "--", "rrlm-solve", ...solveArgs], { cwd: RRLM_DIR, signal, timeout: CALL_TIMEOUT_MS }]
          : ["rrlm-solve", solveArgs, { signal, timeout: CALL_TIMEOUT_MS }];
        const result = await pi.exec(command as string, args as string[], options as object);

        if (result.code !== 0) {
          return {
            content: [
              { type: "text", text: `rlm_solve failed (exit ${result.code}):\n${result.stderr}` },
            ],
            details: result,
            isError: true,
          };
        }
        return formatResult(JSON.parse(result.stdout) as SolvePayload);
      } finally {
        if (tmpDir) await rm(tmpDir, { recursive: true, force: true });
      }
    },
  });
}
