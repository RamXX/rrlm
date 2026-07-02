// rlm-backend: registers an `rlm_solve` tool that delegates data-heavy subtasks
// to the rrlm RLM-first harness (predict-rlm). The large data payload goes into
// the harness REPL, never into Pi's context window, which is the entire point.
//
// Models come from your Pi config: by default the harness orchestrates with the
// SAME model Pi is currently using (read from the tool's execution context), and
// resolves credentials/endpoints from ~/.pi/agent/, local, OpenRouter, OpenAI,
// Anthropic, z.ai, etc. Override per the env table below.
//
// Install (one of):
//   - the install script (rrlm is installed from source, not a package index):
//     `curl -fsSL https://raw.githubusercontent.com/RamXX/rrlm/main/install.sh | bash`
//     then point pi at this extension; or
//   - run from a checkout with RRLM_DIR set (uses `uv run` in that project).
//
// Env knobs (rrlm-solve reads these itself; they are inherited by the child
// process, so setting them in Pi's environment is enough):
//   RRLM_MAIN      orchestrator model ref (Pi 'provider/model'); default: Pi's current model
//   RRLM_SUB       leaf model ref for predict() fan-out; default: same as main
//   RRLM_BACKEND   sandbox backend: 'supervisor' (default), 'jspi', or 'sbx'
//   RRLM_WEB       '1' to give the agent live web retrieval (needs the rrlm 'web' extra)
//   RRLM_TIMEOUT   hard wall-clock ceiling in seconds for one rlm_solve call
//   RRLM_MAX_COST  soft USD ceiling per call (cost-reporting providers only)
//   RRLM_DIR       project checkout to run via `uv run` (dev mode); unset = installed rrlm-solve

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const RRLM_DIR = process.env.RRLM_DIR;
const WEB = /^(1|true|yes|on)$/i.test(process.env.RRLM_WEB ?? "");

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
      "data you can read directly, do NOT use this, read it yourself." +
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
      "pre-summarize it. Keep `instruction` specific and answerable from the data." +
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
      // rrlm-solve fall back to Pi's configured default (~/.pi/config.json).
      const mainRef = process.env.RRLM_MAIN ?? modelRef(ctx?.model);
      const subRef = process.env.RRLM_SUB;

      // Stage inline data to a temp file so huge payloads never hit argv limits.
      let dataArg: string;
      let tmpDir: string | null = null;
      if (params.data_path) {
        dataArg = `@${params.data_path}`;
      } else {
        tmpDir = await mkdtemp(join(tmpdir(), "rlm-solve-"));
        const dataFile = join(tmpDir, "data.txt");
        await writeFile(dataFile, params.data ?? "");
        dataArg = `@${dataFile}`;
      }

      // Backend, web access, timeout, and cost ceiling are read by rrlm-solve
      // itself from the inherited RRLM_* environment; only model overrides need
      // to be passed explicitly (Pi's current model is not in the child's env).
      const solveArgs = [
        "--instruction", params.instruction,
        "--data", dataArg,
        ...(mainRef ? ["--main", mainRef] : []),
        ...(subRef ? ["--sub", subRef] : []),
        "--json",
      ];
      // Installed: call rrlm-solve on PATH. Dev: run it inside the checkout via uv.
      const [command, args, options] = RRLM_DIR
        ? ["uv", ["run", "--", "rrlm-solve", ...solveArgs], { cwd: RRLM_DIR, signal, timeout: 3_600_000 }]
        : ["rrlm-solve", solveArgs, { signal, timeout: 3_600_000 }];

      try {
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

        const payload = JSON.parse(result.stdout);
        const u = payload.usage ?? {};
        const summary =
          `model=${payload.config?.main_model ?? "?"} ` +
          `calls=${u.calls ?? 0} ` +
          `tokens=${u.prompt_tokens ?? 0}+${u.completion_tokens ?? 0} ` +
          `wall=${payload.wall_clock_s}s ` +
          `subs=${JSON.stringify(payload.spawn_stats ?? {})}`;
        return {
          content: [{ type: "text", text: payload.answer || "(no answer)" }],
          details: { summary, ...payload },
        };
      } finally {
        if (tmpDir) await rm(tmpDir, { recursive: true, force: true });
      }
    },
  });
}
