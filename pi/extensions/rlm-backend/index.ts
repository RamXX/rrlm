// rlm-backend: registers an `rlm_solve` tool that delegates data-heavy subtasks
// to the rrlm RLM-first harness (predict-rlm). The large data payload goes into
// the harness REPL, never into Pi's context window -- which is the entire point.
//
// Install (one of):
//   - symlink this dir into ~/.pi/agent/extensions/, or
//   - add its path to settings.json "extensions", or
//   - run pi with -e /path/to/pi/extensions/rlm-backend/index.ts
//
// Requires the rrlm project venv. Set RRLM_DIR to the project root (defaults to
// the dir two levels above this file). The settled local models must be served
// (LM Studio :1234 + supergemma :8771); override via RRLM_MAIN/RRLM_SUB.

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const HERE = dirname(fileURLToPath(import.meta.url));
const RRLM_DIR = process.env.RRLM_DIR ?? resolve(HERE, "..", "..", "..");
const MAIN_MODEL = process.env.RRLM_MAIN ?? "qwen3.6-27b-official-local";
const SUB_MODEL = process.env.RRLM_SUB ?? "supergemma-26b-local";
const BACKEND = process.env.RRLM_BACKEND ?? "jspi";

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
      "data you can read directly, do NOT use this -- read it yourself.",
    promptGuidelines:
      "Pass the FULL data via `data` (or `data_path` for a file on disk); never " +
      "pre-summarize it. Keep `instruction` specific and answerable from the data.",
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
    // Signature-robust across pi versions: the trailing args (onUpdate/ctx/
    // signal) have shifted between releases, so detect the AbortSignal by shape
    // rather than relying on a fixed position, and skip the progress callback.
    async execute(_id, params, ...rest) {
      const signal = rest.find(
        (a): a is AbortSignal => !!a && typeof a === "object" && "aborted" in a,
      );

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

      try {
        const result = await pi.exec(
          "uv",
          [
            "run",
            "--",
            "python",
            "-m",
            "rrlm.solve",
            "--instruction",
            params.instruction,
            "--data",
            dataArg,
            "--main-model",
            MAIN_MODEL,
            "--sub-model",
            SUB_MODEL,
            "--backend",
            BACKEND,
            "--json",
          ],
          { cwd: RRLM_DIR, signal, timeout: 3_600_000 },
        );

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
