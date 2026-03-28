/**
 * Benchmark: Earnings Call Sentiment Classification (Bullish / Bearish / Neutral)
 * Task type: financial
 *
 * Production NLP task in quantitative finance. Demonstrates OPRO optimizing
 * a three-class financial sentiment classifier over domain-specific text.
 *
 * f(p) is evaluated over realistic earnings call language — the same
 * distribution encountered in systematic equity signal generation.
 */

import Anthropic from "@anthropic-ai/sdk";
import { runOPRO } from "./opro-engine.js";
import type { TaskConfig, IterationResult } from "./types.js";

const task: TaskConfig = {
  task_description:
    "Classify the financial sentiment of an earnings call excerpt as bullish, bearish, or neutral.",
  task_type: "financial",
  initial_prompt: "Classify the financial sentiment of this earnings call excerpt.",
  iterations: 5,
  candidates_per_iteration: 3,
  examples: [
    {
      input:
        "Revenue exceeded consensus estimates by 12%, driven by strong enterprise demand and record subscription growth.",
      expected_output: "bullish",
    },
    {
      input:
        "Management guided Q4 EPS below Street expectations citing macro headwinds and softening consumer demand.",
      expected_output: "bearish",
    },
    {
      input:
        "Results were in line with guidance across all segments. No change to full-year outlook.",
      expected_output: "neutral",
    },
    {
      input:
        "Free cash flow conversion improved 800 basis points year-over-year on working capital optimization and disciplined capex.",
      expected_output: "bullish",
    },
    {
      input:
        "The company disclosed an SEC inquiry into revenue recognition practices in the enterprise segment.",
      expected_output: "bearish",
    },
    {
      input:
        "Average selling prices remained stable quarter-over-quarter across all product lines and geographies.",
      expected_output: "neutral",
    },
  ],
};

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("Error: ANTHROPIC_API_KEY environment variable is not set.");
    process.exit(1);
  }

  const client = new Anthropic({ apiKey });

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║         PromptDescent — OPRO Benchmark                   ║");
  console.log("║         Task: Earnings Call Sentiment (Quant Finance)     ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
  console.log(`\nTask: ${task.task_description}`);
  console.log(`Initial prompt: "${task.initial_prompt}"`);
  console.log(`Iterations: ${task.iterations} | Candidates/iter: ${task.candidates_per_iteration}`);
  console.log(`Evaluation set: ${task.examples.length} examples`);
  console.log(`\nNote: f(p) = exact-match accuracy over {bullish, bearish, neutral}`);
  console.log(`      Each evaluation is an independent LLM call — no cross-contamination.`);

  const initialScore = { value: 0 };
  let lastResult: IterationResult | null = null;

  for await (const result of runOPRO(task, client)) {
    lastResult = result;

    if (result.iteration === 1) {
      initialScore.value = result.trajectory[0].score;
    }

    console.log(`\n  ── Iteration ${result.iteration} Summary ──`);
    console.log(`  Candidates evaluated:`);
    for (const c of result.candidates) {
      console.log(`    [${(c.score * 100).toFixed(0).padStart(3)}%] "${c.prompt_text.slice(0, 70)}${c.prompt_text.length > 70 ? "..." : ""}"`);
    }
    console.log(`  Trajectory length: ${result.trajectory.length}`);
    console.log(`  Current best score: ${(result.best_score * 100).toFixed(1)}%`);
  }

  if (lastResult) {
    const delta = lastResult.best_score - initialScore.value;
    console.log("\n╔══════════════════════════════════════════════════════════╗");
    console.log("║                    FINAL RESULTS                         ║");
    console.log("╚══════════════════════════════════════════════════════════╝");
    console.log(`\nInitial prompt:  "${task.initial_prompt}"`);
    console.log(`Initial score:   ${(initialScore.value * 100).toFixed(1)}%`);
    console.log(`\nOptimized prompt: "${lastResult.best_prompt}"`);
    console.log(`Final score:      ${(lastResult.best_score * 100).toFixed(1)}%`);
    console.log(`\nΔ Score: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} percentage points`);
    console.log(`\nFull trajectory τ_T (ascending sort = improvement direction):`);
    const sorted = [...lastResult.trajectory].sort((a, b) => a.score - b.score);
    for (const entry of sorted) {
      console.log(`  [${(entry.score * 100).toFixed(0).padStart(3)}%] "${entry.prompt.slice(0, 70)}${entry.prompt.length > 70 ? "..." : ""}"`);
    }

    // Per-example breakdown of best prompt
    console.log(`\nPer-example evaluation of best prompt:`);
    const bestIteration = lastResult.candidates.reduce(
      (best, c) => (c.score > best.score ? c : best),
      lastResult.candidates[0]
    );
    if (bestIteration?.evaluations?.length) {
      for (const ev of bestIteration.evaluations) {
        const mark = ev.correct ? "✓" : "✗";
        console.log(`  ${mark} expected="${ev.expected}" actual="${ev.actual}" | "${ev.input.slice(0, 50)}..."`);
      }
    }
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
