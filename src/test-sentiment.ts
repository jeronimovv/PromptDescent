/**
 * Benchmark: Movie Review Sentiment Classification
 * Task type: general
 * Demonstrates OPRO optimizing a binary sentiment classifier over 5 iterations.
 */

import Anthropic from "@anthropic-ai/sdk";
import { runOPRO } from "./opro-engine.js";
import type { TaskConfig, IterationResult } from "./types.js";

const task: TaskConfig = {
  task_description: "Classify the sentiment of a movie review as either positive or negative.",
  task_type: "general",
  initial_prompt: "Classify the sentiment of this movie review.",
  iterations: 5,
  candidates_per_iteration: 3,
  examples: [
    {
      input: "This film was an absolute masterpiece. The acting was superb and the story kept me engaged throughout.",
      expected_output: "positive",
    },
    {
      input: "I walked out after 30 minutes. Terrible plot, wooden acting, complete waste of time.",
      expected_output: "negative",
    },
    {
      input: "A beautiful and moving story that will stay with you long after the credits roll.",
      expected_output: "positive",
    },
    {
      input: "The special effects were the only good thing. Everything else was a disaster.",
      expected_output: "negative",
    },
    {
      input: "Brilliant performances from the entire cast. One of the best films of the decade.",
      expected_output: "positive",
    },
    {
      input: "Painfully slow pacing and a nonsensical ending. I cannot recommend this film.",
      expected_output: "negative",
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
  console.log("║         Task: Movie Review Sentiment Classification       ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
  console.log(`\nTask: ${task.task_description}`);
  console.log(`Initial prompt: "${task.initial_prompt}"`);
  console.log(`Iterations: ${task.iterations} | Candidates/iter: ${task.candidates_per_iteration}`);
  console.log(`Evaluation set: ${task.examples.length} examples`);

  const initialScore = { value: 0 };
  let lastResult: IterationResult | null = null;

  for await (const result of runOPRO(task, client)) {
    lastResult = result;

    if (result.iteration === 1) {
      // Initial score was computed before iteration 1; capture from trajectory
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
    console.log(`\nFull trajectory (τ_T):`);
    for (const entry of lastResult.trajectory) {
      console.log(`  [${(entry.score * 100).toFixed(0).padStart(3)}%] "${entry.prompt.slice(0, 70)}${entry.prompt.length > 70 ? "..." : ""}"`);
    }
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
