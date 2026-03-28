export interface TaskConfig {
  task_description: string;
  task_type: "general" | "financial";
  examples: Array<{ input: string; expected_output: string }>;
  iterations: number;
  candidates_per_iteration: number;
  initial_prompt: string;
}

export interface TrajectoryEntry {
  prompt: string;
  score: number;
}

export type Trajectory = TrajectoryEntry[];

export interface EvaluationResult {
  input: string;
  expected: string;
  actual: string;
  correct: boolean;
}

export interface CandidateResult {
  prompt_text: string;
  score: number;
  evaluations: EvaluationResult[];
}

export interface IterationResult {
  iteration: number;
  candidates: CandidateResult[];
  best_prompt: string;
  best_score: number;
  trajectory: Trajectory;
}
