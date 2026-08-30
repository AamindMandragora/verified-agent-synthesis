# AGENTS.md — `synthesis/generate/`

## Scope

Strategy **generation and refinement** prompts and orchestration.

## Rules

- Follow root **`AGENTS.md`** Critical Prompting Rule: neutral tool/API reference is allowed as contract content, but prompts must not add strategy guidance.
- When documenting `AppendTaskGuidance`, keep it as a neutral helper contract:
  append-only, first-call-wins, start-of-CSD placement only. The guidance
  string may describe task-semantic meaning or numeric conventions the grammar
  does not encode; it must not contradict, weaken, or replace earlier task
  instructions, examples, schema, or output-format requirements; do not add
  preferred strategies, benchmark tips, or
  “use when accuracy is low” coaching.
- Changes to **`prompts.py`** affect every synthesis run; keep diffs minimal and auditable.
- **`generator.py`** coordinates LLM calls and failure feedback; avoid embedding benchmark-specific hacks here (delegate via feedback shape or benchmark modules).
- Matrix model ablations must use direct thinking-mode hosted profiles. Do not
  route matrix profiles through Bedrock or Bedrock-backed Gemini placeholders.
- Use the `gemini` matrix profile for direct Gemini API model ablations. The
  legacy `gemini-pro` profile name remains rejected because it historically
  referred to a Bedrock-backed placeholder.
- Keep direct Gemini and Vertex API-key quota fallback in code, not prompt
  prose: rotate through `GEMINI_API_KEY_BACKUP_N` keys on quota exhaustion and
  avoid delayed retries on a key that has already reported quota exhaustion.
- The `codex` author backend must use the pinned Pi provider layer with
  ChatGPT/Codex OAuth and fixed `gpt-5.6-sol`. Call the model provider directly
  with the campaign system instructions and one user message. Do not start a Pi
  agent session, expose tools, carry conversation state, use `OPENAI_API_KEY`,
  or invoke `codex exec`.
- Keep Bedrock as an explicit low-level backend. Do not require `BEDROCK_BASE_URL`
  for client-mode AWS Converse calls; derive regional HTTP runtime URLs lazily
  only when the HTTP fallback path is used.
- If prompt context needs rationale compression, use the rationale-summary path
  or preserve the full rationale. Do not replace rationale claims with
  character/word truncation in model-facing refinement context.

## See also

- **`README.md`** in this folder for file roles and I/O contract.
