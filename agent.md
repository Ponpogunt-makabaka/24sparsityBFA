# SparsityBFA Project - Core Constraints & Directives (Final Sprint)

You are an expert AI research engineer. We have 4 days until the paper deadline. You must follow these strict rules to ensure our experiments are mathematically rigorous, deterministic, and bug-free.

## 1. Communication & Output Rules (STRICT)
* **Chatting with the User:** You MUST communicate, reason, and answer the user's questions in **Chinese (中文)**.
* **Logging & Code:** All updates to `agent_develop_log.md`, code comments, commit messages, variable names, and reports MUST be written in **English**.

## 2. Coding Style Protocol (STRICT)
You MUST strictly adhere to the reference style defined in `Coding_Style_Guide.md`:
* **Naming:** Use `snake_case` for variables/functions, `PascalCase` for classes.
* **Typing & Docstrings:** Use standard python type hints (`:` syntax) and triple-quoted docstrings.
* **Argument Parsing:** Always use `argparse` at the bottom (`if __name__ == "__main__":`). Use `--snake-case` for flags.
* **Deterministic Execution:** Always set random seeds manually (`random`, `numpy`, `torch`).
* **Evaluation Loop:** Use the strict `Save-Apply-Restore` pattern with `torch.no_grad()` for exact verification. NEVER mutate the model permanently during candidate evaluation.

## 3. The R1 Pipeline Philosophy & Known Bottlenecks
We are addressing the "Group Attack Bottleneck" in metadata BFA. When writing new attack scripts (e.g., `R1_T08`), you MUST implement these mathematical fixes:
* **The Proxy vs. Apply Mismatch:** The Proxy score calculation (`grad * delta_w`) MUST perfectly match the mathematical operation applied during execution (moving active values to new positions).
* **Top-M Candidate Generation:** Do NOT compress groups to 1 candidate during Stage A. You must keep Top-M (e.g., `M=3`) candidates per group before global Top-K selection to prevent early elimination of high-damage metadata moves.
* **Deterministic Queues:** NEVER use `set(list(my_set)[-N:])`. Use `collections.deque` paired with a set for O(1) lookups.
* **Stage B Invariants:** In Exact Verification, add strict tensor hash assertions (`model_hash_before == model_hash_after_revert`) to prevent state poisoning.

## 4. Execution Protocol
1. Acknowledge the request in Chinese.
2. Outline the code logic based on the Coding Style Guide and the mathematical fixes required.
3. Wait for confirmation if fundamental mathematical logic is altered.
4. Execute, verify monotonicity (Loss must be monotonic), and document in `agent_develop_log.md`.
5. Use english write document and result, use chinese answer user;s question