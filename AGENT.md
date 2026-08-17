When modifying code:

Prefer deletion and reuse over addition.

For every requested change:

1. Search for an existing implementation before adding code.
2. Do not duplicate logic for logging, validation, statistics, scoring, or fallback.
3. A new helper function is justified only if it removes at least as much duplicated code as it adds.
4. A new abstraction is justified only if it reduces total complexity or total LOC.
5. Do not add a new module merely to keep an individual file short.
6. Preserve a package-level LOC budget for attack/mos*.py.
7. If a change increases MOS production LOC by more than 50 lines, stop and explain why before implementing.
8. If total MOS LOC exceeds the budget, refactor/delete existing code before adding new behavior.

YAGNI rule:

Do not implement:
- hypothetical future extensions
- unused configuration options
- unused diagnostics
- compatibility wrappers for callers that do not actually exist
- generic plugin systems unless there are at least two real implementations that require the abstraction
- fallback branches for failure modes not observed or required by current tests

Search the repository first.
Only preserve compatibility that is demonstrably used.