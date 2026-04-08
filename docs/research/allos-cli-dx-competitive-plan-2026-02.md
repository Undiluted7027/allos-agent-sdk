# Allos-Focused CLI & DX Competitive Plan

Date: 2026-02-24

## Purpose

This document translates the broader tooling landscape into an Allos-specific plan for CLI and DX competitiveness.

It is constrained to **additions/refinements** on top of current architecture:

- Keep `Agent`, `AgentConfig`, `ProviderRegistry`, `ToolRegistry`, and `BaseTool` abstractions.
- Keep current provider model and tool permission model (`ALWAYS_ALLOW`, `ASK_USER`, `ALWAYS_DENY`).
- Improve by filling stubs, extending command surface, and tightening guardrails.

## Baseline: Allos Today (Code-Verified)

### CLI and REPL

- CLI entrypoint with core flags is present in `/Users/sanai/allos-agent-sdk/allos/cli/main.py`.
- Interactive mode exists with minimal slash commands in `/Users/sanai/allos-agent-sdk/allos/cli/interactive.py`.
- Current slash commands: `/help`, `/stream`, `/stream on`, `/stream off`, `/exit`, `/quit`.

### Tool surface (runtime-registered)

- Registered tools today: `edit_file`, `list_directory`, `read_file`, `shell_exec`, `write_file`.
- Registration side effects currently import filesystem + shell only in `/Users/sanai/allos-agent-sdk/allos/tools/__init__.py`.
- Web and MCP tool modules are present but empty stubs:
  - `/Users/sanai/allos-agent-sdk/allos/tools/web/fetch.py`
  - `/Users/sanai/allos-agent-sdk/allos/tools/web/search.py`
  - `/Users/sanai/allos-agent-sdk/allos/tools/mcp/client.py`
  - `/Users/sanai/allos-agent-sdk/allos/tools/mcp/server.py`

### Guardrails and permissions

- Per-tool permission model is implemented in `/Users/sanai/allos-agent-sdk/allos/tools/base.py`.
- `shell_exec` enforces a basic dangerous-command blocklist in `/Users/sanai/allos-agent-sdk/allos/tools/execution/shell.py`.
- Tool approvals are supported via `ASK_USER` and CLI `--auto-approve` behavior.

### Provider and session strengths

- Strong provider abstraction and alias support are already in place (`/Users/sanai/allos-agent-sdk/allos/providers/registry.py`).
- Session save/load exists and is integrated in both one-shot and interactive paths.

## Competitive Matrix with Allos Implications

| Category | Market Baseline (OpenCode/Codex/Cline/Copilot/Gemini/etc.) | Allos Current Status | Gap | Implication for Allos |
|---|---|---|---|---|
| Shell execution | Rich shell semantics, approval scopes, path/url-aware controls | `shell_exec` exists with blocklist + timeout | Medium | Keep shell tool; add policy granularity and safer defaults rather than new architecture |
| File operations | Read/write/edit/delete + patch primitives | Read/write/edit/list present; no patch/delete tools | Medium | Add `delete_file` + `apply_patch_file` tools via existing `BaseTool` pattern |
| Code search/navigation | `grep`/`glob`/symbol tools are table stakes | No registered code-search tools | High | Add `search_files` + `glob_files` as first-class tools to improve agent autonomy |
| Web capabilities | Most tools expose `web_fetch` and often `web_search` | Web modules exist but are empty | High | Fill existing web stubs and register them; this is a direct competitiveness blocker |
| MCP/extensibility | MCP is standard in mature products | MCP modules exist but are empty/unwired | High | Implement minimal MCP client adapter and permission gating without changing agent core |
| REPL command surface | Rich slash command sets (`/model`, `/tools`, `/plan`, `/diff`, etc.) | Only 6 basic commands | High | Expand command registry and expose common operational controls |
| Autocomplete/fuzzy UX | Prompt-toolkit style completion and keyboard navigation is common | No autocomplete currently | High | Implement planned command autocomplete in Phase 4 using `prompt-toolkit` + `rapidfuzz` |
| Safety model ergonomics | Allow/ask/deny policies at tool/path/url granularity | Per-tool permission + global `--auto-approve` | High | Add policy profiles and path/url command policies while preserving `ToolPermission` base model |
| Session ergonomics | Resume, inspect, compact, retry, export are common | Session save/load exists; inspection controls are minimal | Medium | Add slash commands for save/load/status/retry/history and metadata summaries |
| Git/GitHub workflows | Usually shell + MCP GitHub integrations | Shell only; no GitHub-aware helpers | Medium | Add optional GitHub MCP profile and lightweight git-focused slash helpers |
| Workflow orchestration | Todo/plan/task primitives common | No todo/plan REPL primitives | Medium | Add REPL-local todo/plan commands; keep planner optional and non-invasive |

## Prioritized Gap Backlog (Allos-Focused)

### P0: Must-Have to Be Competitive

#### 1) Complete command UX foundation

- Implement command registry and autocomplete exactly where roadmap already points:
  - Use `/Users/sanai/allos-agent-sdk/allos/cli/commands.py` (currently empty)
  - Add `/Users/sanai/allos-agent-sdk/allos/cli/autocomplete.py`
  - Integrate prompt session and keybindings in `/Users/sanai/allos-agent-sdk/allos/cli/interactive.py`
- Add first expanded command set:
  - `/tools`, `/model`, `/provider`, `/status`, `/history`, `/save`, `/load`, `/retry`, `/clear`
- Keep command handlers thin and delegate to existing agent/session/provider APIs.

Acceptance targets:

- Command completion p95 latency < 25ms with 30+ commands.
- Keyboard-only interactive flow works (tab/arrow/enter/ctrl-c).

#### 2) Add code search tools (no architecture change)

- Add `search_files` (`ripgrep`) and `glob_files` tools under `/Users/sanai/allos-agent-sdk/allos/tools/filesystem/` or `/Users/sanai/allos-agent-sdk/allos/tools/execution/`.
- Register via current `@tool` decorator and `ToolRegistry`.
- Default permissions:
  - `search_files`: `ALWAYS_ALLOW`
  - `glob_files`: `ALWAYS_ALLOW`

Acceptance targets:

- Agents can locate symbols/content without shell fallback in common code tasks.
- `--list-tools` displays both tools with clear schema and examples.

#### 3) Fill web tool stubs and register them

- Implement:
  - `/Users/sanai/allos-agent-sdk/allos/tools/web/fetch.py` as `web_fetch`
  - `/Users/sanai/allos-agent-sdk/allos/tools/web/search.py` as `web_search`
- Wire imports in `/Users/sanai/allos-agent-sdk/allos/tools/__init__.py`.
- Add URL allow/deny policy support for web tools.

Acceptance targets:

- Tool list includes web tools.
- Web tools require explicit approval unless host is in safe allowlist profile.

#### 4) Improve safety ergonomics beyond binary auto-approve

- Add configurable policy profiles in CLI/config:
  - `safe` (read-only + ask for shell/write/web)
  - `balanced` (current default-like behavior)
  - `power` (more auto approvals, still deny dangerous commands)
- Keep underlying permission enum; add policy layer in CLI/runtime checks.

Acceptance targets:

- Users can switch policy with one flag.
- Policy behavior is visible via `/status`.

### P1: High-Leverage Next

#### 5) MCP minimal viable integration

- Implement minimal MCP transport/client in existing stubs:
  - `/Users/sanai/allos-agent-sdk/allos/tools/mcp/client.py`
  - `/Users/sanai/allos-agent-sdk/allos/tools/mcp/server.py`
- Start with stdio transport and static server config.
- Add per-server and per-tool approval gates.

Acceptance targets:

- Register and invoke at least one external MCP tool end-to-end.
- Failures surface with actionable errors in CLI.

#### 6) Add REPL observability commands

- `/status`: provider/model/tools/policy/streaming/session path.
- `/last`: print structured summary of last run metadata and tool calls.
- `/context`: show message count and rough token estimate.

Acceptance targets:

- Debugging no longer requires reading raw JSON session files.

#### 7) Git-centric workflow helpers (without hardcoding GitHub APIs)

- Add slash helpers that orchestrate existing tools:
  - `/git status`, `/git diff`, `/git log`
- Keep implementation as wrappers over `shell_exec` + safety profile checks.
- Add optional MCP profile for GitHub if configured.

Acceptance targets:

- Common code-review and patch workflows complete without leaving REPL.

### P2: Differentiators

#### 8) Lightweight planning/todo UX

- Add REPL-local todo primitives (`/todo add`, `/todo list`, `/todo done`).
- Keep state in session context metadata; do not introduce new architecture.

#### 9) Command macros and user shortcuts

- User-defined command aliases in config.
- “Recent commands” ranking feeds autocomplete ordering.

#### 10) Advanced session operations

- `/export markdown|json`
- `/compact` trigger using existing context management components where available.

## Exact File-Level Change Map

### CLI/DX files

- Update `/Users/sanai/allos-agent-sdk/allos/cli/commands.py`: command registry + handlers.
- Add `/Users/sanai/allos-agent-sdk/allos/cli/autocomplete.py`: completer + prompt session.
- Update `/Users/sanai/allos-agent-sdk/allos/cli/interactive.py`: use new registry/completion and new slash handlers.
- Update `/Users/sanai/allos-agent-sdk/allos/cli/main.py`: policy profile flag(s), optional autocomplete toggles, richer status output.

### Tools files

- Update `/Users/sanai/allos-agent-sdk/allos/tools/__init__.py`: include web/mcp side-effect imports.
- Implement `/Users/sanai/allos-agent-sdk/allos/tools/web/fetch.py`.
- Implement `/Users/sanai/allos-agent-sdk/allos/tools/web/search.py`.
- Implement `/Users/sanai/allos-agent-sdk/allos/tools/mcp/client.py` and `/Users/sanai/allos-agent-sdk/allos/tools/mcp/server.py`.
- Add new code-search tool modules and register them.

### Docs and tests

- Update `/Users/sanai/allos-agent-sdk/docs/guides/tools.md` to include real registered tool set.
- Update `/Users/sanai/allos-agent-sdk/docs/reference/cli-reference.md` with expanded slash commands and policy profiles.
- Add unit/e2e tests under:
  - `/Users/sanai/allos-agent-sdk/tests/unit/cli/`
  - `/Users/sanai/allos-agent-sdk/tests/unit/tools/`
  - `/Users/sanai/allos-agent-sdk/tests/e2e/`

## KPIs to Track Competitiveness

- Time-to-first-successful-task in REPL (new user): target < 3 minutes.
- Percent of coding tasks solved without manual shell command by user: target +30% from baseline.
- Approval prompt fatigue (prompts per successful task): target -25% with policy profiles.
- REPL command completion latency p95: < 25ms.
- Crash-free interactive sessions: > 99%.

## Risks and Mitigations

- Risk: Tool sprawl without safety coherence.
  - Mitigation: policy profiles + explicit per-tool docs + default conservative mode.
- Risk: Autocomplete introduces input instability.
  - Mitigation: feature flag, fall back to plain input loop.
- Risk: MCP adds operational complexity.
  - Mitigation: start with one transport and minimal server schema.

## Sources

External landscape references:

- OpenCode docs: https://opencode.ai/docs/
- Kilo docs: https://kilo.ai/docs/getting-started
- GitHub Copilot CLI docs: https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli
- Claude Code docs: https://docs.claude.com/en/docs/claude-code/settings
- Cline docs: https://docs.cline.bot/
- Codex CLI docs: https://developers.openai.com/codex/cli/
- Gemini CLI docs: https://geminicli.com/docs/
- Cursor tools docs: https://docs.cursor.com/en/agent/tools
- VS Code agent tools docs: https://code.visualstudio.com/docs/copilot/agents/agent-tools
- Kiro docs: https://kiro.dev/docs/
- Antigravity docs: https://antigravity.im/agents

Internal repository evidence:

- `/Users/sanai/allos-agent-sdk/docs/research/agent-tooling-landscape-2026-02.md`
- `/Users/sanai/allos-agent-sdk/allos/cli/main.py`
- `/Users/sanai/allos-agent-sdk/allos/cli/interactive.py`
- `/Users/sanai/allos-agent-sdk/allos/tools/__init__.py`
- `/Users/sanai/allos-agent-sdk/allos/tools/execution/shell.py`
- `/Users/sanai/allos-agent-sdk/docs/reference/cli-reference.md`
- `/Users/sanai/allos-agent-sdk/docs/guides/tools.md`
- `/Users/sanai/allos-agent-sdk/POST_MVP_ROADMAPv2.md`
