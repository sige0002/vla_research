# Workspace Rules

## Prohibited Operations

The following destructive operations are strictly forbidden:

- `rm -rf /workspace` or `rm -rf /workspace/*` - Do NOT delete the entire workspace
- `rm -rf .` or `rm -rf *` when in /workspace - Do NOT recursively delete workspace contents
- `> /workspace/<file>` to truncate important files without backup
- `mv /workspace /somewhere-else` - Do NOT move the workspace directory

## Firewall Protection

The following commands are prohibited (they affect the network outside the container):

- `sudo iptables ...` - Do NOT modify IPv4 firewall rules
- `sudo ip6tables ...` - Do NOT modify IPv6 firewall rules
- `sudo ipset ...` - Do NOT modify IP sets
- `sudo nft ...` - Do NOT modify nftables rules
- Any attempt to disable, flush, or modify the firewall configuration

## File Protection Rules

- Before deleting any file, confirm the operation is intentional and targeted
- Never use `rm -rf` with wildcard patterns on /workspace without explicit user confirmation
- Do not modify or delete `.git` directories unless explicitly asked
- Preserve CLAUDE.md - do not delete or empty this file

## General Safety

- Prefer `rm` over `rm -rf` when deleting individual files
- Use `git` for version control rather than manual file management
- When cleaning up, remove specific files by name rather than using broad patterns

## Agent Coordination Rules

- Parallel agent invocation: **基本3体、最大4体まで**
- 1回のメッセージで同時に召喚するエージェント（Task tool）は基本3体とする
- 特別な理由がある場合に限り最大4体まで許容する
- 5体以上の同時召喚は禁止
- Pythonを使う場合はuvの仮想環境を利用すること．ローカル環境へのpip installは禁止．

## Development Workflow
- Use plan mode (Ctrl+Shift+P) for architecture decisions before coding
- Use "think hard" for complex implementations
- Use "ultrathink" for algorithm design from papers
- Always write tests alongside implementation
- Run /compact when context reaches 60%

## Python Environment
- MUST use uv for all Python package management
- Create venv: uv venv .venv && source .venv/bin/activate
- Install packages: uv pip install <package>
- NEVER use bare pip install

## Quality Cycle
- After implementation: run tests
- After tests pass: security review
- After security: code simplification check
- Continuously verify efficiency and accuracy improvements

## Skills Available
- /implement-app - Application development workflow
- /implement-algorithm - Paper algorithm implementation
- /modify-vlm - VLM modification workflow
- /modify-oss - OSS modification workflow
- /security-check - Security review for installations
