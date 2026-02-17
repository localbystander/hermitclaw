# ULTRAZHINK Agent Harness Security Review — HermitClaw

## 1) System Map

### Components diagram (text)

```text
[Browser UI]
   |  HTTP+WS (no auth)
   v
[FastAPI server (server.py)]
   |--> /api/message, /api/focus-mode, /api/files, /api/raw, /ws
   |--> static frontend serving
   v
[Brain orchestrator (brain.py)]
   |--> Builds prompts (prompts.py)
   |--> Calls LLM provider client (providers.py)
   |--> Executes tools (tools.py)
             |--> shell -> subprocess.run(shell=True)
             |--> python shell rewrite -> pysandbox.py
             |--> fetch_url/web_search/web_fetch -> network egress
   v
[Environment folder *_box/]
   |--> user drop files, generated files, memory stream
   v
[Host OS + network + secrets]
```

### Trust boundaries and privilege levels

1. **Internet/user input -> model context**: web results, fetched URLs, user chat, and dropped files are injected into model context with no policy firewall.
2. **Model output -> tool execution**: model can issue `shell`, `fetch_url`, and web tooling calls; shell calls are executed through `subprocess.run(..., shell=True)`.
3. **App process -> host resources**: process runs with user privileges, can access env vars, project files, and outbound network from non-sandbox tool paths.
4. **Remote clients -> control plane**: API/WS has no authentication; anyone with network reach can influence agent state and read generated content.

### Data flow highlights

- **Secrets path**: environment vars/API keys loaded in config and used by providers/tool calls.
- **Prompt/tool-output path**: tool output and user input are appended back into model input.
- **Network path**: `fetch_url`, `web_search`, `web_fetch`, and model API calls all perform egress.
- **Logging path**: full-ish API interactions are appended to `hermitclaw.log.jsonl`.

---

## 2) Attack Surface Inventory

### Entry points

- HTTP endpoints: `/api/*` including message injection, file listing/reading, raw model I/O history.
- WebSocket endpoints: `/ws`, `/ws/{crab_id}` for live event stream.
- Agent-controlled tool inputs: shell command strings, URLs for fetch tools.
- User-dropped files and web content fed into prompt context.
- Dependency/runtime surface via Python packages and npm frontend dependencies.

### Privilege-bearing operations

- Command execution via `subprocess.run(shell=True)`.
- Python execution with partial monkey-patch sandbox.
- Filesystem read/write inside agent loop and API file endpoints.
- Unrestricted outbound network from non-sandbox code paths.
- Runtime brain creation and lifecycle controls via API.

### User/agent-controlled strings sink locations

- Prompt/user/tool strings -> model context (prompt-injection risk).
- `shell.command` -> shell interpreter.
- `fetch_url.url` -> backend HTTP client (SSRF).
- `path` in file API endpoint -> filesystem reads (guarded only by realpath prefix check).

---

## 3) Findings Table

| ID | Title | Severity | Exploitability | Where (file:function) | What goes wrong | Exploit narrative (short) | Fix (short) | Suggested test |
|---|---|---|---|---|---|---|---|---|
| HC-001 | Shell blocklist bypass enables arbitrary command chains | CRITICAL | Low preconditions / high reliability / host-level blast radius | `hermitclaw/tools.py:run_command,_is_safe_command` | Prefix-only checks are bypassable; `shell=True` executes chained payloads | Agent emits benign prefix then `;` dangerous command to run host commands/exfil | Replace string blocklist with strict allowlist + argv execution (`shell=False`) | Unit tests with separators, env-prefix, quoting bypass cases |
| HC-002 | Python sandbox can read outside env via unpatched low-level APIs | CRITICAL | Moderate complexity / high reliability / host-level blast radius | `hermitclaw/pysandbox.py:setup` | Only selected functions patched; `os.open/os.read` and related paths can escape env root | Agent runs `python -c` to read host files/secrets outside box | Move to OS/container sandbox (nsjail/firejail/container), not monkey-patching | Integration test attempts `os.open('/etc/hosts')` and must fail |
| HC-003 | Unauthenticated API + wildcard CORS enables remote control/data exposure | HIGH | Very low complexity / deterministic | `hermitclaw/server.py` API and middleware setup | No authn/authz on control/read endpoints; CORS `*` broadens abuse | Any network-reachable party drives agent, reads files/events/raw logs | Add auth token/session, restrict bind/CORS, RBAC per crab | HTTP tests: unauthenticated requests denied |
| HC-004 | SSRF and metadata probing via `fetch_url` | HIGH | Low complexity / environment-dependent impact | `hermitclaw/tools.py:fetch_url` | Allows arbitrary http/https URLs including internal metadata/admin endpoints | Agent or injected prompt requests cloud metadata/internal services | Add URL allowlist/deny private CIDRs, metadata hard-block, egress policy | Tests for blocked 169.254.169.254, localhost, RFC1918 |
| HC-005 | Prompt/tool-output injection can steer high-privilege actions | HIGH | Low complexity / moderate reliability | `hermitclaw/brain.py` tool loop + prompt assembly | Untrusted tool/user/file content is fed back with no safety classifier/policy gate | Malicious webpage/file instructs model to run shell actions, agent complies | Introduce policy engine + action approval/filters per capability | Adversarial corpus tests for tool-output injection |
| HC-006 | Sensitive model I/O persisted to log file | MEDIUM | Low complexity / high detectability | `hermitclaw/brain.py:_emit_api_call` | Inputs/outputs logged to project root; may include secrets and personal data | Local attacker/process reads logs for credentials/prompts/tool output | Redact secrets/PII, minimize logs, rotate and permissions hardening | Tests ensure key patterns are masked |
| HC-007 | File-read endpoint can expose all box contents to any caller | MEDIUM | Low complexity | `hermitclaw/server.py:get_file,get_files` | API provides directory listing and file reads without auth | Remote user enumerates research/memory/user dropped files | Require auth and per-crab ownership checks | API tests for unauthorized 401/403 |
| HC-008 | Supply-chain exposure via agent-driven package installs | MEDIUM | Moderate complexity / variable reliability | `prompts.py` + `tools.py` pip rewrite | Prompt encourages arbitrary `pip install`; no package policy/pinning | Agent installs typosquatted package -> arbitrary code in sandbox context | Disable package installs by default or policy-gate with allowlist | Test denies non-allowlisted package install |

---

## 4) Deep Dives

### HC-001: Shell blocklist bypass -> command execution/exfil (CRITICAL)

**Evidence:** command filtering is prefix-based and shell invocation uses `shell=True`. This combination permits separator/operator bypass and alternate command forms.

**Step chain:**
1. Malicious prompt/tool-output convinces model to run a shell command that starts with allowed token.
2. Command appends forbidden action using `;`, `&&`, pipes, or wrapper binaries (`command`, `env`, variable assignments).
3. Backend executes complete string via shell interpreter.
4. Agent accesses host data/network not intended by policy.

**Why this matters:** this is a direct policy bypass in the primary execution substrate.

### HC-002: Monkey-patch sandbox bypass via unpatched APIs (CRITICAL)

**Evidence:** sandbox patches `builtins.open` and some `os.*` path functions, but does not comprehensively mediate syscalls (e.g., fd-level file operations).

**Step chain:**
1. Agent executes Python via `python -c` rewritten to sandbox wrapper.
2. Script uses low-level file APIs not wrapped by `_check_path`.
3. Reads host files (keys/config/history), then writes output back into box or logs for retrieval.

**Why this matters:** Python sandbox is treated as control boundary but is bypassable with standard library primitives.

### HC-003 + HC-005 chain: Remote unauth + injection -> delegated compromise

1. External actor reaches exposed server (0.0.0.0).
2. Sends crafted `/api/message` content to shape agent goals.
3. Agent uses tool chain to execute risky commands or internal URL fetches.
4. Data returned through `/api/events`, `/api/raw`, or generated files.

**Impact:** one weak control plane + one model trust issue yields practical hostile automation.

### Environment-dependent escalations

- **CI runner**: fetched secrets from env/build cache + egress via web tools.
- **Kubernetes**: SSRF to kubelet/internal services; if SA token mounted, agent can read and pivot.
- **Cloud VM/serverless**: metadata endpoint probing for temporary credentials.
- **Airgapped**: exfil through logs/artifacts or covert timing/content channels.

---

## 5) ULTRAZHINK Boundary-Breaker Playbook

Each item = preconditions -> chain -> impact -> hardening.

1. **Separator laundering**: allowed shell prefix + `;` chain -> blocked command executes -> host compromise -> parse into argv and disallow shell metacharacters.
2. **Wrapper command bypass**: `command bash`/`env sh` wrappers -> prefix filter evaded -> arbitrary script execution -> canonicalize first executable token.
3. **Quoted traversal bypass**: quoted `../` token slips traversal check -> access outside cwd in shell utilities -> host file read -> robust parser + path canonicalization post-parse.
4. **Absolute-path quoting bypass**: quoted `/etc/...` evades regex -> sensitive file reads -> secrets exposure -> enforce resolved-path policy on every file arg.
5. **Tool-output injection**: webpage says “run command X now” -> agent treats as instruction -> command execution -> add tool-output tainting and instruction sandboxing.
6. **Respond-tool capability laundering** (confused deputy): attacker messages social-engineered instructions -> model performs privileged actions -> delegated abuse -> add safety policy before high-risk tools.
7. **Fetch->Shell laundering** (confused deputy): fetched content encodes command recipe -> agent executes shell -> remote code/data impact -> require explicit provenance + deny executable transformations.
8. **Planning-file poisoning** (confused deputy): malicious file influences `projects.md` plan -> persistent malicious goals -> long-lived compromise -> isolate user input from durable strategic memory.
9. **API raw-history exfil channel**: attacker reads `/api/raw` -> captures prompts/tool outputs -> secret leakage -> authenticate + redact + disable in prod.
10. **Event-stream covert exfil** (side channel): encode secrets in thought cadence/message lengths -> observer extracts over WS -> data leak even with egress blocked -> rate/format normalization and anomaly detection.
11. **Log-based exfil** (side channel): secrets copied into logged tool outputs -> harvested from log shipping -> indirect exfil -> redact and segregate sensitive logs.
12. **Filename covert channel** (side channel): agent writes many files with encoded bits in names -> sync/backup exports data -> covert leak -> enforce filename policy and monitor unusual patterns.
13. **Timing channel via API polling** (side channel): secret bitstream encoded by response delay patterns -> remote extractor recovers data -> random jitter + bounded deterministic scheduling.
14. **Resource exhaustion channel** (side/DoS): crafted input causes giant tool outputs -> truncation artifacts leak structured secrets + service degradation -> strict output quotas and per-tool caps.
15. **Dependency preinstall trap** (supply chain): malicious package via `pip install` in prompt -> code executes during install/import -> compromise -> lock/allowlist, no dynamic installs.
16. **Transitive dependency drift** (supply chain): unpinned dependency update introduces backdoor -> latent compromise -> lockfile verification, SBOM, integrity checks.
17. **Build-step abuse** (supply chain): compromised frontend package script in npm install/build -> CI secret theft -> isolate CI tokens, `npm ci --ignore-scripts` where possible.
18. **Provider adapter trust abuse** (supply chain/confused deputy): compromised model provider returns crafted tool calls -> privileged actions run -> strong tool-policy validation independent of model.

Coverage counts: capability laundering/confused-deputy (>=5): #6, #7, #8, #18, plus #5 as implicit deputy abuse. Side/covert channel (>=5): #10–#14. Supply-chain/build/plugin (>=5): #15–#18 plus #17 and #16.

---

## 6) Hardening Plan

### Now (0-7 days)

- Replace shell execution with structured command execution (`shell=False`) and explicit allowlisted commands/args.
- Disable or heavily gate `pip install` and dynamic code execution in default profile.
- Add mandatory API auth (token/session), disable wildcard CORS in non-dev, bind localhost by default.
- Block SSRF targets: localhost, link-local, RFC1918, metadata IPs/domains.
- Turn off `/api/raw` in production builds; redact logs.

### Next (1-4 weeks)

- Replace monkey-patch sandbox with real isolation boundary (container/VM/nsjail) with:
  - read-only root FS, writable scratch only,
  - no host mounts except controlled workdir,
  - seccomp/AppArmor, no-new-privileges,
  - CPU/memory/pid/file descriptor limits,
  - controlled DNS/egress.
- Add policy engine for tool calls:
  - deny-by-default high-risk tools,
  - JSON schema validation and argument normalization,
  - risk scoring + optional human approval for risky actions.

### Later (1-2 months)

- Multi-tenant security model for multiple crabs/users.
- Secret broker pattern (short-lived scoped tokens, no raw env secret exposure).
- Security telemetry: abnormal tool-call sequences, command entropy spikes, injection signatures.

### Concrete guardrails

- **Allowlists**: executable names, URL domains, file extensions/paths.
- **Validation**: strict typed schema for each tool and URL parser with DNS/IP revalidation.
- **Timeouts/caps**: per-tool CPU time, stdout bytes, file size, loop iterations.
- **Memory controls**: isolate trust levels (user/tool/system) and avoid direct instruction carryover from untrusted sources.

### Secrets handling strategy

- Remove long-lived secrets from process env where possible.
- Inject scoped runtime credentials only to components requiring them.
- Redact secrets from logs/events/model context.
- Rotate keys and provide kill-switch for suspected compromise.

### Operational controls

- Run as non-root dedicated user.
- Container flags: `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`, drop all capabilities, seccomp `RuntimeDefault`, AppArmor/SELinux profile.
- NetworkPolicy egress allowlist; explicit deny metadata endpoints.

---

## 7) Regression Tests & Security Checks

### Unit/integration tests to add

1. **Shell policy bypass suite**: separators, wrappers, quoting, variable-prefix, multiline commands.
2. **Sandbox escape suite**: fd-level I/O, symlink tricks, temp file races, import bypass attempts.
3. **SSRF suite**: private/link-local/metadata/localhost URLs and DNS rebinding cases.
4. **AuthZ suite**: all `/api/*` and `/ws*` endpoints require auth; per-crab access control.
5. **Prompt/tool-output injection suite**: adversarial corpora asserting “never escalate privileges from untrusted content.”
6. **Log redaction suite**: API keys/tokens/password patterns are masked before persistence.

### Static checks (enforcement targets)

- Ban `subprocess.*(shell=True)` except approved wrappers.
- Flag dynamic exec/eval and unsafe deserialization.
- Detect open CORS in production config.
- Detect unauthenticated route registration for privileged endpoints.
- Enforce dependency pinning and lockfile integrity checks in CI.

### Agent-harness abuse tests

- Multi-step chain tests (prompt injection -> tool output -> privileged action).
- Capability laundering tests across tool combinations.
- Covert-channel detection checks (timing/length modulation anomalies).
- Chaos tests with compromised tool/provider responses.

---

## Missing artifacts that would increase confidence

- Deployment manifests (Dockerfile, Kubernetes/Helm, CI workflows).
- Runtime auth model and reverse-proxy/network topology.
- Production logging pipeline and retention policies.
- Dependency lock/security scan outputs for both Python and frontend stacks.

## Subsequent phases (recommended execution)

1. **Phase 1 — Immediate containment:** close unauth control plane + shell/SSRF critical paths.
2. **Phase 2 — Isolation hardening:** migrate execution to real sandbox boundary with constrained egress.
3. **Phase 3 — Policy & trust architecture:** taint-aware prompt/tool pipeline and high-risk action policy gate.
4. **Phase 4 — Verification program:** adversarial regression suite, SAST/DAST, dependency and runtime posture checks.
5. **Phase 5 — Operationalization:** incident playbooks, security telemetry, key rotation drills.

## Best reporting format going forward

Use a two-layer report:

- **Layer A (Exec dashboard, 1 page):** risk heatmap, top 5 blockers, remediation timeline, owner/status.
- **Layer B (Engineer deep pack):** findings table + deep dives + reproducible tests + patch references.

Cadence:

- Weekly delta report (new findings, fixed, reopened).
- Milestone gate report before each release (must-pass critical controls checklist).
