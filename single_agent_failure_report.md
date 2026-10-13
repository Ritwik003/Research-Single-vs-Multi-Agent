# Why Single Agents Fail or Give Inaccurate Answers

*A research synthesis of the Five-Mode Failure Taxonomy*

## Research question

Why can a single-agent LLM system fail to perform, or return confident but inaccurate answers, at production scale — and does better modelling fix it?

**Verdict:** Yes, the source article answers this directly. Its central finding is that single-agent failures are **structural** (problems of orchestration and state), **not model-quality problems**. A stronger model can raise the *level* of a performance curve, but it never changes the *slope* — the failure still grows with scale.

## Core thesis

A single agent conflates four things that production forces apart, all inside one context window and one tool surface:

- conversational state
- world state (the changing data it acts on)
- tool-selection logic
- control flow

When any one of these grows, it crowds out the others. That coupling is the root cause, and it shows up as five measurable failure modes. You cannot prompt your way out of it, and you cannot pay your way out with a frontier model.

## The Five-Mode Failure Taxonomy

| Failure mode | What happens | Why a better model does not fix it |
|---|---|---|
| **Tool explosion** | Tool-selection accuracy collapses as the tool count grows | More tools = more plausible wrong choices, regardless of model |
| **State drift** | Agent acts on a world that changed mid-run | The model cannot know about a mutation it never observed |
| **Context window collapse** | Early instructions decay out of effective attention | A bigger window delays the collapse; it does not prevent it |
| **Latency cascade** | Sequential tool hops compound into long-tail latency | Faster tokens do not remove the round-trip count |
| **Cost runaway** | Token spend grows super-linearly with retries and context | A pricier model makes the same runaway more expensive |

### 1. Tool explosion
As the tool set grows (3 → 8 → 16 → 30), selection accuracy drops sharply (≈98% → 61% in the article's composite figures) because overlapping tool descriptions compete for the same intent. The model is not getting dumber; the decision is getting harder faster than linearly.
**Fix:** retrieval-based tool filtering, meta-tools, or splitting the tool surface across specialised agents — not just rewriting docstrings.

### 2. State drift
A second actor (human, job, or another service) changes a record between the agent's *read* and its *write*, so the agent acts on stale data — e.g. issuing a duplicate refund. The prompt rule "never refund twice" was never the problem.
**Fix:** read-before-write guards and optimistic concurrency *inside the tool*, made atomic (transaction or version check) — not a stronger prompt.

### 3. Context window collapse
This is the silent failure with no error. As the conversation grows, early instructions (policy, tone, compliance) decay out of the part of the context the model attends to. The agent stops following the rule intermittently — precisely on the long tickets where it matters most. This is the "lost in the middle" / U-shaped attention effect.
**Fix:** give each concern its own bounded context; do not rely on a larger window.

### 4. Latency cascade
Agent-tool round trips are sequential by construction — the model can't pick hop 4 until hop 3 returns. A five-hop ticket pays five round trips end to end, and system p95 is dominated by the longest chains, not the average hop.
**Fix:** budget latency per hop *and* per chain; parallelise independent subtasks (impossible in one sequential loop).

### 5. Cost runaway
The last alarm to fire and the most expensive. Two forces multiply token spend super-linearly: (a) the full growing conversation is re-sent on every hop, so a 10-hop ticket pays for early context ten times over; (b) naive retries re-send that swelling context each attempt.
**Fix:** bound context growth, cap hops, and deduplicate re-sent context (delta checkpointing where available).

## The Single-Agent Ceiling

The point where the marginal failure rate from adding capability (one more tool, more context, another step) exceeds the marginal competence it buys. Below the crossover, additions help; above it, they hurt. The ceiling is **real but not fixed** — a better model moves the crossover higher, which is why "upgrade the model" works for a while and then stops.

## Supporting evidence

- **Liu et al. (2024)** — "lost in the middle": retrieval accuracy drops >30% when key information sits mid-context; persists in frontier models.
- **Gao et al. (2025)** — the multi-agent advantage *shrinks* as base models improve at long-context reasoning; a caution against multi-agent-too-early.
- **Cemri et al. (MAST, NeurIPS 2025)** — 14 multi-agent failure modes from 1,600+ traces; concludes failures stem from *system design*, not model limits — the multi-agent echo of this article's argument.
- **Chroma (2025)** — context degradation confirmed in frontier models.
- **Anthropic** — "context rot" (finite attention budget per token); its production multi-agent system uses ~15× the tokens of a single chat, so multi-agent is *not* a way to make tokens cheaper.

## Diagnostic decision rule

Map the symptom to the mode before choosing a fix:

- **Tool explosion, context collapse, or latency cascade** *and* you have already tuned the single agent → you have hit the ceiling; **multi-agent topology is the structural answer.**
- **State drift or cost runaway** → fix in the **tools and state layer first**; these have single-agent fixes, and adding agents without fixing them just multiplies the bug.
- **Not yet tuned the single agent** → you don't know your ceiling. Tighten prompts and tools on a frontier model first. Multi-agent too early is the most common and most expensive mistake.

## Conclusion for the research

The article fully supports the research topic: single agents fail or give inaccurate answers because a single loop must hold too much context, choose among too many tools, and track a world that keeps changing — structural limits, not intelligence limits. A better model shifts a curve's level but never bends its slope.

The correct research stance is therefore *diagnostic before prescriptive*: name the failure mode, measure it, and only then decide whether the answer is a better-scoped single agent or a multi-agent architecture. Multi-agent is one way past the ceiling — but it trades token cost for capability and parallelism, and is not always justified.
