"""
=============================================================================
 RESEARCH: Multi-Agent vs Single-Agent LLM Systems at Equal Compute
=============================================================================

Goal
----
Multi-agent systems (generator + critic) are said to beat a single agent.
But they also use MORE compute (more model calls => more tokens). So is the
gain from the STRUCTURE, or just from spending more compute?

This script compares THREE systems on the SAME problems and, crucially,
records the TOKENS each one spends. That lets us compare them FAIRLY
(at equal compute) instead of unfairly (multi-agent secretly spending more).

The three systems
-----------------
  1. SINGLE          : one model call. The plain baseline.
  2. SINGLE_VOTE     : one model, answered N times, majority vote.
                       (A single agent spending EXTRA compute on its own,
                        WITHOUT multi-agent structure.)
  3. MULTI           : generator -> critic -> generator revises.
                       (The multi-agent structure.)

The key scientific comparison:
  - MULTI vs SINGLE        -> does multi-agent help at all?
  - MULTI vs SINGLE_VOTE   -> at similar token cost, does the STRUCTURE
                              beat just "spend more compute"? This is the
                              question that actually matters.

Model: local Mistral via Ollama (same as the existing project).
=============================================================================
"""

import re
import csv
import time
import requests
from collections import Counter

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
N_VOTES = 5          # how many samples SINGLE_VOTE draws
REQUEST_TIMEOUT = 180


# -----------------------------------------------------------------------------
# LLM CALL  (same pattern as your project, but now it RETURNS TOKEN COUNTS)
# -----------------------------------------------------------------------------
def call_llm(prompt, temperature=0.7):
    """
    Calls Mistral via Ollama and returns (text, tokens_used).

    Ollama reports token counts in its response:
      prompt_eval_count = input tokens
      eval_count        = output tokens
    We sum them: that is the COMPUTE this call spent.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()
        text = data.get("response", "")
        in_tok = data.get("prompt_eval_count", 0)
        out_tok = data.get("eval_count", 0)
        return text, (in_tok + out_tok)
    except Exception as e:
        return f"LLM_ERROR: {e}", 0


# -----------------------------------------------------------------------------
# ANSWER EXTRACTION + GRADING
# -----------------------------------------------------------------------------
def extract_number(text):
    """
    Pull the final numeric answer out of the model's text.
    We prefer a line like 'Answer: 224' or '#### 224', else last number seen.
    """
    # look for an explicit answer marker first
    m = re.search(r"(?:answer|ans|result)\s*[:=]?\s*\$?(-?\d[\d,]*\.?\d*)",
                  text, re.IGNORECASE)
    if m:
        return _clean_num(m.group(1))
    # otherwise take the last number in the text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return _clean_num(nums[-1])
    return None


def _clean_num(s):
    s = s.replace(",", "")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def is_correct(predicted, gold):
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


# -----------------------------------------------------------------------------
# SYSTEM 1: SINGLE AGENT  (one call)
# -----------------------------------------------------------------------------
def system_single(question):
    prompt = f"""Solve this math problem step by step.
End with a line exactly like: Answer: <number>

Problem: {question}
"""
    text, tokens = call_llm(prompt, temperature=0.0)
    return {
        "answer": extract_number(text),
        "tokens": tokens,
        "raw": text,
    }


# -----------------------------------------------------------------------------
# SYSTEM 2: SINGLE + SELF-CONSISTENCY  (answer N times, majority vote)
# -----------------------------------------------------------------------------
def system_single_vote(question, n=N_VOTES):
    prompt = f"""Solve this math problem step by step.
End with a line exactly like: Answer: <number>

Problem: {question}
"""
    answers = []
    total_tokens = 0
    for _ in range(n):
        text, tokens = call_llm(prompt, temperature=0.8)  # variety needed
        total_tokens += tokens
        ans = extract_number(text)
        if ans is not None:
            answers.append(ans)

    if not answers:
        return {"answer": None, "tokens": total_tokens, "raw": "no answers"}

    # majority vote
    winner = Counter(answers).most_common(1)[0][0]
    return {
        "answer": winner,
        "tokens": total_tokens,
        "raw": f"votes={answers} -> {winner}",
    }


# -----------------------------------------------------------------------------
# SYSTEM 3: MULTI-AGENT  (generator -> critic -> revise)
# -----------------------------------------------------------------------------
def system_multi(question):
    total_tokens = 0

    # --- Agent 1: GENERATOR produces a first solution ---
    gen_prompt = f"""You are a careful math solver.
Solve this problem step by step.
End with a line exactly like: Answer: <number>

Problem: {question}
"""
    gen_text, t1 = call_llm(gen_prompt, temperature=0.3)
    total_tokens += t1

    # --- Agent 2: CRITIC checks the solution for errors ---
    critic_prompt = f"""You are a strict math checker.
Below is a problem and a proposed solution.
Find any arithmetic or reasoning mistakes. Be specific.
If the solution is fully correct, say "NO ERRORS".

Problem: {question}

Proposed solution:
{gen_text}

Your critique:
"""
    critic_text, t2 = call_llm(critic_prompt, temperature=0.0)
    total_tokens += t2

    # --- Agent 1 again: REVISE using the critique ---
    revise_prompt = f"""You are a careful math solver.
Here is the problem, your first attempt, and a checker's critique.
Produce a corrected final solution.
End with a line exactly like: Answer: <number>

Problem: {question}

First attempt:
{gen_text}

Checker's critique:
{critic_text}

Corrected solution:
"""
    final_text, t3 = call_llm(revise_prompt, temperature=0.0)
    total_tokens += t3

    return {
        "answer": extract_number(final_text),
        "tokens": total_tokens,
        "raw": final_text,
        "critique": critic_text,
    }


# -----------------------------------------------------------------------------
# THE EVALUATION HARNESS  (runs all systems, records everything)
# -----------------------------------------------------------------------------
SYSTEMS = {
    "single": system_single,
    "single_vote": system_single_vote,
    "multi": system_multi,
}


def run_experiment(problems, out_csv="results.csv"):
    rows = []
    print(f"\nRunning {len(problems)} problems x {len(SYSTEMS)} systems...\n")

    for i, prob in enumerate(problems, 1):
        q = prob["question"]
        gold = prob["answer"]
        print(f"[{i}/{len(problems)}] {q[:60]}...")

        for sys_name, sys_fn in SYSTEMS.items():
            t0 = time.time()
            result = sys_fn(q)
            elapsed = round(time.time() - t0, 1)

            correct = is_correct(result["answer"], gold)
            rows.append({
                "problem_id": i,
                "system": sys_name,
                "gold": gold,
                "predicted": result["answer"],
                "correct": int(correct),
                "tokens": result["tokens"],
                "seconds": elapsed,
            })
            mark = "OK " if correct else "XX "
            print(f"    {mark} {sys_name:12s} "
                  f"pred={result['answer']} tokens={result['tokens']}")

    # write raw results
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw results saved to {out_csv}")
    return rows


# -----------------------------------------------------------------------------
# ANALYSIS  (the two research points: compute cost + structure)
# -----------------------------------------------------------------------------
def analyze(rows):
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)

    systems = ["single", "single_vote", "multi"]
    summary = {}
    for s in systems:
        srows = [r for r in rows if r["system"] == s]
        n = len(srows)
        acc = sum(r["correct"] for r in srows) / n * 100
        avg_tok = sum(r["tokens"] for r in srows) / n
        summary[s] = {"accuracy": acc, "avg_tokens": avg_tok}
        print(f"\n {s.upper()}")
        print(f"   Accuracy    : {acc:.1f}%")
        print(f"   Avg tokens  : {avg_tok:.0f}")

    print("\n" + "=" * 60)
    print(" INTERPRETATION")
    print("=" * 60)

    single = summary["single"]
    vote = summary["single_vote"]
    multi = summary["multi"]

    print(f"\n 1) BASED ON COMPUTE COST:")
    print(f"    single      -> {single['accuracy']:.0f}% at {single['avg_tokens']:.0f} tok")
    print(f"    single_vote -> {vote['accuracy']:.0f}% at {vote['avg_tokens']:.0f} tok")
    print(f"    multi       -> {multi['accuracy']:.0f}% at {multi['avg_tokens']:.0f} tok")

    print(f"\n 2) BASED ON STRUCTURE (multi vs single_vote at similar cost):")
    diff = multi["accuracy"] - vote["accuracy"]
    if diff > 2:
        print(f"    Multi beats single_vote by {diff:.1f} points at similar")
        print(f"    compute => the STRUCTURE is doing real work.")
    elif diff < -2:
        print(f"    single_vote beats multi by {-diff:.1f} points => the")
        print(f"    multi-agent structure is NOT worth its cost here;")
        print(f"    plain extra compute is better spent on sampling.")
    else:
        print(f"    They are within {abs(diff):.1f} points => at equal compute")
        print(f"    the structure gives NO clear advantage. The apparent")
        print(f"    'multi-agent win' was mostly just extra compute.")

    return summary


# -----------------------------------------------------------------------------
# SAMPLE PROBLEM SET  (swap in GSM8K later; these are checkable word problems)
# -----------------------------------------------------------------------------
PROBLEMS = [
    {"question": "A store sells notebooks at 40 rupees each. Buying 5 or more "
                 "gives 20% off the whole purchase. Ravi buys 7 notebooks. "
                 "How much does he pay?", "answer": 224},
    {"question": "A train travels 60 km in the first hour and 80 km in the "
                 "second hour. What is its average speed in km/h?", "answer": 70},
    {"question": "Sara has 3 boxes with 12 pencils each. She gives away 10 "
                 "pencils. How many pencils are left?", "answer": 26},
    {"question": "A shirt costs 500 rupees. It is discounted 10%, then another "
                 "5% is taken off the discounted price. What is the final "
                 "price?", "answer": 427.5},
    {"question": "There are 24 students. One third play football and the rest "
                 "play cricket. How many play cricket?", "answer": 16},
    {"question": "A tank holds 200 litres. It leaks 8 litres per hour. After "
                 "how many hours is it empty?", "answer": 25},
    {"question": "A book has 300 pages. Meena reads 25 pages a day. How many "
                 "days to finish?", "answer": 12},
    {"question": "5 workers build a wall in 12 days. How many days would 10 "
                 "workers take at the same rate?", "answer": 6},
    {"question": "A rectangle is 8 cm long and 5 cm wide. What is its area in "
                 "square cm?", "answer": 40},
    {"question": "Ram scored 45, 60 and 75 in three tests. What is his average "
                 "score?", "answer": 60},
]


if __name__ == "__main__":
    rows = run_experiment(PROBLEMS)
    analyze(rows)
    print("\nDone. Open results.csv to see every run.")
