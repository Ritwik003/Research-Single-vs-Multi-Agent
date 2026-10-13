# Multi-Agent vs Single-Agent at Equal Compute

A fair, controlled comparison. Not a new method — a clean experiment that
separates "the multi-agent structure helps" from "it just spent more compute."

## The three systems
| System | What it does | Why it's here |
|---|---|---|
| **single** | one model call | the plain baseline |
| **single_vote** | one model, answered 5x, majority vote | a single agent spending EXTRA compute WITHOUT structure |
| **multi** | generator → critic → revise | the multi-agent structure |

The one comparison that matters: **multi vs single_vote**. If multi wins at
similar token cost, the *structure* is doing real work. If not, the classic
"multi-agent is better" claim was really just "it used more compute."

## How to run
1. Make sure Ollama is running with Mistral:
   ```
   ollama run mistral
   ```
2. Run the experiment (produces `results.csv` + prints the summary):
   ```
   python research.py
   ```
3. Make the headline chart:
   ```
   python plot.py      # produces result_chart.png
   ```

## What you get
- `results.csv` — every run: system, predicted answer, correct?, tokens, seconds.
- Printed summary — accuracy + avg tokens per system, and the two conclusions
  (based on compute cost / based on structure).
- `result_chart.png` — accuracy vs tokens. Up-and-left = better.

## The two research points (your notes)
1. **Based on compute cost** — read the chart vertically. At the same tokens,
   whose accuracy is higher?
2. **Based on structure** — compare `multi` vs `single_vote`. Similar cost;
   different design. The gap (if any) is what the structure buys.

## To make it a stronger result (next steps)
- Swap the 10 sample problems for the real **GSM8K** dataset (hundreds of
  problems → real statistics). Just replace the `PROBLEMS` list.
- Add a second task type (e.g. simple coding problems graded by running the
  code) to test whether the answer depends on task type.
- Do the **error analysis**: open `results.csv`, find problems where `multi`
  is right but `single` is wrong, and read the `critique` — that shows WHAT
  kind of error the critic caught. Also find where the critic made a right
  answer wrong (the cost side).

## How to present it (suggested slide order)
1. The problem: multi-agent "wins" but uses more compute — is that fair?
2. The three systems (the table above).
3. The method: same problems, count tokens, grade automatically.
4. **The chart** (your headline).
5. Conclusion on compute cost.
6. Conclusion on structure (multi vs single_vote).
7. Error analysis: what the critic actually catches.
8. Limitations + next steps.
