# optimize_synthesizer.py
import dspy
from agent.dspy_signatures import Synthesizer

# Reuse same LM
lm = dspy.LM('ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434')
dspy.configure(lm=lm)

example = dspy.Example(
    question="Return days for unopened Beverages",
    format_hint="int",
    sql_result="",
    retrieved_docs="product_policy.md::chunk0: Beverages unopened: 14 days",
    used_tables=[],
    final_answer="14",
    explanation="From product policy",
    citations=["product_policy.md::chunk0"]
).with_inputs("question", "format_hint", "sql_result", "retrieved_docs", "used_tables")

compiled = dspy.ChainOfThought(Synthesizer).compile(trainset=[example])
compiled.save("synth_optimized.json")
print("DSPy optimization done!")