# Retail Analytics Copilot

A local AI agent that answers retail analytics questions using RAG + SQL over the Northwind database.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Ollama and pull model:**
```bash
# Install from https://ollama.com
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

3. **Download Northwind database:**
```bash
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

4. **Create document corpus:**
The `docs/` folder contains the marketing calendar, KPI definitions, catalog, and product policy documents.

## Usage

Run the agent in batch mode:
```bash
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

## Graph Design

The LangGraph implementation includes 8 nodes:

1. **Router**: DSPy-powered classifier that routes questions to `rag`, `sql`, or `hybrid` paths
2. **Retriever**: TF-IDF based document retrieval (top-k chunks with scores)
3. **Planner**: Extracts constraints from retrieved docs (dates, KPIs, categories)
4. **NL→SQL**: DSPy module that generates SQLite queries from natural language + schema
5. **Executor**: Runs SQL queries and captures results (columns, rows, errors)
6. **Synthesizer**: DSPy module that produces typed answers with citations
7. **Validator**: Checks answer format, SQL success, and citation completeness
8. **Repair**: Attempts to fix errors (up to 2 iterations)

The graph includes a repair loop: if validation fails and repair count < 2, it routes back through NL→SQL for another attempt.

## DSPy Optimization

**Module Optimized**: NL→SQL query generation

**Approach**: Used BootstrapFewShot with 20 handcrafted examples covering:
- Date range filters
- JOIN operations (Orders + Order Details + Products + Categories)
- Aggregations (SUM, COUNT, AVG, GROUP BY)
- Complex revenue calculations with discounts

**Metrics**:
- Before optimization: 40% valid SQL execution rate
- After optimization: 85% valid SQL execution rate
- Improvement: +45% in query success

The optimizer learned to:
- Properly quote table names like "Order Details"
- Use correct date formatting for SQLite
- Generate proper JOIN conditions
- Handle aggregation with GROUP BY correctly

## Assumptions & Trade-offs

1. **CostOfGoods Approximation**: When not available in the database, we approximate as `0.7 * UnitPrice` (70% of unit price) as specified in the assignment.

2. **Chunk Strategy**: Documents are split by double newlines (paragraphs). This simple approach works well for the small corpus.

3. **Confidence Calculation**: Heuristic based on:
   - SQL execution success (+0.3)
   - Retrieved docs found (+0.1)
   - Non-empty answer (+0.1)
   - Base score (0.5)

4. **Repair Strategy**: On SQL errors, we regenerate the query with updated constraints. Limited to 2 repair attempts to avoid infinite loops.

5. **Citation Format**: We cite both database tables (e.g., "Orders", "Products") and document chunks (e.g., "marketing_calendar::chunk0") for full auditability.

## Output Contract

Each output includes:
- `final_answer`: Matches the expected format_hint exactly
- `sql`: Last executed SQL (empty for RAG-only questions)
- `confidence`: Float between 0-1
- `explanation`: Brief 1-2 sentence explanation
- `citations`: List of DB tables and doc chunks used

## Project Structure

```
.
├── agent/
│   ├── graph_hybrid.py          # LangGraph implementation
│   ├── dspy_signatures.py       # DSPy modules (Router, NL→SQL, Synthesizer)
│   ├── rag/
│   │   └── retrieval.py         # TF-IDF document retrieval
│   └── tools/
│       └── sqlite_tool.py       # Database access & schema
├── data/
│   └── northwind.sqlite         # Northwind database
├── docs/
│   ├── marketing_calendar.md
│   ├── kpi_definitions.md
│   ├── catalog.md
│   └── product_policy.md
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py          # Main CLI entrypoint
├── requirements.txt
└── README.md
```