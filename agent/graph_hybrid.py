"""LangGraph implementation with ≥6 nodes and repair loop."""
from typing import TypedDict, Literal, Optional, Any
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import route_question, generate_sql, synthesize_answer
from agent.rag.retrieval import retrieve_docs
from agent.tools.sqlite_tool import execute_query, get_schema
import json

class GraphState(TypedDict):
    """State that flows through the graph."""
    question: str
    question_id: str
    format_hint: str
    route: Optional[str]
    retrieved_docs: list
    constraints: dict
    sql: str
    sql_result: dict
    final_answer: Any
    confidence: float
    explanation: str
    citations: list
    errors: list
    repair_count: int
    trace: list

def router_node(state: GraphState) -> GraphState:
    """Node 1: Route the question to rag, sql, or hybrid."""
    print("  [Router] Analyzing question...")
    state['trace'].append({'node': 'router', 'question': state['question']})
    
    # Use DSPy to classify (with fallback)
    try:
        route = route_question(state['question'])
        print(f"  [Router] Route: {route}")
    except Exception as e:
        print(f"  [Router] Error: {e}, using fallback")
        route = 'hybrid'
    
    state['route'] = route
    state['trace'].append({'node': 'router', 'route': route})
    
    return state

def retriever_node(state: GraphState) -> GraphState:
    """Node 2: Retrieve relevant document chunks."""
    print("  [Retriever] Searching documents...")
    state['trace'].append({'node': 'retriever', 'start': True})
    
    docs = retrieve_docs(state['question'], top_k=3)
    state['retrieved_docs'] = docs
    print(f"  [Retriever] Found {len(docs)} relevant documents")
    state['trace'].append({'node': 'retriever', 'docs_found': len(docs)})
    
    return state

def planner_node(state: GraphState) -> GraphState:
    """Node 3: Extract constraints from docs (dates, KPIs, categories)."""
    print("  [Planner] Extracting constraints...")
    state['trace'].append({'node': 'planner', 'start': True})
    
    constraints = {}
    
    for doc in state['retrieved_docs']:
        content = doc['content'].lower()
        
        # Extract date ranges
        if 'summer beverages 1997' in content:
            constraints['date_start'] = '1997-06-01'
            constraints['date_end'] = '1997-06-30'
            constraints['campaign'] = 'Summer Beverages 1997'
        elif 'winter classics 1997' in content:
            constraints['date_start'] = '1997-12-01'
            constraints['date_end'] = '1997-12-31'
            constraints['campaign'] = 'Winter Classics 1997'
        
        # Extract KPI formulas
        if 'aov' in content or 'average order value' in content:
            constraints['kpi'] = 'AOV'
            constraints['formula'] = 'SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)'
        if 'gross margin' in content or 'gm' in content:
            constraints['kpi'] = 'Gross Margin'
            constraints['cost_approximation'] = '0.7 * UnitPrice'
        
        if 'beverages' in state['question'].lower():
            constraints['category'] = 'Beverages'
        
        # Extract return policy
        if 'return' in content and 'days' in content:
            if 'beverages unopened: 14 days' in content:
                constraints['return_days_beverages'] = 14
    
    state['constraints'] = constraints
    print(f"  [Planner] Extracted constraints: {list(constraints.keys())}")
    state['trace'].append({'node': 'planner', 'constraints': constraints})
    
    return state

def nl_to_sql_node(state: GraphState) -> GraphState:
    """Node 4: Generate SQL using DSPy."""
    print("  [NL→SQL] Generating SQL query...")
    state['trace'].append({'node': 'nl_to_sql', 'start': True})
    
    schema = get_schema()
    
    try:
        sql = generate_sql(
            question=state['question'],
            schema=schema,
            constraints=state['constraints']
        )
        if sql:
            print(f"  [NL→SQL] Generated SQL ({len(sql)} chars)")
        else:
            print(f"  [NL→SQL] No SQL needed (RAG-only question)")
    except Exception as e:
        print(f"  [NL→SQL] Error: {e}")
        sql = ""
    
    state['sql'] = sql
    state['trace'].append({'node': 'nl_to_sql', 'sql_length': len(sql)})
    
    return state

def executor_node(state: GraphState) -> GraphState:
    """Node 5: Execute SQL and capture results."""
    print("  [Executor] Running SQL query...")
    state['trace'].append({'node': 'executor', 'start': True})
    
    if not state['sql']:
        state['sql_result'] = {'columns': [], 'rows': [], 'error': None}
        print("  [Executor] No SQL to execute")
        return state
    
    result = execute_query(state['sql'])
    state['sql_result'] = result
    
    if result.get('error'):
        state['errors'].append(f"SQL Error: {result['error']}")
        print(f"  [Executor] ✗ Error: {result['error']}")
        state['trace'].append({'node': 'executor', 'error': result['error']})
    else:
        print(f"  [Executor] ✓ Success: {len(result.get('rows', []))} rows")
        state['trace'].append({'node': 'executor', 'rows': len(result.get('rows', []))})
    
    return state

def synthesizer_node(state: GraphState) -> GraphState:
    """Node 6: Synthesize final answer with DSPy."""
    print("  [Synthesizer] Creating answer...")
    state['trace'].append({'node': 'synthesizer', 'start': True})
    
    try:
        result = synthesize_answer(
            question=state['question'],
            format_hint=state['format_hint'],
            retrieved_docs=state['retrieved_docs'],
            sql_result=state['sql_result'],
            constraints=state['constraints']
        )
        
        state['final_answer'] = result['answer']
        state['confidence'] = result['confidence']
        state['explanation'] = result['explanation']
        state['citations'] = result['citations']
        
        print(f"  [Synthesizer] Answer: {result['answer']}")
        print(f"  [Synthesizer] Confidence: {result['confidence']:.2f}")
        state['trace'].append({'node': 'synthesizer', 'answer': result['answer']})
    except Exception as e:
        print(f"  [Synthesizer] Error: {e}")
        state['errors'].append(f"Synthesis error: {str(e)}")
    
    return state

def validator_node(state: GraphState) -> GraphState:
    """Validate answer format and citations."""
    print("  [Validator] Checking answer quality...")
    state['trace'].append({'node': 'validator', 'start': True})
    
    is_valid = True
    
    # For RAG-only questions, SQL errors are OK
    is_rag_only = state['route'] == 'rag' or not state['sql']
    
    # Check if SQL had errors (only fail if SQL was needed)
    if state['sql_result'].get('error') and not is_rag_only:
        is_valid = False
        state['errors'].append("SQL execution failed")
    
    # Check if answer matches format hint
    format_hint = state['format_hint']
    answer = state['final_answer']
    
    if format_hint == 'int':
        if not isinstance(answer, int):
            is_valid = False
            state['errors'].append(f"Expected int, got {type(answer).__name__}")
    elif format_hint == 'float':
        if not isinstance(answer, (int, float)):
            is_valid = False
            state['errors'].append(f"Expected float, got {type(answer).__name__}")
    elif '{category:str, quantity:int}' in format_hint:
        if not isinstance(answer, dict) or 'category' not in answer:
            is_valid = False
            state['errors'].append(f"Expected dict with category/quantity")
        elif answer.get('category') == '' or answer.get('quantity') == 0:
            # Empty result - probably SQL failed
            if not is_rag_only:
                is_valid = False
                state['errors'].append("Got empty result from query")
    elif 'list[' in format_hint:
        if not isinstance(answer, list):
            is_valid = False
            state['errors'].append(f"Expected list, got {type(answer).__name__}")
        elif len(answer) == 0:
            is_valid = False
            state['errors'].append("Got empty list result")
    elif '{customer:str, margin:float}' in format_hint:
        if not isinstance(answer, dict) or 'customer' not in answer:
            is_valid = False
            state['errors'].append(f"Expected dict with customer/margin")
        elif answer.get('customer') == '':
            is_valid = False
            state['errors'].append("Got empty customer result")
    
    # Don't fail on missing citations, just warn
    if not state['citations']:
        print("  [Validator] Warning: No citations provided")
    
    if is_valid:
        print("  [Validator] ✓ Answer is valid")
    else:
        print(f"  [Validator] ✗ Validation failed: {state['errors']}")
    
    state['trace'].append({'node': 'validator', 'is_valid': is_valid})
    
    return state

def repair_node(state: GraphState) -> GraphState:
    """Attempt to repair errors."""
    state['repair_count'] += 1
    print(f"  [Repair] Attempt #{state['repair_count']}")
    state['trace'].append({'node': 'repair', 'attempt': state['repair_count']})
    
    if state['sql_result'].get('error'):
        print("  [Repair] Clearing failed SQL for regeneration")
        state['sql'] = ""
    
    state['errors'] = []
    
    return state

def should_repair(state: GraphState) -> Literal["repair", "end"]:
    """Decide if we should repair or end."""
    if state['errors'] and state['repair_count'] < 2:
        print("  [Decision] Routing to repair")
        return "repair"
    print("  [Decision] Ending execution")
    return "end"

def route_after_router(state: GraphState) -> Literal["retriever", "nl_to_sql"]:
    """Route based on classification."""
    route = state['route']
    if route in ['rag', 'hybrid']:
        return "retriever"
    else:
        return "nl_to_sql"

def route_after_planner(state: GraphState) -> Literal["nl_to_sql", "synthesizer"]:
    """Decide if we need SQL."""
    # Check if question needs SQL or is RAG-only
    if 'return' in state['question'].lower() and 'policy' in state['question'].lower():
        return "synthesizer"  # RAG-only
    return "nl_to_sql"

def build_graph() -> StateGraph:
    """Build the LangGraph with ≥6 nodes and repair loop."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("nl_to_sql", nl_to_sql_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("repair", repair_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retriever": "retriever",
            "nl_to_sql": "nl_to_sql"
        }
    )
    
    workflow.add_edge("retriever", "planner")
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "nl_to_sql": "nl_to_sql",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("nl_to_sql", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        should_repair,
        {
            "repair": "repair",
            "end": END
        }
    )
    
    workflow.add_edge("repair", "nl_to_sql")
    
    return workflow.compile()

def run_question(graph, question_data: dict) -> dict:
    """Run a single question through the graph."""
    initial_state = GraphState(
        question=question_data['question'],
        question_id=question_data['id'],
        format_hint=question_data['format_hint'],
        route=None,
        retrieved_docs=[],
        constraints={},
        sql="",
        sql_result={},
        final_answer=None,
        confidence=0.0,
        explanation="",
        citations=[],
        errors=[],
        repair_count=0,
        trace=[]
    )
    
    final_state = graph.invoke(initial_state)
    
    return {
        "id": final_state['question_id'],
        "final_answer": final_state['final_answer'],
        "sql": final_state['sql'],
        "confidence": final_state['confidence'],
        "explanation": final_state['explanation'],
        "citations": final_state['citations']
    }