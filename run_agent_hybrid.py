#!/usr/bin/env python3
"""Main entrypoint for the Retail Analytics Copilot."""
import click
import json
from pathlib import Path
from rich.console import Console
from agent.graph_hybrid import build_graph, run_question

console = Console()

@click.command()
@click.option('--batch', required=True, help='Path to JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file path')
def main(batch: str, out: str):
    """Run the hybrid retail analytics agent in batch mode."""
    console.print(f"[bold blue]Loading questions from {batch}...[/bold blue]")
    
    # Load questions
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    console.print(f"[green]Loaded {len(questions)} questions[/green]")
    
    # Build graph
    console.print("[bold blue]Building agent graph...[/bold blue]")
    graph = build_graph()
    
    # Process each question
    results = []
    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold yellow]Question {i}/{len(questions)}: {q['id']}[/bold yellow]")
        console.print(f"[dim]{q['question']}[/dim]")
        
        result = run_question(graph, q)
        results.append(result)
        
        console.print(f"[green]✓ Answer: {result['final_answer']}[/green]")
        console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")
    
    # Write outputs
    console.print(f"\n[bold blue]Writing results to {out}...[/bold blue]")
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"[bold green]✓ Done! Processed {len(results)} questions[/bold green]")

if __name__ == '__main__':
    main()