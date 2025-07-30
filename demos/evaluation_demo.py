"""
Demo script for the evaluation-only document extraction framework.

This demo showcases how the evaluation service can assess the outputs of
existing OCR-plus-prompt pipelines and provide optimization feedback.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from src.models.evaluation_models import (
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    FieldEvaluationResult,
    ExtractionStatus
)
from src.api.evaluation_service import DocumentExtractionEvaluator

console = Console()


def create_mock_evaluation_data() -> List[Dict[str, Any]]:
    """Create mock evaluation data simulating OCR-plus-prompt pipeline outputs."""
    
    return [
        {
            "document_id": "invoice_001",
            "document_type": "invoice",
            "extracted_fields": {
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "total_amount": "1250.00",
                "due_date": "2024-02-15"
            },
            "ground_truth": {
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "total_amount": "1250.00",
                "due_date": "2024-02-15"
            },
            "confidence_scores": {
                "vendor_name": 0.95,
                "invoice_number": 0.98,
                "invoice_date": 0.92,
                "total_amount": 0.96,
                "due_date": 0.89
            }
        },
        {
            "document_id": "invoice_002",
            "document_type": "invoice",
            "extracted_fields": {
                "vendor_name": "Tech Solutions Inc",
                "invoice_number": "TSI-2024-005",
                "invoice_date": "2024-01-20",
                "total_amount": "875.50",
                "due_date": "2024-02-20"
            },
            "ground_truth": {
                "vendor_name": "Tech Solutions Inc.",
                "invoice_number": "TSI-2024-005",
                "invoice_date": "2024-01-20",
                "total_amount": "875.50",
                "due_date": "2024-02-20"
            },
            "confidence_scores": {
                "vendor_name": 0.88,
                "invoice_number": 0.95,
                "invoice_date": 0.91,
                "total_amount": 0.97,
                "due_date": 0.85
            }
        },
        {
            "document_id": "receipt_001",
            "document_type": "receipt",
            "extracted_fields": {
                "merchant_name": "Coffee Shop",
                "transaction_date": "2024-01-25",
                "total_amount": "12.75",
                "payment_method": "Credit Card"
            },
            "ground_truth": {
                "merchant_name": "Coffee Shop",
                "transaction_date": "2024-01-25",
                "total_amount": "12.75",
                "payment_method": "Credit Card"
            },
            "confidence_scores": {
                "merchant_name": 0.92,
                "transaction_date": 0.94,
                "total_amount": 0.99,
                "payment_method": 0.87
            }
        },
        {
            "document_id": "invoice_003",
            "document_type": "invoice",
            "extracted_fields": {
                "vendor_name": "Global Services",
                "invoice_number": "GS-2024-012",
                "invoice_date": "2024-01-30",
                "total_amount": "2100.00",
                "due_date": "2024-03-01"
            },
            "ground_truth": {
                "vendor_name": "Global Services Ltd",
                "invoice_number": "GS-2024-012",
                "invoice_date": "2024-01-30",
                "total_amount": "2100.00",
                "due_date": "2024-03-01"
            },
            "confidence_scores": {
                "vendor_name": 0.82,
                "invoice_number": 0.96,
                "invoice_date": 0.93,
                "total_amount": 0.98,
                "due_date": 0.90
            }
        }
    ]


def display_evaluation_results(results: List[DocumentEvaluationResult]) -> None:
    """Display evaluation results in a formatted table."""
    
    # Create summary table
    summary_table = Table(title="📊 Document Evaluation Summary")
    summary_table.add_column("Document ID", style="cyan")
    summary_table.add_column("Type", style="magenta")
    summary_table.add_column("Overall Accuracy", style="green")
    summary_table.add_column("Fields Evaluated", style="yellow")
    summary_table.add_column("Successful Fields", style="green")
    summary_table.add_column("Failed Fields", style="red")
    
    for result in results:
        successful_count = len([f for f in result.field_evaluations if f.is_successful()])
        failed_count = len([f for f in result.field_evaluations if f.is_failed()])
        
        summary_table.add_row(
            result.document_id,
            result.document_type,
            f"{result.overall_accuracy:.1%}",
            str(len(result.field_evaluations)),
            str(successful_count),
            str(failed_count)
        )
    
    console.print(summary_table)
    console.print()


def display_field_details(results: List[DocumentEvaluationResult]) -> None:
    """Display detailed field-level evaluation results."""
    
    for result in results:
        # Create field details table for each document
        field_table = Table(title=f"📋 Field Details - {result.document_id}")
        field_table.add_column("Field Name", style="cyan")
        field_table.add_column("Expected", style="blue")
        field_table.add_column("Extracted", style="yellow")
        field_table.add_column("Status", style="green")
        field_table.add_column("Score", style="magenta")
        field_table.add_column("Confidence", style="blue")
        
        for field in result.field_evaluations:
            status_color = {
                ExtractionStatus.SUCCESS: "green",
                ExtractionStatus.PARTIAL: "yellow",
                ExtractionStatus.FAILED: "red",
                ExtractionStatus.MISSING: "red"
            }.get(field.status, "white")
            
            field_table.add_row(
                field.field_name,
                str(field.expected_value or "N/A"),
                str(field.extracted_value or "N/A"),
                f"[{status_color}]{field.status.value}[/{status_color}]",
                f"{field.evaluation_score:.2f}",
                f"{field.confidence_score:.2f}"
            )
        
        console.print(field_table)
        console.print()


def display_optimization_demo(evaluator: DocumentExtractionEvaluator) -> None:
    """Demonstrate prompt optimization capabilities."""
    
    console.print(Panel.fit(
        "🚀 Prompt Optimization Demo",
        style="bold blue"
    ))
    
    # Mock current prompt
    current_prompt = """
    Extract the following fields from the document:
    - vendor_name: The name of the vendor or company
    - invoice_number: The invoice number
    - invoice_date: The date of the invoice
    - total_amount: The total amount due
    - due_date: The due date for payment
    
    Please provide accurate and complete information.
    """
    
    console.print(f"[bold]Current Prompt:[/bold]\n{current_prompt}")
    console.print()
    
    # Simulate optimization request
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating optimization recommendations...", total=None)
        
        # Note: In a real scenario, this would use the actual optimizer
        # For demo purposes, we'll simulate the process
        time.sleep(2)
        
        progress.update(task, description="Optimization complete!")
    
    # Mock optimization result
    optimized_prompt = """
    Extract the following fields from the document with high precision:
    
    - vendor_name: The complete legal name of the vendor or company (include full company name with Inc, Corp, Ltd, etc.)
    - invoice_number: The invoice number or reference number (preserve exact format including prefixes)
    - invoice_date: The date of the invoice in YYYY-MM-DD format
    - total_amount: The total amount due as a decimal number (e.g., 1250.00)
    - due_date: The payment due date in YYYY-MM-DD format
    
    Important guidelines:
    - For vendor names, include the complete legal entity name
    - For dates, always use YYYY-MM-DD format
    - For amounts, include cents even if .00
    - Preserve exact formatting from the original document
    """
    
    console.print(f"[bold green]Optimized Prompt:[/bold green]\n{optimized_prompt}")
    console.print()
    
    # Show expected improvements
    improvement_table = Table(title="📈 Expected Improvements")
    improvement_table.add_column("Metric", style="cyan")
    improvement_table.add_column("Current", style="yellow")
    improvement_table.add_column("Expected", style="green")
    improvement_table.add_column("Improvement", style="magenta")
    
    improvement_table.add_row("Vendor Name Accuracy", "85%", "95%", "+10%")
    improvement_table.add_row("Date Format Consistency", "90%", "98%", "+8%")
    improvement_table.add_row("Overall Accuracy", "88%", "94%", "+6%")
    
    console.print(improvement_table)


def run_evaluation_demo() -> None:
    """Run the complete evaluation demo."""
    
    console.print(Panel.fit(
        "🔍 Document Extraction Evaluation Framework Demo",
        style="bold blue"
    ))
    
    console.print("This demo showcases the evaluation-only framework for assessing\n"
                 "OCR-plus-prompt pipeline outputs and generating optimization feedback.\n")
    
    # Create mock data
    mock_data = create_mock_evaluation_data()
    console.print(f"[green]✓[/green] Created {len(mock_data)} mock evaluation samples")
    
    # Initialize evaluator
    evaluator = DocumentExtractionEvaluator()
    console.print("[green]✓[/green] Initialized evaluation service")
    console.print()
    
    # Run evaluations
    console.print(Panel.fit("📊 Running Document Evaluations", style="bold yellow"))
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for i, data in enumerate(mock_data):
            task = progress.add_task(f"Evaluating {data['document_id']}...", total=None)
            
            # Create evaluation input
            evaluation_input = DocumentEvaluationInput(**data)
            
            # Perform evaluation
            result = evaluator.evaluate_document(evaluation_input)
            results.append(result)
            
            progress.update(task, description=f"Completed {data['document_id']}")
            time.sleep(0.5)  # Simulate processing time
    
    console.print(f"[green]✓[/green] Completed {len(results)} evaluations")
    console.print()
    
    # Display results
    display_evaluation_results(results)
    display_field_details(results)
    
    # Show optimization demo
    display_optimization_demo(evaluator)
    
    # Summary
    console.print(Panel.fit(
        "🎯 Evaluation Framework Benefits\n\n"
        "• [green]Decoupled Architecture[/green]: Works with any existing OCR pipeline\n"
        "• [green]Metrics-Driven[/green]: Provides quantifiable performance insights\n"
        "• [green]Optimization Feedback[/green]: Generates actionable improvement recommendations\n"
        "• [green]Lightweight[/green]: CPU-only service, scales with JSON throughput\n"
        "• [green]Continuous Improvement[/green]: Enables data-driven prompt evolution",
        style="bold green"
    ))


def run_batch_evaluation_demo() -> None:
    """Demonstrate batch evaluation capabilities."""
    
    console.print(Panel.fit("📦 Batch Evaluation Demo", style="bold blue"))
    
    # Create larger batch of mock data
    batch_data = []
    for i in range(10):
        base_data = create_mock_evaluation_data()[0]  # Use first example as template
        base_data["document_id"] = f"batch_doc_{i:03d}"
        base_data["extracted_fields"]["invoice_number"] = f"BATCH-{i:03d}"
        batch_data.append(base_data)
    
    evaluator = DocumentExtractionEvaluator()
    
    # Simulate batch processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing batch evaluation...", total=len(batch_data))
        
        batch_results = []
        for data in batch_data:
            evaluation_input = DocumentEvaluationInput(**data)
            result = evaluator.evaluate_document(evaluation_input)
            batch_results.append(result)
            progress.advance(task)
            time.sleep(0.1)
    
    # Calculate batch statistics
    total_accuracy = sum(r.overall_accuracy for r in batch_results) / len(batch_results)
    total_fields = sum(len(r.field_evaluations) for r in batch_results)
    successful_fields = sum(
        len([f for f in r.field_evaluations if f.is_successful()]) 
        for r in batch_results
    )
    
    console.print(f"[green]✓[/green] Batch evaluation completed:")
    console.print(f"  • Documents processed: {len(batch_results)}")
    console.print(f"  • Average accuracy: {total_accuracy:.1%}")
    console.print(f"  • Total fields evaluated: {total_fields}")
    console.print(f"  • Successful extractions: {successful_fields}")
    console.print(f"  • Success rate: {successful_fields/total_fields:.1%}")


if __name__ == "__main__":
    try:
        # Run main demo
        run_evaluation_demo()
        console.print()
        
        # Run batch demo
        run_batch_evaluation_demo()
        
        console.print("\n[bold green]🎉 Demo completed successfully![/bold green]")
        console.print("\nTo use the evaluation service:")
        console.print("1. Start the FastAPI service: python -m src.api.evaluation_service")
        console.print("2. Send POST requests to /evaluate with evaluation data")
        console.print("3. Get statistics from /stats endpoint")
        console.print("4. Generate optimizations via /optimize endpoint")
        
    except Exception as e:
        console.print(f"[bold red]Error running demo: {e}[/bold red]")
        console.print("\nMake sure you have:")
        console.print("• DSPy configured with an LLM provider")
        console.print("• All dependencies installed")
        console.print("• Proper API keys set in environment variables") 