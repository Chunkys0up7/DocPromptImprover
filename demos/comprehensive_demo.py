"""
Comprehensive demonstration of the evaluation framework.

This script demonstrates all major features of the evaluation framework including
field evaluation, document aggregation, error pattern detection, and statistics.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text

from src.evaluators.field_evaluator import FieldEvaluator
from src.evaluators.document_aggregator import DocumentAggregator
from src.evaluators.error_pattern_detector import ErrorPatternDetector
from src.statistics.statistics_engine import StatisticsEngine
from src.models.evaluation_models import (
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    ExtractionStatus
)
from data.dummy_data_generator import DummyDataGenerator

console = Console()


class ComprehensiveDemo:
    """
    Comprehensive demonstration of the evaluation framework.
    
    This class provides a complete demonstration of all framework features
    with rich console output and detailed analysis.
    """
    
    def __init__(self):
        """Initialize the comprehensive demo."""
        self.field_evaluator = FieldEvaluator()
        self.document_aggregator = DocumentAggregator()
        self.error_detector = ErrorPatternDetector()
        self.statistics_engine = StatisticsEngine()
        self.data_generator = DummyDataGenerator()
        
        console.print(Panel.fit(
            "[bold blue]Document Extraction Evaluation Framework[/bold blue]\n"
            "[italic]Comprehensive Demonstration[/italic]",
            border_style="blue"
        ))
    
    def run_demo(self):
        """Run the complete demonstration."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Generate test data
            task1 = progress.add_task("Generating test data...", total=None)
            evaluation_inputs = self.data_generator.generate_evaluation_inputs(50)
            progress.update(task1, completed=True)
            
            # Step 2: Field-level evaluation
            task2 = progress.add_task("Performing field-level evaluations...", total=len(evaluation_inputs))
            field_evaluations = []
            for input_data in evaluation_inputs:
                field_evaluations.extend(self._evaluate_document_fields(input_data))
                progress.advance(task2)
            
            # Step 3: Document aggregation
            task3 = progress.add_task("Aggregating document results...", total=len(evaluation_inputs))
            document_results = []
            for input_data in evaluation_inputs:
                doc_result = self._aggregate_document(input_data, field_evaluations)
                document_results.append(doc_result)
                progress.advance(task3)
            
            # Step 4: Statistics collection
            task4 = progress.add_task("Collecting statistics...", total=len(document_results))
            for doc_result in document_results:
                self.statistics_engine.update_statistics(doc_result)
                progress.advance(task4)
            
            # Step 5: Error pattern detection
            task5 = progress.add_task("Detecting error patterns...", total=None)
            error_patterns = self.error_detector.detect_patterns(document_results)
            progress.update(task5, completed=True)
            
            # Step 6: Analysis and reporting
            task6 = progress.add_task("Generating analysis reports...", total=None)
            self._generate_reports(document_results, error_patterns)
            progress.update(task6, completed=True)
        
        console.print("\n[bold green]✓ Demo completed successfully![/bold green]\n")
    
    def _evaluate_document_fields(self, evaluation_input: DocumentEvaluationInput) -> List[Any]:
        """Evaluate fields for a single document."""
        
        field_evaluations = []
        
        for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
            expected_value = evaluation_input.ground_truth.get(field_name)
            extracted_value = evaluation_input.extracted_fields.get(field_name)
            confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
            
            # Determine field type
            field_type = self._determine_field_type(field_name, expected_value)
            
            # Evaluate field
            field_result = self.field_evaluator.evaluate_field(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                field_type=field_type
            )
            field_evaluations.append(field_result)
        
        return field_evaluations
    
    def _aggregate_document(self, evaluation_input: DocumentEvaluationInput, all_field_evaluations: List[Any]) -> DocumentEvaluationResult:
        """Aggregate field evaluations into document result."""
        
        # Find field evaluations for this document
        doc_field_evaluations = []
        for field_eval in all_field_evaluations:
            # This is a simplified approach - in practice, you'd track which fields belong to which document
            if field_eval.field_name in evaluation_input.extracted_fields or field_eval.field_name in evaluation_input.ground_truth:
                doc_field_evaluations.append(field_eval)
        
        # Aggregate document result
        document_result = self.document_aggregator.aggregate_evaluations(
            field_evaluations=doc_field_evaluations,
            document_id=evaluation_input.document_id,
            document_type=evaluation_input.document_type,
            confidence_scores=evaluation_input.confidence_scores,
            prompt_version=evaluation_input.prompt_version
        )
        
        return document_result
    
    def _determine_field_type(self, field_name: str, value: Any) -> str:
        """Determine field type based on field name and value."""
        
        if any(date_word in field_name.lower() for date_word in ["date", "birth"]):
            return "date"
        elif any(num_word in field_name.lower() for num_word in ["amount", "total", "tax", "number"]):
            return "number"
        elif "email" in field_name.lower():
            return "email"
        elif "phone" in field_name.lower():
            return "phone"
        else:
            return "text"
    
    def _generate_reports(self, document_results: List[DocumentEvaluationResult], error_patterns: List[Any]):
        """Generate comprehensive analysis reports."""
        
        # Overall statistics
        self._display_overall_statistics(document_results)
        
        # Field performance analysis
        self._display_field_performance()
        
        # Document type performance
        self._display_document_type_performance()
        
        # Error pattern analysis
        self._display_error_patterns(error_patterns)
        
        # Performance trends
        self._display_performance_trends()
        
        # Quality assessment
        self._display_quality_assessment(document_results)
    
    def _display_overall_statistics(self, document_results: List[DocumentEvaluationResult]):
        """Display overall statistics."""
        
        console.print("\n[bold cyan]Overall Statistics[/bold cyan]")
        
        # Calculate basic statistics
        total_documents = len(document_results)
        total_fields = sum(len(doc.field_evaluations) for doc in document_results)
        
        successful_fields = 0
        failed_fields = 0
        partial_fields = 0
        missing_fields = 0
        
        for doc in document_results:
            for field_eval in doc.field_evaluations:
                if field_eval.is_successful():
                    successful_fields += 1
                elif field_eval.is_failed():
                    failed_fields += 1
                elif field_eval.is_partial():
                    partial_fields += 1
                elif field_eval.status == ExtractionStatus.MISSING:
                    missing_fields += 1
        
        # Create statistics table
        table = Table(title="Overall Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Percentage", style="yellow")
        
        table.add_row("Total Documents", str(total_documents), "100%")
        table.add_row("Total Fields", str(total_fields), "100%")
        table.add_row("Successful Fields", str(successful_fields), f"{successful_fields/total_fields*100:.1f}%")
        table.add_row("Failed Fields", str(failed_fields), f"{failed_fields/total_fields*100:.1f}%")
        table.add_row("Partial Fields", str(partial_fields), f"{partial_fields/total_fields*100:.1f}%")
        table.add_row("Missing Fields", str(missing_fields), f"{missing_fields/total_fields*100:.1f}%")
        
        console.print(table)
        
        # Average accuracy
        avg_accuracy = sum(doc.overall_accuracy for doc in document_results) / total_documents
        console.print(f"\n[bold]Average Document Accuracy:[/bold] {avg_accuracy:.3f}")
    
    def _display_field_performance(self):
        """Display field performance analysis."""
        
        console.print("\n[bold cyan]Field Performance Analysis[/bold cyan]")
        
        field_performance = self.statistics_engine.get_field_performance()
        
        if not field_performance:
            console.print("[yellow]No field performance data available[/yellow]")
            return
        
        # Create field performance table
        table = Table(title="Field-Level Performance")
        table.add_column("Field Name", style="cyan")
        table.add_column("Success Rate", style="green")
        table.add_column("Consistency", style="blue")
        table.add_column("Evaluations", style="yellow")
        table.add_column("Trend", style="magenta")
        
        for field_name, performance in field_performance.items():
            table.add_row(
                field_name,
                f"{performance['success_rate']:.3f}",
                f"{performance['consistency']:.3f}",
                str(performance['total_evaluations']),
                performance['recent_trend']
            )
        
        console.print(table)
    
    def _display_document_type_performance(self):
        """Display document type performance analysis."""
        
        console.print("\n[bold cyan]Document Type Performance[/bold cyan]")
        
        doc_type_performance = self.statistics_engine.get_document_type_performance()
        
        if not doc_type_performance:
            console.print("[yellow]No document type performance data available[/yellow]")
            return
        
        # Create document type performance table
        table = Table(title="Document Type Performance")
        table.add_column("Document Type", style="cyan")
        table.add_column("Average Accuracy", style="green")
        table.add_column("Documents", style="blue")
        table.add_column("Min Accuracy", style="yellow")
        table.add_column("Max Accuracy", style="magenta")
        table.add_column("Trend", style="red")
        
        for doc_type, performance in doc_type_performance.items():
            table.add_row(
                doc_type,
                f"{performance['average_accuracy']:.3f}",
                str(performance['total_documents']),
                f"{performance['min_accuracy']:.3f}",
                f"{performance['max_accuracy']:.3f}",
                performance['trend']
            )
        
        console.print(table)
    
    def _display_error_patterns(self, error_patterns: List[Any]):
        """Display error pattern analysis."""
        
        console.print("\n[bold cyan]Error Pattern Analysis[/bold cyan]")
        
        if not error_patterns:
            console.print("[green]No error patterns detected - excellent performance![/green]")
            return
        
        # Get pattern summary
        pattern_summary = self.error_detector.get_pattern_summary(error_patterns)
        
        # Display summary
        console.print(f"[bold]Total Patterns Detected:[/bold] {pattern_summary['total_patterns']}")
        console.print(f"[bold]Total Failures:[/bold] {pattern_summary['total_failures']}")
        console.print(f"[bold]Average Impact Score:[/bold] {pattern_summary['average_impact_score']:.3f}")
        
        # Display top patterns
        if pattern_summary['top_patterns']:
            table = Table(title="Top Error Patterns")
            table.add_column("Pattern Type", style="cyan")
            table.add_column("Frequency", style="red")
            table.add_column("Impact Score", style="yellow")
            table.add_column("Affected Fields", style="green")
            
            for pattern in pattern_summary['top_patterns'][:5]:
                table.add_row(
                    pattern['pattern_type'],
                    str(pattern['frequency']),
                    f"{pattern['impact_score']:.3f}",
                    ", ".join(pattern['affected_fields'][:3]) + ("..." if len(pattern['affected_fields']) > 3 else "")
                )
            
            console.print(table)
    
    def _display_performance_trends(self):
        """Display performance trends."""
        
        console.print("\n[bold cyan]Performance Trends[/bold cyan]")
        
        trend_analysis = self.statistics_engine.get_trend_analysis(days=30)
        
        if "message" in trend_analysis:
            console.print(f"[yellow]{trend_analysis['message']}[/yellow]")
            return
        
        console.print(f"[bold]Overall Trend:[/bold] {trend_analysis['trend']}")
        console.print(f"[bold]Total Days Analyzed:[/bold] {trend_analysis['total_days']}")
        console.print(f"[bold]Overall Trend Accuracy:[/bold] {trend_analysis['overall_trend_accuracy']:.3f}")
        
        # Display daily averages if available
        if trend_analysis['daily_averages']:
            table = Table(title="Daily Performance Averages")
            table.add_column("Date", style="cyan")
            table.add_column("Average Accuracy", style="green")
            table.add_column("Document Count", style="blue")
            
            for day in trend_analysis['daily_averages'][-7:]:  # Last 7 days
                table.add_row(
                    day['date'],
                    f"{day['average_accuracy']:.3f}",
                    str(day['document_count'])
                )
            
            console.print(table)
    
    def _display_quality_assessment(self, document_results: List[DocumentEvaluationResult]):
        """Display quality assessment."""
        
        console.print("\n[bold cyan]Quality Assessment[/bold cyan]")
        
        # Calculate quality metrics
        quality_scores = []
        for doc_result in document_results:
            quality_score = self._calculate_quality_score(doc_result)
            quality_scores.append(quality_score)
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            
            # Quality distribution
            excellent = sum(1 for score in quality_scores if score >= 0.9)
            good = sum(1 for score in quality_scores if 0.8 <= score < 0.9)
            acceptable = sum(1 for score in quality_scores if 0.7 <= score < 0.8)
            needs_improvement = sum(1 for score in quality_scores if 0.6 <= score < 0.7)
            poor = sum(1 for score in quality_scores if score < 0.6)
            
            # Create quality table
            table = Table(title="Quality Distribution")
            table.add_column("Quality Level", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Percentage", style="yellow")
            
            table.add_row("Excellent (≥0.9)", str(excellent), f"{excellent/len(quality_scores)*100:.1f}%")
            table.add_row("Good (0.8-0.9)", str(good), f"{good/len(quality_scores)*100:.1f}%")
            table.add_row("Acceptable (0.7-0.8)", str(acceptable), f"{acceptable/len(quality_scores)*100:.1f}%")
            table.add_row("Needs Improvement (0.6-0.7)", str(needs_improvement), f"{needs_improvement/len(quality_scores)*100:.1f}%")
            table.add_row("Poor (<0.6)", str(poor), f"{poor/len(quality_scores)*100:.1f}%")
            
            console.print(table)
            
            # Quality summary
            console.print(f"\n[bold]Average Quality Score:[/bold] {avg_quality:.3f}")
            console.print(f"[bold]Quality Range:[/bold] {min_quality:.3f} - {max_quality:.3f}")
    
    def _calculate_quality_score(self, document_result: DocumentEvaluationResult) -> float:
        """Calculate quality score for a document."""
        
        # Weighted combination of accuracy and confidence correlation
        accuracy_weight = 0.7
        correlation_weight = 0.3
        
        quality_score = (
            document_result.overall_accuracy * accuracy_weight +
            document_result.confidence_correlation * correlation_weight
        )
        
        return min(1.0, max(0.0, quality_score))


def main():
    """Run the comprehensive demonstration."""
    
    demo = ComprehensiveDemo()
    demo.run_demo()


if __name__ == "__main__":
    main() 