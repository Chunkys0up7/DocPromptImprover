#!/usr/bin/env python3
"""
Command-line interface for the Document Extraction Evaluation Framework.

This module provides a CLI for running evaluations, generating reports,
and managing the evaluation framework.
"""

import click
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..evaluators.field_evaluator import FieldEvaluator
from ..evaluators.document_aggregator import DocumentAggregator
from ..evaluators.error_pattern_detector import ErrorPatternDetector
from ..statistics.statistics_engine import StatisticsEngine
from ..models.evaluation_models import DocumentEvaluationInput
from data.dummy_data_generator import DummyDataGenerator


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Document Extraction Evaluation Framework CLI.
    
    This tool provides command-line access to the evaluation framework
    for assessing document extraction performance and generating reports.
    """
    pass


@cli.command()
@click.option("--input-file", "-i", type=click.Path(exists=True), help="Input JSON file with evaluation data")
@click.option("--output-file", "-o", type=click.Path(), help="Output file for results")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def evaluate(input_file: Optional[str], output_file: Optional[str], config: Optional[str], verbose: bool):
    """Run document extraction evaluation."""
    
    click.echo("🔍 Running Document Extraction Evaluation")
    
    if input_file:
        # Load evaluation data from file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        evaluation_inputs = [DocumentEvaluationInput(**item) for item in data.get("evaluation_inputs", [])]
    else:
        # Generate dummy data
        click.echo("📊 Generating dummy evaluation data...")
        generator = DummyDataGenerator()
        evaluation_inputs = generator.generate_evaluation_inputs(10)
    
    # Initialize evaluators
    field_evaluator = FieldEvaluator()
    document_aggregator = DocumentAggregator()
    statistics_engine = StatisticsEngine()
    
    # Process evaluations
    results = []
    with click.progressbar(evaluation_inputs, label="Processing evaluations") as inputs:
        for evaluation_input in inputs:
            # Field-level evaluation
            field_evaluations = []
            for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
                expected_value = evaluation_input.ground_truth.get(field_name)
                extracted_value = evaluation_input.extracted_fields.get(field_name)
                confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
                
                field_type = _determine_field_type(field_name, expected_value)
                
                field_result = field_evaluator.evaluate_field(
                    field_name=field_name,
                    expected_value=expected_value,
                    extracted_value=extracted_value,
                    confidence_score=confidence_score,
                    field_type=field_type
                )
                field_evaluations.append(field_result)
            
            # Document aggregation
            document_result = document_aggregator.aggregate_evaluations(
                field_evaluations=field_evaluations,
                document_id=evaluation_input.document_id,
                document_type=evaluation_input.document_type,
                confidence_scores=evaluation_input.confidence_scores,
                prompt_version=evaluation_input.prompt_version
            )
            
            results.append(document_result)
            statistics_engine.update_statistics(document_result)
    
    # Generate reports
    click.echo("\n📈 Generating Reports...")
    
    # Overall statistics
    metrics = statistics_engine.get_performance_metrics()
    click.echo(f"Average Accuracy: {metrics.get('average_accuracy', 0):.3f}")
    click.echo(f"Success Rate: {metrics.get('success_rate', 0):.3f}")
    click.echo(f"Failure Rate: {metrics.get('failure_rate', 0):.3f}")
    
    # Field performance
    field_performance = statistics_engine.get_field_performance()
    if field_performance:
        click.echo(f"\nField Performance:")
        for field_name, performance in field_performance.items():
            click.echo(f"  {field_name}: {performance['success_rate']:.3f}")
    
    # Save results
    if output_file:
        output_data = {
            "results": [result.dict() for result in results],
            "statistics": statistics_engine.statistics.dict(),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        click.echo(f"\n✅ Results saved to {output_file}")
    
    click.echo("\n🎉 Evaluation completed successfully!")


@cli.command()
@click.option("--num-documents", "-n", default=50, help="Number of documents to generate")
@click.option("--output-file", "-o", default="dummy_data.json", help="Output file")
@click.option("--include-edge-cases", is_flag=True, help="Include edge cases")
def generate_data(num_documents: int, output_file: str, include_edge_cases: bool):
    """Generate dummy evaluation data."""
    
    click.echo(f"📊 Generating {num_documents} dummy evaluation documents...")
    
    generator = DummyDataGenerator()
    generator.save_dummy_data(output_file, num_documents)
    
    if include_edge_cases:
        click.echo("🔍 Including edge cases...")
        # Edge cases are already included in the generator
    
    click.echo(f"✅ Dummy data saved to {output_file}")


@cli.command()
@click.option("--input-file", "-i", type=click.Path(exists=True), required=True, help="Input file with evaluation results")
@click.option("--output-file", "-o", type=click.Path(), help="Output file for analysis")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "html"]), default="json", help="Output format")
def analyze(input_file: str, output_file: Optional[str], format: str):
    """Analyze evaluation results and generate insights."""
    
    click.echo("🔍 Analyzing evaluation results...")
    
    # Load results
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize analyzers
    error_detector = ErrorPatternDetector()
    statistics_engine = StatisticsEngine()
    
    # Load results into statistics engine
    results = []
    for result_data in data.get("results", []):
        # This would need proper deserialization in practice
        # For now, we'll work with the data as-is
        pass
    
    # Detect error patterns
    error_patterns = error_detector.detect_patterns(results)
    
    # Generate analysis
    analysis = {
        "error_patterns": [pattern.dict() for pattern in error_patterns],
        "pattern_summary": error_detector.get_pattern_summary(error_patterns),
        "performance_metrics": statistics_engine.get_performance_metrics(),
        "field_performance": statistics_engine.get_field_performance(),
        "document_type_performance": statistics_engine.get_document_type_performance(),
        "trend_analysis": statistics_engine.get_trend_analysis(),
        "generated_at": datetime.now().isoformat()
    }
    
    # Save analysis
    if output_file:
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
        elif format == "csv":
            # Convert to CSV format
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for key, value in analysis.items():
                    if isinstance(value, (str, int, float)):
                        writer.writerow([key, value])
        elif format == "html":
            # Generate HTML report
            html_content = _generate_html_report(analysis)
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        click.echo(f"✅ Analysis saved to {output_file}")
    else:
        # Print summary to console
        click.echo(f"\n📊 Analysis Summary:")
        click.echo(f"Error Patterns: {len(error_patterns)}")
        click.echo(f"Performance Metrics: {len(analysis['performance_metrics'])}")
        click.echo(f"Field Performance: {len(analysis['field_performance'])}")


@cli.command()
@click.option("--port", "-p", default=8000, help="Port to run the server on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(port: int, host: str, reload: bool):
    """Start the evaluation API server."""
    
    click.echo(f"🚀 Starting evaluation API server on {host}:{port}")
    
    try:
        import uvicorn
        from ..api.evaluation_service import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        click.echo("❌ uvicorn not found. Please install it: pip install uvicorn")
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}")


@cli.command()
@click.option("--test-type", "-t", type=click.Choice(["unit", "integration", "all"]), default="all", help="Type of tests to run")
@click.option("--coverage", is_flag=True, help="Generate coverage report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(test_type: str, coverage: bool, verbose: bool):
    """Run tests."""
    
    click.echo(f"🧪 Running {test_type} tests...")
    
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    else:
        cmd.append("tests/")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, check=True)
        click.echo("✅ All tests passed!")
    except subprocess.CalledProcessError:
        click.echo("❌ Some tests failed!")
        sys.exit(1)


@cli.command()
def demo():
    """Run the comprehensive demonstration."""
    
    click.echo("🎬 Running comprehensive demonstration...")
    
    try:
        from demos.comprehensive_demo import main as run_demo
        run_demo()
    except ImportError as e:
        click.echo(f"❌ Failed to import demo: {e}")
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")


def _determine_field_type(field_name: str, value) -> str:
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


def _generate_html_report(analysis: dict) -> str:
    """Generate HTML report from analysis data."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
            .metric { margin: 10px 0; }
            .metric-name { font-weight: bold; }
            .metric-value { color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Document Extraction Evaluation Analysis</h1>
        <p>Generated at: {generated_at}</p>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            {performance_metrics}
        </div>
        
        <div class="section">
            <h2>Error Patterns</h2>
            <p>Total Patterns: {pattern_count}</p>
            {error_patterns}
        </div>
        
        <div class="section">
            <h2>Field Performance</h2>
            {field_performance}
        </div>
    </body>
    </html>
    """
    
    # Format the data for HTML
    performance_metrics_html = ""
    for key, value in analysis.get("performance_metrics", {}).items():
        if isinstance(value, (int, float)):
            performance_metrics_html += f'<div class="metric"><span class="metric-name">{key}:</span> <span class="metric-value">{value:.3f}</span></div>'
    
    error_patterns_html = ""
    for pattern in analysis.get("error_patterns", [])[:5]:  # Top 5 patterns
        error_patterns_html += f'<div class="metric"><span class="metric-name">{pattern.get("pattern_type", "Unknown")}:</span> <span class="metric-value">{pattern.get("frequency", 0)} occurrences</span></div>'
    
    field_performance_html = ""
    for field_name, performance in analysis.get("field_performance", {}).items():
        field_performance_html += f'<div class="metric"><span class="metric-name">{field_name}:</span> <span class="metric-value">{performance.get("success_rate", 0):.3f}</span></div>'
    
    return html.format(
        generated_at=analysis.get("generated_at", ""),
        performance_metrics=performance_metrics_html,
        pattern_count=len(analysis.get("error_patterns", [])),
        error_patterns=error_patterns_html,
        field_performance=field_performance_html
    )


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main() 