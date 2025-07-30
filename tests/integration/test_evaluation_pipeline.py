"""
Integration tests for the complete evaluation pipeline.

This module tests the end-to-end evaluation pipeline including field evaluation,
document aggregation, and statistics collection.
"""

import pytest
from datetime import datetime
from typing import Any

from src.evaluators.field_evaluator import FieldEvaluator
from src.evaluators.document_aggregator import DocumentAggregator
from src.statistics.statistics_engine import StatisticsEngine
from src.models.evaluation_models import (
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    FieldEvaluationResult,
    ExtractionStatus
)


class TestEvaluationPipeline:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.fixture
    def field_evaluator(self):
        """Create a field evaluator instance."""
        return FieldEvaluator()
    
    @pytest.fixture
    def document_aggregator(self):
        """Create a document aggregator instance."""
        return DocumentAggregator()
    
    @pytest.fixture
    def statistics_engine(self):
        """Create a statistics engine instance."""
        return StatisticsEngine()
    
    def test_complete_evaluation_pipeline(self, field_evaluator, document_aggregator, statistics_engine):
        """Test the complete evaluation pipeline."""
        
        # Create evaluation input
        evaluation_input = DocumentEvaluationInput(
            document_id="test_invoice_001",
            document_type="invoice",
            extracted_fields={
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "total_amount": "1250.00",
                "due_date": "2024-02-15"
            },
            ground_truth={
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "total_amount": "1250.00",
                "due_date": "2024-02-15"
            },
            confidence_scores={
                "vendor_name": 0.95,
                "invoice_number": 0.98,
                "invoice_date": 0.92,
                "total_amount": 0.96,
                "due_date": 0.89
            },
            prompt_version="v1.0"
        )
        
        # Step 1: Field-level evaluation
        field_evaluations = []
        for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
            expected_value = evaluation_input.ground_truth.get(field_name)
            extracted_value = evaluation_input.extracted_fields.get(field_name)
            confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
            
            # Determine field type
            field_type = self._determine_field_type(field_name, expected_value)
            
            # Evaluate field
            field_result = field_evaluator.evaluate_field(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                field_type=field_type
            )
            field_evaluations.append(field_result)
        
        # Verify field evaluations
        assert len(field_evaluations) == 5
        for field_eval in field_evaluations:
            assert field_eval.status == ExtractionStatus.SUCCESS
            assert field_eval.evaluation_score == 1.0
        
        # Step 2: Document aggregation
        document_result = document_aggregator.aggregate_evaluations(
            field_evaluations=field_evaluations,
            document_id=evaluation_input.document_id,
            document_type=evaluation_input.document_type,
            confidence_scores=evaluation_input.confidence_scores,
            prompt_version=evaluation_input.prompt_version
        )
        
        # Verify document result
        assert document_result.document_id == "test_invoice_001"
        assert document_result.document_type == "invoice"
        assert document_result.overall_accuracy == 1.0
        assert document_result.confidence_correlation > 0.8
        assert len(document_result.field_evaluations) == 5
        
        # Step 3: Statistics collection
        statistics_engine.update_statistics(document_result)
        
        # Verify statistics
        assert statistics_engine.statistics.total_documents == 1
        assert statistics_engine.statistics.total_fields == 5
        assert statistics_engine.statistics.successful_extractions == 5
        assert statistics_engine.statistics.failed_extractions == 0
        assert statistics_engine.statistics.average_accuracy == 1.0
    
    def test_pipeline_with_failures(self, field_evaluator, document_aggregator, statistics_engine):
        """Test pipeline with some field failures."""
        
        # Create evaluation input with some failures
        evaluation_input = DocumentEvaluationInput(
            document_id="test_invoice_002",
            document_type="invoice",
            extracted_fields={
                "vendor_name": "Acme Corp",  # Partial match
                "invoice_number": "INV-2024-001",  # Exact match
                "invoice_date": "01/15/2024",  # Different format
                "total_amount": "1250.00",  # Exact match
                "due_date": None  # Missing
            },
            ground_truth={
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "total_amount": "1250.00",
                "due_date": "2024-02-15"
            },
            confidence_scores={
                "vendor_name": 0.85,
                "invoice_number": 0.98,
                "invoice_date": 0.92,
                "total_amount": 0.96,
                "due_date": 0.0
            },
            prompt_version="v1.0"
        )
        
        # Field-level evaluation
        field_evaluations = []
        for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
            expected_value = evaluation_input.ground_truth.get(field_name)
            extracted_value = evaluation_input.extracted_fields.get(field_name)
            confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
            
            field_type = self._determine_field_type(field_name, expected_value)
            
            field_result = field_evaluator.evaluate_field(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                field_type=field_type
            )
            field_evaluations.append(field_result)
        
        # Verify mixed results
        statuses = [f.status for f in field_evaluations]
        assert ExtractionStatus.SUCCESS in statuses
        assert ExtractionStatus.PARTIAL in statuses
        assert ExtractionStatus.MISSING in statuses
        
        # Document aggregation
        document_result = document_aggregator.aggregate_evaluations(
            field_evaluations=field_evaluations,
            document_id=evaluation_input.document_id,
            document_type=evaluation_input.document_type,
            confidence_scores=evaluation_input.confidence_scores,
            prompt_version=evaluation_input.prompt_version
        )
        
        # Verify aggregated result
        assert 0.0 < document_result.overall_accuracy < 1.0
        assert document_result.confidence_correlation > 0.0
        
        # Statistics collection
        statistics_engine.update_statistics(document_result)
        
        # Verify updated statistics
        assert statistics_engine.statistics.total_documents == 2
        assert statistics_engine.statistics.total_fields == 10
        assert statistics_engine.statistics.successful_extractions > 5
        assert statistics_engine.statistics.failed_extractions > 0
    
    def test_batch_evaluation_pipeline(self, field_evaluator, document_aggregator, statistics_engine):
        """Test batch evaluation pipeline."""
        
        # Create multiple evaluation inputs
        evaluation_inputs = []
        for i in range(3):
            evaluation_input = DocumentEvaluationInput(
                document_id=f"batch_test_{i+1}",
                document_type="invoice",
                extracted_fields={
                    "vendor_name": f"Vendor {i+1}",
                    "invoice_number": f"INV-2024-{i+1:03d}",
                    "total_amount": f"{1000 + i*100}.00"
                },
                ground_truth={
                    "vendor_name": f"Vendor {i+1}",
                    "invoice_number": f"INV-2024-{i+1:03d}",
                    "total_amount": f"{1000 + i*100}.00"
                },
                confidence_scores={
                    "vendor_name": 0.9,
                    "invoice_number": 0.95,
                    "total_amount": 0.92
                },
                prompt_version="v1.0"
            )
            evaluation_inputs.append(evaluation_input)
        
        # Process each evaluation
        document_results = []
        for evaluation_input in evaluation_inputs:
            # Field evaluation
            field_evaluations = []
            for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
                expected_value = evaluation_input.ground_truth.get(field_name)
                extracted_value = evaluation_input.extracted_fields.get(field_name)
                confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
                
                field_type = self._determine_field_type(field_name, expected_value)
                
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
            document_results.append(document_result)
            
            # Statistics collection
            statistics_engine.update_statistics(document_result)
        
        # Verify batch results
        assert len(document_results) == 3
        for result in document_results:
            assert result.overall_accuracy > 0.8
        
        # Verify batch statistics
        assert statistics_engine.statistics.total_documents == 5  # Including previous tests
        assert statistics_engine.statistics.total_fields == 19  # Including previous tests
    
    def test_performance_metrics(self, statistics_engine):
        """Test performance metrics calculation."""
        
        # Generate some test data first
        self._generate_test_data(statistics_engine)
        
        # Test performance metrics
        metrics = statistics_engine.get_performance_metrics()
        
        assert "total_documents" in metrics
        assert "average_accuracy" in metrics
        assert "success_rate" in metrics
        assert "failure_rate" in metrics
        assert metrics["total_documents"] > 0
        assert 0.0 <= metrics["average_accuracy"] <= 1.0
    
    def test_field_performance_analysis(self, statistics_engine):
        """Test field performance analysis."""
        
        # Generate test data
        self._generate_test_data(statistics_engine)
        
        # Get field performance
        field_performance = statistics_engine.get_field_performance()
        
        assert isinstance(field_performance, dict)
        for field_name, performance in field_performance.items():
            assert "success_rate" in performance
            assert "consistency" in performance
            assert "total_evaluations" in performance
            assert "recent_trend" in performance
            assert 0.0 <= performance["success_rate"] <= 1.0
    
    def test_document_type_performance(self, statistics_engine):
        """Test document type performance analysis."""
        
        # Generate test data
        self._generate_test_data(statistics_engine)
        
        # Get document type performance
        doc_type_performance = statistics_engine.get_document_type_performance()
        
        assert isinstance(doc_type_performance, dict)
        for doc_type, performance in doc_type_performance.items():
            assert "average_accuracy" in performance
            assert "total_documents" in performance
            assert "trend" in performance
            assert 0.0 <= performance["average_accuracy"] <= 1.0
    
    def test_error_analysis(self, statistics_engine):
        """Test error analysis functionality."""
        
        # Generate test data
        self._generate_test_data(statistics_engine)
        
        # Get error analysis
        error_analysis = statistics_engine.get_error_analysis()
        
        assert "top_errors" in error_analysis
        assert "error_categories" in error_analysis
        assert "confidence_analysis" in error_analysis
        assert "total_errors" in error_analysis
    
    def test_trend_analysis(self, statistics_engine):
        """Test trend analysis functionality."""
        
        # Generate test data
        self._generate_test_data(statistics_engine)
        
        # Get trend analysis
        trend_analysis = statistics_engine.get_trend_analysis(days=30)
        
        if "message" not in trend_analysis:  # If we have sufficient data
            assert "trend" in trend_analysis
            assert "daily_averages" in trend_analysis
            assert "total_days" in trend_analysis
            assert "overall_trend_accuracy" in trend_analysis
    
    def test_optimization_metrics(self, statistics_engine):
        """Test optimization metrics generation."""
        
        # Generate test data
        self._generate_test_data(statistics_engine)
        
        # Get optimization metrics
        optimization_metrics = statistics_engine.get_optimization_metrics()
        
        assert isinstance(optimization_metrics, dict)
        for metric_name, value in optimization_metrics.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
    
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
    
    def _generate_test_data(self, statistics_engine):
        """Generate test data for statistics engine."""
        
        # Create some test evaluation results
        from src.evaluators.field_evaluator import FieldEvaluator
        from src.evaluators.document_aggregator import DocumentAggregator
        
        field_evaluator = FieldEvaluator()
        document_aggregator = DocumentAggregator()
        
        # Generate a few test documents
        for i in range(5):
            field_evaluations = []
            
            # Create field evaluations
            for field_name in ["vendor_name", "invoice_number", "total_amount"]:
                expected_value = f"test_value_{i}"
                extracted_value = f"test_value_{i}" if i % 2 == 0 else f"wrong_value_{i}"
                confidence_score = 0.9 if i % 2 == 0 else 0.5
                
                field_result = field_evaluator.evaluate_field(
                    field_name=field_name,
                    expected_value=expected_value,
                    extracted_value=extracted_value,
                    confidence_score=confidence_score,
                    field_type="text"
                )
                field_evaluations.append(field_result)
            
            # Create document result
            document_result = document_aggregator.aggregate_evaluations(
                field_evaluations=field_evaluations,
                document_id=f"test_doc_{i}",
                document_type="invoice",
                confidence_scores={"vendor_name": 0.9, "invoice_number": 0.9, "total_amount": 0.9},
                prompt_version="v1.0"
            )
            
            # Update statistics
            statistics_engine.update_statistics(document_result) 