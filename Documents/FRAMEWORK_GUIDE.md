# Document Extraction Evaluation Framework - Complete Guide

## Overview

The Document Extraction Evaluation Framework is a pure statistical evaluation service designed to assess the quality of document extraction results from OCR-plus-prompt pipelines. It provides comprehensive metrics, error analysis, and performance insights without requiring AI/LLM dependencies.

## Architecture

### Core Components

1. **Field Evaluator** - Evaluates individual field extractions
2. **Document Aggregator** - Combines field results into document-level metrics
3. **Error Pattern Detector** - Identifies failure patterns and trends
4. **Statistics Engine** - Collects and analyzes performance data
5. **Evaluation Service** - Main API interface for the framework

### Data Flow

```
Input: DocumentEvaluationInput
    ↓
Field Evaluation (per field)
    ↓
Document Aggregation
    ↓
Error Pattern Detection
    ↓
Statistics Collection
    ↓
Output: DocumentEvaluationResult + Statistics
```

## How Field Evaluation Works

### Field Evaluation Process

1. **Input Processing**
   - Receives expected value (ground truth)
   - Receives extracted value (from OCR/prompt pipeline)
   - Receives confidence score (from extraction model)

2. **Value Normalization**
   - **Text fields**: Case normalization, whitespace normalization
   - **Number fields**: Remove currency symbols, normalize decimals
   - **Date fields**: Standardize date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
   - **Email fields**: Lowercase, trim whitespace
   - **Phone fields**: Remove all non-digits for comparison

3. **Scoring Calculation**
   - **Text fields**: String similarity using SequenceMatcher
   - **Number fields**: Exact match or percentage difference
   - **Date fields**: Date difference in days, normalized to 0-1 scale
   - **Email fields**: Exact match or domain similarity
   - **Phone fields**: Digit sequence comparison

4. **Status Determination**
   - **SUCCESS**: Score ≥ success_threshold (default: 0.8) AND confidence ≥ confidence_threshold (default: 0.7)
   - **PARTIAL**: Score ≥ partial_threshold (default: 0.5) but < success_threshold
   - **FAILED**: Score < partial_threshold OR confidence < confidence_threshold
   - **MISSING**: Extracted value is None/empty

### Scoring Examples

#### Text Field Scoring
```
Expected: "Acme Corporation"
Extracted: "Acme Corp"
Score: 0.72 (72% similarity)
Status: PARTIAL (0.72 < 0.8 success threshold)
```

#### Date Field Scoring
```
Expected: "2024-01-15"
Extracted: "01/15/2024"
Score: 1.0 (same date, different format)
Status: SUCCESS (1.0 ≥ 0.8)
```

#### Phone Field Scoring
```
Expected: "+1-555-123-4567"
Extracted: "5551234567"
Score: 0.91 (missing country code)
Status: PARTIAL (0.91 < 1.0 for strict matching)
```

## Document Aggregation Metrics

### Overall Accuracy
- **Calculation**: Weighted average of field evaluation scores
- **Weights**: Based on field importance (configurable)
- **Range**: 0.0 to 1.0

### Confidence Correlation
- **Calculation**: Pearson correlation between confidence scores and actual accuracy
- **Purpose**: Measures how well confidence scores predict actual performance
- **Special Case**: Returns 1.0 when all scores are perfect (1.0) and confidence is high (≥0.8)

### Field Success Rates
- **Calculation**: Percentage of successful extractions per field
- **Formula**: (Successful fields / Total fields) × 100
- **Usage**: Identifies problematic fields

### Document Type Performance
- **Calculation**: Average accuracy per document type
- **Purpose**: Compare performance across different document types
- **Example**: Invoices vs Receipts vs Forms

## Error Pattern Detection

### Pattern Types

1. **Field-Specific Failures**
   - Fields that consistently fail across documents
   - Example: "date_of_birth" field failing in 90% of cases

2. **Error Message Patterns**
   - Common error messages and their frequency
   - Example: "Field was not extracted" appearing 50 times

3. **Document Type Failures**
   - Patterns specific to document types
   - Example: Medical records having higher failure rates

4. **Confidence Patterns**
   - Low confidence failures (confidence < 0.5)
   - High confidence failures (confidence > 0.8 but wrong)

5. **Value Pattern Failures**
   - Similar value patterns causing failures
   - Example: Date format issues, number formatting problems

### Impact Score Calculation
```
Impact Score = (Frequency Factor × 0.4) + (Importance Factor × 0.4) + (Score Factor × 0.2)

Where:
- Frequency Factor = min(occurrences / 10, 1.0)
- Importance Factor = important_fields_count / total_fields
- Score Factor = 1.0 - average_evaluation_score
```

## Statistics Engine

### Collected Metrics

1. **Basic Counts**
   - Total documents processed
   - Total fields evaluated
   - Successful/partial/failed/missing extractions

2. **Performance Metrics**
   - Average accuracy across all documents
   - Average confidence score
   - Success rates by field and document type

3. **Error Analysis**
   - Common error messages and frequencies
   - Confidence score distribution
   - Field-specific failure patterns

4. **Trend Analysis**
   - Performance over time
   - Daily/weekly averages
   - Improvement/decline detection

### Quality Assessment

#### Quality Levels
- **Excellent**: ≥0.9 accuracy
- **Good**: 0.8-0.9 accuracy
- **Acceptable**: 0.7-0.8 accuracy
- **Needs Improvement**: 0.6-0.7 accuracy
- **Poor**: <0.6 accuracy

#### Quality Score Calculation
```
Quality Score = (Overall Accuracy × 0.4) + (Confidence Correlation × 0.3) + (Success Rate × 0.3)
```

## Configuration Options

### Evaluation Config
```python
EvaluationConfig(
    confidence_threshold=0.7,      # Minimum confidence for success
    success_threshold=0.8,         # Minimum score for success
    partial_threshold=0.5,         # Minimum score for partial
    enable_partial_credit=True,    # Enable partial scoring
    strict_matching=False,         # Exact string matching
    case_sensitive=False,          # Case-sensitive comparison
    normalize_whitespace=True      # Normalize whitespace
)
```

### Field Weights
- **Important fields**: invoice_number, total_amount, vendor_name, invoice_date
- **Standard fields**: All other fields
- **Weight calculation**: Important fields get higher weight in overall accuracy

## Usage Examples

### Basic Field Evaluation
```python
from src.evaluators.field_evaluator import FieldEvaluator

evaluator = FieldEvaluator()
result = evaluator.evaluate_field(
    field_name="vendor_name",
    expected_value="Acme Corporation",
    extracted_value="Acme Corp",
    confidence_score=0.85,
    field_type="text"
)

print(f"Status: {result.status}")
print(f"Score: {result.evaluation_score}")
print(f"Error: {result.error_message}")
```

### Document Evaluation
```python
from src.evaluators.document_aggregator import DocumentAggregator

aggregator = DocumentAggregator()
doc_result = aggregator.aggregate_evaluations(
    field_evaluations=field_results,
    document_id="INV-001",
    document_type="invoice",
    confidence_scores=confidence_scores,
    prompt_version="v1.0"
)

print(f"Overall Accuracy: {doc_result.overall_accuracy}")
print(f"Confidence Correlation: {doc_result.confidence_correlation}")
```

### Error Pattern Detection
```python
from src.evaluators.error_pattern_detector import ErrorPatternDetector

detector = ErrorPatternDetector()
patterns = detector.detect_patterns(document_results)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Frequency: {pattern.frequency}")
    print(f"Impact: {pattern.impact_score}")
    print(f"Fixes: {pattern.suggested_fixes}")
```

## Running Demos

### Simple Demo
```bash
python -m demos.simple_demo
```
- Evaluates 3 test documents
- Shows field-by-field results
- Displays error patterns and statistics

### Comprehensive Demo
```bash
python -m demos.comprehensive_demo
```
- Evaluates 50 test documents
- Rich tables with detailed analysis
- Performance trends and quality assessment

## Testing

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Specific test file
python -m pytest tests/unit/test_field_evaluator.py -v
```

### Test Coverage
- **20 unit tests**: Field evaluator functionality
- **9 integration tests**: Complete pipeline workflow
- **Coverage**: All core components and edge cases

## Key Metrics Summary

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Field Score** | String similarity / Date difference / Exact match | Individual field accuracy |
| **Overall Accuracy** | Weighted average of field scores | Document-level performance |
| **Confidence Correlation** | Pearson correlation (confidence vs accuracy) | Confidence calibration |
| **Success Rate** | Successful fields / Total fields | Field reliability |
| **Impact Score** | Frequency + Importance + Score factors | Error pattern severity |
| **Quality Score** | Accuracy + Correlation + Success rate | Overall quality assessment |

## Best Practices

1. **Use appropriate field types** for accurate scoring
2. **Set realistic thresholds** based on your use case
3. **Monitor confidence correlation** to ensure reliable confidence scores
4. **Analyze error patterns** to identify systematic issues
5. **Track trends over time** to measure improvements
6. **Use field weights** to prioritize important fields

## Troubleshooting

### Common Issues

1. **Low confidence correlation**: Confidence scores don't match actual performance
   - **Solution**: Review confidence calibration in extraction model

2. **High failure rates**: Many fields failing consistently
   - **Solution**: Check field type assignments and normalization

3. **Pattern detection issues**: No patterns detected
   - **Solution**: Ensure sufficient data volume (minimum 2-3 documents)

4. **Strict matching failures**: Exact matches failing
   - **Solution**: Check for hidden characters or encoding issues

### Performance Tips

1. **Batch processing**: Process multiple documents together
2. **Caching**: Reuse evaluator instances
3. **Parallel processing**: Use multiple evaluators for large datasets
4. **Memory management**: Clear statistics periodically for long-running processes

This framework provides comprehensive evaluation capabilities for document extraction quality assessment, enabling data-driven improvements to OCR and prompt-based extraction systems. 