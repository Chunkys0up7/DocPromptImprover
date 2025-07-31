# Metrics Quick Reference Card

## Field-Level Metrics

### Field Evaluation Score
- **Range**: 0.0 to 1.0
- **Text Fields**: String similarity using SequenceMatcher
- **Number Fields**: Exact match or percentage difference
- **Date Fields**: Date difference normalized to 0-1 scale
- **Email Fields**: Exact match or domain similarity
- **Phone Fields**: Digit sequence comparison

### Field Status
- **SUCCESS**: Score ≥ 0.8 AND confidence ≥ 0.7
- **PARTIAL**: Score ≥ 0.5 but < 0.8
- **FAILED**: Score < 0.5 OR confidence < 0.7
- **MISSING**: Extracted value is None/empty

## Document-Level Metrics

### Overall Accuracy
```
Overall Accuracy = Σ(field_score × field_weight) / Σ(field_weights)
```
- **Range**: 0.0 to 1.0
- **Important fields**: invoice_number, total_amount, vendor_name, invoice_date (higher weight)
- **Standard fields**: All other fields (normal weight)

### Confidence Correlation
```
Confidence Correlation = Pearson correlation(confidence_scores, evaluation_scores)
```
- **Range**: 0.0 to 1.0
- **Special case**: Returns 1.0 when all scores are perfect (1.0) and confidence ≥ 0.8
- **Purpose**: Measures how well confidence predicts actual performance

### Field Success Rate
```
Success Rate = (Successful fields / Total fields) × 100
```
- **Range**: 0% to 100%
- **Usage**: Identifies problematic fields

## Error Pattern Metrics

### Impact Score
```
Impact Score = (Frequency Factor × 0.4) + (Importance Factor × 0.4) + (Score Factor × 0.2)

Where:
- Frequency Factor = min(occurrences / 10, 1.0)
- Importance Factor = important_fields_count / total_fields
- Score Factor = 1.0 - average_evaluation_score
```
- **Range**: 0.0 to 1.0
- **Purpose**: Measures severity of error patterns

### Pattern Types
1. **Field-Specific Failures**: Fields consistently failing
2. **Error Message Patterns**: Common error messages
3. **Document Type Failures**: Patterns by document type
4. **Confidence Patterns**: Low/high confidence failures
5. **Value Pattern Failures**: Similar value issues

## Quality Assessment

### Quality Levels
- **Excellent**: ≥0.9 accuracy
- **Good**: 0.8-0.9 accuracy
- **Acceptable**: 0.7-0.8 accuracy
- **Needs Improvement**: 0.6-0.7 accuracy
- **Poor**: <0.6 accuracy

### Quality Score
```
Quality Score = (Overall Accuracy × 0.4) + (Confidence Correlation × 0.3) + (Success Rate × 0.3)
```
- **Range**: 0.0 to 1.0
- **Purpose**: Overall quality assessment

## Statistics Summary

### Basic Counts
- **Total Documents**: Number of documents processed
- **Total Fields**: Number of fields evaluated
- **Successful Extractions**: Fields with SUCCESS status
- **Failed Extractions**: Fields with FAILED status
- **Partial Extractions**: Fields with PARTIAL status
- **Missing Extractions**: Fields with MISSING status

### Performance Metrics
- **Average Accuracy**: Mean accuracy across all documents
- **Average Confidence**: Mean confidence score
- **Success Rate**: Percentage of successful extractions
- **Failure Rate**: Percentage of failed extractions

### Distribution Metrics
- **Field Success Rates**: Success rate per field
- **Document Type Performance**: Average accuracy per document type
- **Confidence Distribution**: Distribution of confidence scores
- **Error Distribution**: Frequency of error types

## Configuration Thresholds

### Default Values
- **Confidence Threshold**: 0.7 (minimum confidence for success)
- **Success Threshold**: 0.8 (minimum score for success)
- **Partial Threshold**: 0.5 (minimum score for partial)
- **Enable Partial Credit**: True
- **Strict Matching**: False
- **Case Sensitive**: False
- **Normalize Whitespace**: True

### Field Weights
- **Important Fields**: 2.0x weight
  - invoice_number, total_amount, vendor_name, invoice_date
- **Standard Fields**: 1.0x weight
  - All other fields

## Quick Commands

### Run Demos
```bash
# Simple demo (3 documents)
python -m demos.simple_demo

# Comprehensive demo (50 documents)
python -m demos.comprehensive_demo
```

### Run Tests
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v
```

### Basic Usage
```python
# Field evaluation
result = evaluator.evaluate_field(
    field_name="vendor_name",
    expected_value="Acme Corporation",
    extracted_value="Acme Corp",
    confidence_score=0.85,
    field_type="text"
)

# Document aggregation
doc_result = aggregator.aggregate_evaluations(
    field_evaluations=field_results,
    document_id="INV-001",
    document_type="invoice",
    confidence_scores=confidence_scores
)

# Error pattern detection
patterns = detector.detect_patterns(document_results)
```

## Interpretation Guide

### Good Performance Indicators
- **Overall Accuracy**: >0.8
- **Confidence Correlation**: >0.7
- **Success Rate**: >80%
- **Quality Score**: >0.7

### Warning Signs
- **Low Confidence Correlation**: <0.5 (confidence scores unreliable)
- **High Failure Rate**: >50% (systematic issues)
- **Low Quality Score**: <0.5 (overall poor performance)
- **Field-Specific Failures**: >90% failure rate for specific fields

### Action Items
1. **Low confidence correlation**: Review confidence calibration
2. **High failure rates**: Check field type assignments
3. **Field-specific failures**: Improve field extraction logic
4. **Document type failures**: Optimize for specific document types
5. **Value pattern failures**: Address systematic formatting issues 