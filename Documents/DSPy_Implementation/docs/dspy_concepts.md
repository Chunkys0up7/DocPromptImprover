# DSPy Concepts for Document Evaluation

## Overview

DSPy (Declarative Self-Improving Python) is a framework for building AI-powered applications with automatic prompt optimization. This document explains key DSPy concepts and how they apply to document extraction evaluation.

## Core DSPy Concepts

### 1. Signatures

**What are Signatures?**
Signatures define the input/output behavior of AI modules. They specify what inputs the AI expects and what outputs it should produce.

**Example for Document Evaluation:**
```python
class FieldEvaluationSignature(dspy.Signature):
    field_name = dspy.InputField(desc="Name of the field being evaluated")
    expected_value = dspy.InputField(desc="Ground truth value")
    extracted_value = dspy.InputField(desc="Value extracted by the model")
    
    evaluation_score = dspy.OutputField(desc="Evaluation score between 0.0 and 1.0")
    status = dspy.OutputField(desc="Extraction status: success, partial, failed, or missing")
```

**Benefits:**
- **Type Safety**: Ensures consistent input/output structure
- **Documentation**: Self-documenting code with clear descriptions
- **Validation**: Automatic validation of inputs and outputs
- **Optimization**: DSPy can optimize prompts based on signature definitions

### 2. Modules

**What are Modules?**
Modules are the building blocks of DSPy applications. They use signatures to define how AI should process inputs and generate outputs.

**Example:**
```python
class DSPyFieldEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.field_evaluator = dspy.ChainOfThought(FieldEvaluationSignature)
    
    def forward(self, field_name, expected_value, extracted_value, ...):
        return self.field_evaluator(...)
```

**Types of Modules:**
- **ChainOfThought**: Step-by-step reasoning
- **ReAct**: Reasoning and action
- **BootstrapFewShot**: Few-shot learning
- **Custom Modules**: User-defined logic

### 3. Optimizers

**What are Optimizers?**
Optimizers automatically improve prompts and module behavior based on training data and feedback.

**MIPROv2 Optimizer:**
```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    num_candidates=5,
    init_temperature=0.5,
    max_bootstrapped_demos=8
)

compiled_program = optimizer.compile(program, trainset, valset)
```

**How it Works:**
1. **Training Data**: Provide examples of inputs and expected outputs
2. **Optimization**: DSPy automatically generates and tests different prompts
3. **Selection**: Best performing prompts are selected
4. **Compilation**: Final optimized program is created

### 4. Language Models

**Supported Providers:**
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude-3 Sonnet, Claude-3 Haiku
- **Cohere**: Command, Command-R
- **Local Models**: Ollama, LM Studio

**Configuration:**
```python
import dspy

# OpenAI
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key="your-key")

# Anthropic
lm = dspy.Anthropic(model="claude-3-sonnet-20240229", api_key="your-key")

# Configure DSPy
dspy.settings.configure(lm=lm)
```

## DSPy in Document Evaluation

### 1. Field-Level Evaluation

**Traditional Approach:**
```python
def evaluate_field(expected, extracted):
    if expected == extracted:
        return 1.0
    elif expected.lower() == extracted.lower():
        return 0.9
    else:
        return 0.0
```

**DSPy Approach:**
```python
class FieldEvaluationSignature(dspy.Signature):
    # Inputs
    field_name = dspy.InputField(desc="Name of the field")
    expected_value = dspy.InputField(desc="Ground truth value")
    extracted_value = dspy.InputField(desc="Extracted value")
    field_type = dspy.InputField(desc="Field type: text, number, date, email, phone")
    
    # Outputs
    evaluation_score = dspy.OutputField(desc="Score between 0.0 and 1.0")
    status = dspy.OutputField(desc="Status: success, partial, failed, missing")
    error_message = dspy.OutputField(desc="Error description if failed")
    evaluation_notes = dspy.OutputField(desc="Detailed evaluation reasoning")
```

**Benefits:**
- **Context Awareness**: AI understands field context and relationships
- **Intelligent Scoring**: Considers field type, format variations, and business rules
- **Detailed Feedback**: Provides explanations for evaluation decisions
- **Adaptive Learning**: Improves over time with more examples

### 2. Document-Level Aggregation

**Traditional Approach:**
```python
def aggregate_document(field_evaluations):
    scores = [eval.evaluation_score for eval in field_evaluations]
    return sum(scores) / len(scores)
```

**DSPy Approach:**
```python
class DocumentAggregationSignature(dspy.Signature):
    field_evaluations = dspy.InputField(desc="List of field evaluation results")
    document_type = dspy.InputField(desc="Type of document")
    confidence_scores = dspy.InputField(desc="Confidence scores for each field")
    
    overall_accuracy = dspy.OutputField(desc="Overall document accuracy")
    confidence_correlation = dspy.OutputField(desc="Correlation between confidence and accuracy")
    quality_assessment = dspy.OutputField(desc="Overall quality: excellent, good, fair, poor")
    critical_errors = dspy.OutputField(desc="List of critical errors")
    improvement_suggestions = dspy.OutputField(desc="Specific improvement suggestions")
```

**Benefits:**
- **Intelligent Aggregation**: Considers field importance and relationships
- **Quality Assessment**: Provides holistic quality evaluation
- **Error Analysis**: Identifies critical issues and their impact
- **Actionable Insights**: Suggests specific improvements

### 3. Pattern Detection

**Traditional Approach:**
```python
def detect_patterns(evaluations):
    patterns = {}
    for eval in evaluations:
        if eval.status == "failed":
            error_type = classify_error(eval.error_message)
            patterns[error_type] = patterns.get(error_type, 0) + 1
    return patterns
```

**DSPy Approach:**
```python
class FailurePatternAnalysisSignature(dspy.Signature):
    evaluation_results = dspy.InputField(desc="Multiple evaluation results")
    document_types = dspy.InputField(desc="Document types in the dataset")
    time_period = dspy.InputField(desc="Time period covered")
    
    common_patterns = dspy.OutputField(desc="List of common failure patterns")
    pattern_frequency = dspy.OutputField(desc="Frequency of each pattern")
    pattern_severity = dspy.OutputField(desc="Severity assessment")
    root_causes = dspy.OutputField(desc="Root causes for each pattern")
    suggested_fixes = dspy.OutputField(desc="Specific fixes for each pattern")
```

**Benefits:**
- **Intelligent Pattern Recognition**: Identifies complex patterns and relationships
- **Root Cause Analysis**: Understands underlying causes of failures
- **Severity Assessment**: Prioritizes issues by impact
- **Actionable Fixes**: Provides specific solutions for each pattern

### 4. Prompt Optimization

**Traditional Approach:**
```python
# Manual prompt engineering
prompt = """
Extract the following fields from this document:
- vendor_name: The name of the vendor
- total_amount: The total amount
- invoice_date: The invoice date
"""
```

**DSPy Approach:**
```python
class PromptOptimizationSignature(dspy.Signature):
    current_prompt = dspy.InputField(desc="Current prompt")
    evaluation_statistics = dspy.InputField(desc="Evaluation performance data")
    failure_patterns = dspy.InputField(desc="Identified failure patterns")
    target_improvement = dspy.InputField(desc="Target improvement percentage")
    
    optimized_prompt = dspy.OutputField(desc="Improved prompt")
    improvement_rationale = dspy.OutputField(desc="Explanation of improvements")
    expected_improvement = dspy.OutputField(desc="Expected improvement percentage")
    confidence_in_improvement = dspy.OutputField(desc="Confidence in improvement")
```

**Benefits:**
- **Automatic Optimization**: Continuously improves prompts based on performance
- **Data-Driven**: Uses actual evaluation results to guide improvements
- **Targeted Improvements**: Addresses specific failure patterns
- **Performance Prediction**: Estimates expected improvements

## Hybrid Approach

### Combining Statistical and AI Evaluation

**Best of Both Worlds:**
```python
class HybridEvaluator:
    def evaluate_field(self, field_name, expected, extracted, confidence):
        # Always use statistical evaluation
        statistical_result = self.statistical_evaluator.evaluate(...)
        
        # Use AI evaluation if confidence is high
        if confidence >= self.ai_threshold:
            ai_result = self.ai_evaluator.evaluate(...)
            
            # Combine results
            combined_score = (statistical_result.score * 0.6 + 
                            ai_result.score * 0.4)
            
            return {
                "score": combined_score,
                "method": "hybrid",
                "statistical_score": statistical_result.score,
                "ai_score": ai_result.score
            }
        
        return {
            "score": statistical_result.score,
            "method": "statistical"
        }
```

**Benefits:**
- **Reliability**: Statistical evaluation as fallback
- **Intelligence**: AI evaluation for complex cases
- **Cost Efficiency**: Only use AI when beneficial
- **Transparency**: Clear indication of evaluation method

## Key Advantages of DSPy

### 1. Declarative Programming
- **Clear Intent**: Focus on what you want, not how to do it
- **Maintainable**: Easy to understand and modify
- **Composable**: Build complex systems from simple components

### 2. Automatic Optimization
- **Continuous Improvement**: System gets better over time
- **Data-Driven**: Optimizations based on actual performance
- **Adaptive**: Responds to changing requirements and data

### 3. Type Safety
- **Validation**: Automatic input/output validation
- **Documentation**: Self-documenting code
- **Debugging**: Clear error messages and traceability

### 4. Flexibility
- **Multiple Providers**: Support for various LLM providers
- **Custom Logic**: Easy to add custom processing
- **Integration**: Seamless integration with existing systems

## Best Practices

### 1. Signature Design
- **Clear Descriptions**: Provide detailed descriptions for all fields
- **Appropriate Types**: Use correct field types and constraints
- **Comprehensive Outputs**: Include all necessary information in outputs

### 2. Module Organization
- **Single Responsibility**: Each module should have one clear purpose
- **Composability**: Design modules to work together
- **Error Handling**: Include proper error handling and fallbacks

### 3. Optimization Strategy
- **Quality Data**: Use high-quality training data
- **Regular Updates**: Periodically retrain and optimize
- **Performance Monitoring**: Track optimization effectiveness

### 4. Cost Management
- **Budget Limits**: Set appropriate budget constraints
- **Efficient Usage**: Use AI evaluation strategically
- **Monitoring**: Track costs and usage patterns

## Conclusion

DSPy provides a powerful framework for building intelligent document evaluation systems. By combining declarative programming with automatic optimization, it enables the creation of systems that are both intelligent and maintainable. The hybrid approach ensures reliability while leveraging AI capabilities for complex evaluation tasks. 