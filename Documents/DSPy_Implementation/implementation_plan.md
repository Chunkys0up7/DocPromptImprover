# Detailed DSPy Implementation Plan

## Phase 1: Core DSPy Integration

### Step 1.1: Environment Setup
```bash
# Install DSPy and dependencies
pip install dspy-ai openai anthropic cohere
pip install pydantic-settings python-dotenv
```

### Step 1.2: Configuration Setup
Create `src/dspy/dspy_config.py`:
```python
import dspy
from typing import Optional
from pydantic_settings import BaseSettings

class DSPyConfig(BaseSettings):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    default_provider: str = "openai"
    default_model: str = "gpt-3.5-turbo"
    
    class Config:
        env_file = ".env"

def configure_dspy():
    config = DSPyConfig()
    
    if config.default_provider == "openai":
        lm = dspy.OpenAI(model=config.default_model, api_key=config.openai_api_key)
    elif config.default_provider == "anthropic":
        lm = dspy.Anthropic(model="claude-3-sonnet-20240229", api_key=config.anthropic_api_key)
    elif config.default_provider == "cohere":
        lm = dspy.Cohere(model="command", api_key=config.cohere_api_key)
    
    dspy.settings.configure(lm=lm)
    return lm
```

### Step 1.3: Basic Signature Definition
Create `src/dspy/dspy_signatures.py`:
```python
import dspy
from typing import Optional

class FieldEvaluationSignature(dspy.Signature):
    """Evaluate a single field extraction."""
    
    field_name = dspy.InputField(desc="Name of the field being evaluated")
    expected_value = dspy.InputField(desc="Ground truth value")
    extracted_value = dspy.InputField(desc="Value extracted by the model")
    confidence_score = dspy.InputField(desc="Confidence score from extraction")
    field_type = dspy.InputField(desc="Expected data type of the field")
    
    evaluation_score = dspy.OutputField(desc="Evaluation score between 0.0 and 1.0")
    status = dspy.OutputField(desc="Extraction status: success, partial, failed, or missing")
    error_message = dspy.OutputField(desc="Error message if evaluation failed")
    evaluation_notes = dspy.OutputField(desc="Detailed evaluation notes")

class DocumentAggregationSignature(dspy.Signature):
    """Aggregate field evaluations into document-level metrics."""
    
    field_evaluations = dspy.InputField(desc="JSON string of field evaluation results")
    document_type = dspy.InputField(desc="Type of document being evaluated")
    confidence_scores = dspy.InputField(desc="JSON string of confidence scores")
    
    overall_accuracy = dspy.OutputField(desc="Overall document accuracy score")
    confidence_correlation = dspy.OutputField(desc="Correlation between confidence and accuracy")
    quality_assessment = dspy.OutputField(desc="Overall quality assessment")
```

## Phase 2: Advanced Evaluation Signatures

### Step 2.1: Error Pattern Analysis
```python
class FailurePatternAnalysisSignature(dspy.Signature):
    """Analyze failure patterns across multiple evaluations."""
    
    evaluation_results = dspy.InputField(desc="JSON string of evaluation results")
    document_types = dspy.InputField(desc="List of document types")
    
    common_patterns = dspy.OutputField(desc="List of common failure patterns")
    pattern_frequency = dspy.OutputField(desc="Frequency of each pattern")
    suggested_fixes = dspy.OutputField(desc="Suggested fixes for each pattern")

class ConfidenceCalibrationSignature(dspy.Signature):
    """Calibrate confidence scores based on actual accuracy."""
    
    confidence_scores = dspy.InputField(desc="List of confidence scores")
    actual_accuracy = dspy.InputField(desc="List of actual accuracy scores")
    
    calibration_factor = dspy.OutputField(desc="Calibration factor to apply")
    calibrated_scores = dspy.OutputField(desc="List of calibrated confidence scores")
    calibration_quality = dspy.OutputField(desc="Quality of calibration")
```

### Step 2.2: Prompt Optimization
```python
class PromptOptimizationSignature(dspy.Signature):
    """Generate optimized prompts based on evaluation results."""
    
    current_prompt = dspy.InputField(desc="Current prompt being used")
    evaluation_statistics = dspy.InputField(desc="JSON string of evaluation statistics")
    failure_patterns = dspy.InputField(desc="JSON string of failure patterns")
    target_improvement = dspy.InputField(desc="Target improvement percentage")
    
    optimized_prompt = dspy.OutputField(desc="Optimized version of the prompt")
    improvement_rationale = dspy.OutputField(desc="Explanation of improvements made")
    expected_improvement = dspy.OutputField(desc="Expected improvement percentage")
    confidence_in_improvement = dspy.OutputField(desc="Confidence in the improvement")
```

## Phase 3: DSPy Module Implementation

### Step 3.1: Field Evaluator Module
Create `src/dspy/dspy_modules.py`:
```python
import dspy
from typing import Dict, Any, Optional
from .dspy_signatures import FieldEvaluationSignature, DocumentAggregationSignature

class DSPyFieldEvaluator(dspy.Module):
    """AI-powered field evaluator using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.field_evaluator = dspy.ChainOfThought(FieldEvaluationSignature)
    
    def forward(self, field_name: str, expected_value: str, extracted_value: str, 
                confidence_score: float, field_type: str) -> Dict[str, Any]:
        """Evaluate a single field using AI."""
        
        result = self.field_evaluator(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            field_type=field_type
        )
        
        return {
            "evaluation_score": float(result.evaluation_score),
            "status": result.status.lower(),
            "error_message": result.error_message,
            "evaluation_notes": result.evaluation_notes
        }

class DSPyDocumentAggregator(dspy.Module):
    """AI-powered document aggregator using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.aggregator = dspy.ChainOfThought(DocumentAggregationSignature)
    
    def forward(self, field_evaluations: list, document_type: str, 
                confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate field evaluations using AI."""
        
        result = self.aggregator(
            field_evaluations=str(field_evaluations),
            document_type=document_type,
            confidence_scores=str(confidence_scores)
        )
        
        return {
            "overall_accuracy": float(result.overall_accuracy),
            "confidence_correlation": float(result.confidence_correlation),
            "quality_assessment": result.quality_assessment
        }
```

### Step 3.2: Optimization Module
```python
class DSPyOptimizer(dspy.Module):
    """AI-powered prompt optimizer using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.optimizer = dspy.ChainOfThought(PromptOptimizationSignature)
        self.pattern_analyzer = dspy.ChainOfThought(FailurePatternAnalysisSignature)
    
    def forward(self, current_prompt: str, evaluation_results: list, 
                target_improvement: float = 0.1) -> Dict[str, Any]:
        """Generate optimized prompt."""
        
        # Analyze failure patterns
        pattern_result = self.pattern_analyzer(
            evaluation_results=str(evaluation_results),
            document_types=list(set([r["document_type"] for r in evaluation_results]))
        )
        
        # Generate optimized prompt
        optimization_result = self.optimizer(
            current_prompt=current_prompt,
            evaluation_statistics=str(evaluation_results),
            failure_patterns=str(pattern_result.common_patterns),
            target_improvement=target_improvement
        )
        
        return {
            "optimized_prompt": optimization_result.optimized_prompt,
            "improvement_rationale": optimization_result.improvement_rationale,
            "expected_improvement": float(optimization_result.expected_improvement),
            "confidence_in_improvement": float(optimization_result.confidence_in_improvement),
            "failure_patterns": pattern_result.common_patterns
        }
```

## Phase 4: MIPROv2 Integration

### Step 4.1: Optimizer Configuration
Create `src/dspy/dspy_optimizers.py`:
```python
import dspy
from dspy.teleprompt import MIPROv2
from typing import List, Dict, Any

class DocumentExtractionOptimizer:
    """MIPROv2-based optimizer for document extraction prompts."""
    
    def __init__(self, num_candidates: int = 5, init_temperature: float = 0.5):
        self.optimizer = MIPROv2(
            num_candidates=num_candidates,
            init_temperature=init_temperature
        )
        self.evaluator = DSPyFieldEvaluator()
    
    def optimize_prompt(self, training_data: List[Dict[str, Any]], 
                       current_prompt: str) -> str:
        """Optimize prompt using MIPROv2."""
        
        # Define the program to optimize
        class DocumentExtractionProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.evaluator = DSPyFieldEvaluator()
            
            def forward(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
                # Extract fields using the prompt
                extracted_fields = self.extract_fields(document_data, current_prompt)
                
                # Evaluate each field
                evaluations = []
                for field_name, expected_value in document_data["ground_truth"].items():
                    extracted_value = extracted_fields.get(field_name)
                    confidence = document_data["confidence_scores"].get(field_name, 0.5)
                    
                    evaluation = self.evaluator(
                        field_name=field_name,
                        expected_value=expected_value,
                        extracted_value=extracted_value,
                        confidence_score=confidence,
                        field_type=self.determine_field_type(field_name)
                    )
                    evaluations.append(evaluation)
                
                return {"evaluations": evaluations}
        
        # Compile and optimize
        compiled_program = self.optimizer.compile(
            DocumentExtractionProgram(),
            trainset=training_data,
            valset=training_data[:10]  # Use subset for validation
        )
        
        return compiled_program
```

## Phase 5: Integration with Existing System

### Step 5.1: Hybrid Evaluator
Create `src/dspy/dspy_integration.py`:
```python
from typing import Dict, Any, Optional
from ..evaluators.field_evaluator import FieldEvaluator
from ..evaluators.document_aggregator import DocumentAggregator
from .dspy_modules import DSPyFieldEvaluator, DSPyDocumentAggregator

class HybridEvaluator:
    """Combines statistical and AI-powered evaluation."""
    
    def __init__(self, use_ai: bool = True, ai_confidence_threshold: float = 0.8):
        self.use_ai = use_ai
        self.ai_confidence_threshold = ai_confidence_threshold
        
        # Statistical evaluators
        self.statistical_field_evaluator = FieldEvaluator()
        self.statistical_document_aggregator = DocumentAggregator()
        
        # AI-powered evaluators
        if use_ai:
            self.ai_field_evaluator = DSPyFieldEvaluator()
            self.ai_document_aggregator = DSPyDocumentAggregator()
    
    def evaluate_field(self, field_name: str, expected_value: str, 
                      extracted_value: str, confidence_score: float, 
                      field_type: str) -> Dict[str, Any]:
        """Evaluate field using hybrid approach."""
        
        # Always use statistical evaluation
        statistical_result = self.statistical_field_evaluator.evaluate_field(
            field_name, expected_value, extracted_value, confidence_score, field_type
        )
        
        # Use AI evaluation if enabled and confidence is high
        if self.use_ai and confidence_score >= self.ai_confidence_threshold:
            try:
                ai_result = self.ai_field_evaluator(
                    field_name, expected_value, extracted_value, confidence_score, field_type
                )
                
                # Combine results (weighted average)
                combined_score = (statistical_result.evaluation_score * 0.7 + 
                                ai_result["evaluation_score"] * 0.3)
                
                return {
                    "evaluation_score": combined_score,
                    "status": ai_result["status"],
                    "error_message": ai_result["error_message"],
                    "evaluation_notes": f"Hybrid: {statistical_result.evaluation_notes}; AI: {ai_result['evaluation_notes']}",
                    "method": "hybrid"
                }
            except Exception as e:
                # Fallback to statistical only
                return {
                    "evaluation_score": statistical_result.evaluation_score,
                    "status": statistical_result.status.value,
                    "error_message": statistical_result.error_message,
                    "evaluation_notes": f"{statistical_result.evaluation_notes} (AI failed: {str(e)})",
                    "method": "statistical_fallback"
                }
        
        return {
            "evaluation_score": statistical_result.evaluation_score,
            "status": statistical_result.status.value,
            "error_message": statistical_result.error_message,
            "evaluation_notes": statistical_result.evaluation_notes,
            "method": "statistical"
        }
```

## Implementation Checklist

### Phase 1: Core Setup
- [ ] Install DSPy and dependencies
- [ ] Configure language model provider
- [ ] Create basic signatures
- [ ] Test basic evaluation

### Phase 2: Advanced Features
- [ ] Implement error pattern analysis
- [ ] Add confidence calibration
- [ ] Create prompt optimization signatures
- [ ] Test advanced features

### Phase 3: Module Development
- [ ] Build field evaluator module
- [ ] Build document aggregator module
- [ ] Build optimizer module
- [ ] Test all modules

### Phase 4: Optimization
- [ ] Integrate MIPROv2
- [ ] Configure optimization parameters
- [ ] Test prompt optimization
- [ ] Validate improvements

### Phase 5: Integration
- [ ] Create hybrid evaluator
- [ ] Integrate with existing API
- [ ] Add configuration options
- [ ] Test full integration

## Testing Strategy

1. **Unit Tests**: Test each signature and module individually
2. **Integration Tests**: Test the complete evaluation pipeline
3. **Performance Tests**: Measure AI vs. statistical performance
4. **A/B Tests**: Compare optimized vs. original prompts
5. **Regression Tests**: Ensure existing functionality is preserved

## Deployment Considerations

1. **API Keys**: Secure storage of LLM provider keys
2. **Rate Limiting**: Handle API rate limits gracefully
3. **Fallback Strategy**: Ensure system works without AI
4. **Cost Management**: Monitor and control API usage
5. **Performance Monitoring**: Track evaluation quality and speed 