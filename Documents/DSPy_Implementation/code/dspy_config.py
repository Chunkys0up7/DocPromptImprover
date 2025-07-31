"""
DSPy Configuration for Document Extraction Evaluation

This module handles DSPy configuration, language model setup, and optimization settings
for the AI-powered document evaluation system.
"""

import dspy
import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class DSPyConfig(BaseSettings):
    """Configuration settings for DSPy integration."""
    
    # Language Model Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    
    # Provider Selection
    default_provider: str = Field("openai", env="DSPY_DEFAULT_PROVIDER")
    default_model: str = Field("gpt-3.5-turbo", env="DSPY_DEFAULT_MODEL")
    
    # Model-specific configurations
    openai_model: str = Field("gpt-3.5-turbo", env="DSPY_OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-sonnet-20240229", env="DSPY_ANTHROPIC_MODEL")
    cohere_model: str = Field("command", env="DSPY_COHERE_MODEL")
    
    # Optimization Configuration
    optimization_enabled: bool = Field(True, env="DSPY_OPTIMIZATION_ENABLED")
    miprov2_num_candidates: int = Field(5, env="DSPY_MIPROV2_NUM_CANDIDATES")
    miprov2_init_temperature: float = Field(0.5, env="DSPY_MIPROV2_INIT_TEMPERATURE")
    optimization_interval: int = Field(100, env="DSPY_OPTIMIZATION_INTERVAL")
    
    # Evaluation Configuration
    ai_evaluation_enabled: bool = Field(True, env="DSPY_AI_EVALUATION_ENABLED")
    ai_confidence_threshold: float = Field(0.8, env="DSPY_AI_CONFIDENCE_THRESHOLD")
    hybrid_evaluation_enabled: bool = Field(True, env="DSPY_HYBRID_EVALUATION_ENABLED")
    
    # Performance Configuration
    max_tokens: int = Field(1000, env="DSPY_MAX_TOKENS")
    temperature: float = Field(0.1, env="DSPY_TEMPERATURE")
    timeout: int = Field(30, env="DSPY_TIMEOUT")
    
    # Cost Management
    cost_tracking_enabled: bool = Field(True, env="DSPY_COST_TRACKING_ENABLED")
    max_cost_per_evaluation: float = Field(0.01, env="DSPY_MAX_COST_PER_EVALUATION")
    budget_limit: float = Field(10.00, env="DSPY_BUDGET_LIMIT")
    
    # Development
    debug_mode: bool = Field(False, env="DSPY_DEBUG_MODE")
    mock_responses: bool = Field(False, env="DSPY_MOCK_RESPONSES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DSPyLanguageModelManager:
    """Manages language model configuration and initialization."""
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.lm = None
        self.provider = None
    
    def initialize_language_model(self) -> dspy.LM:
        """Initialize the language model based on configuration."""
        
        if self.config.default_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            
            self.lm = dspy.OpenAI(
                model=self.config.openai_model,
                api_key=self.config.openai_api_key,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            self.provider = "openai"
            
        elif self.config.default_provider == "anthropic":
            if not self.config.anthropic_api_key:
                raise ValueError("Anthropic API key is required for Anthropic provider")
            
            self.lm = dspy.Anthropic(
                model=self.config.anthropic_model,
                api_key=self.config.anthropic_api_key,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            self.provider = "anthropic"
            
        elif self.config.default_provider == "cohere":
            if not self.config.cohere_api_key:
                raise ValueError("Cohere API key is required for Cohere provider")
            
            self.lm = dspy.Cohere(
                model=self.config.cohere_model,
                api_key=self.config.cohere_api_key,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            self.provider = "cohere"
            
        else:
            raise ValueError(f"Unsupported provider: {self.config.default_provider}")
        
        # Configure DSPy settings
        dspy.settings.configure(lm=self.lm)
        
        return self.lm
    
    def get_language_model(self) -> dspy.LM:
        """Get the current language model instance."""
        if self.lm is None:
            self.initialize_language_model()
        return self.lm
    
    def switch_provider(self, provider: str) -> dspy.LM:
        """Switch to a different language model provider."""
        self.config.default_provider = provider
        return self.initialize_language_model()


class DSPyOptimizationManager:
    """Manages DSPy optimization settings and configurations."""
    
    def __init__(self, config: DSPyConfig):
        self.config = config
    
    def get_miprov2_config(self) -> Dict[str, Any]:
        """Get MIPROv2 optimizer configuration."""
        return {
            "num_candidates": self.config.miprov2_num_candidates,
            "init_temperature": self.config.miprov2_init_temperature,
            "max_bootstrapped_demos": 8,
            "max_labeled_demos": 8,
            "num_threads": 4
        }
    
    def create_miprov2_optimizer(self):
        """Create a MIPROv2 optimizer instance."""
        from dspy.teleprompt import MIPROv2
        
        config = self.get_miprov2_config()
        return MIPROv2(**config)
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get general optimization settings."""
        return {
            "enabled": self.config.optimization_enabled,
            "interval": self.config.optimization_interval,
            "max_cost_per_evaluation": self.config.max_cost_per_evaluation,
            "budget_limit": self.config.budget_limit
        }


class DSPyCostTracker:
    """Tracks and manages costs for DSPy operations."""
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.total_cost = 0.0
        self.evaluation_count = 0
        self.cost_history = []
    
    def track_evaluation_cost(self, cost: float):
        """Track the cost of an evaluation."""
        if not self.config.cost_tracking_enabled:
            return
        
        self.total_cost += cost
        self.evaluation_count += 1
        self.cost_history.append(cost)
        
        # Check budget limit
        if self.total_cost > self.config.budget_limit:
            raise ValueError(f"Budget limit exceeded: ${self.total_cost:.2f}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary information."""
        return {
            "total_cost": self.total_cost,
            "evaluation_count": self.evaluation_count,
            "average_cost_per_evaluation": self.total_cost / self.evaluation_count if self.evaluation_count > 0 else 0.0,
            "budget_remaining": self.config.budget_limit - self.total_cost,
            "cost_history": self.cost_history[-100:]  # Last 100 evaluations
        }
    
    def reset_costs(self):
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.evaluation_count = 0
        self.cost_history = []


class DSPyManager:
    """Main manager class for DSPy integration."""
    
    def __init__(self, config: Optional[DSPyConfig] = None):
        self.config = config or DSPyConfig()
        self.lm_manager = DSPyLanguageModelManager(self.config)
        self.optimization_manager = DSPyOptimizationManager(self.config)
        self.cost_tracker = DSPyCostTracker(self.config)
        self.initialized = False
    
    def initialize(self):
        """Initialize DSPy with all components."""
        try:
            # Initialize language model
            self.lm_manager.initialize_language_model()
            
            # Test the configuration
            self._test_configuration()
            
            self.initialized = True
            print(f"DSPy initialized successfully with {self.lm_manager.provider} provider")
            
        except Exception as e:
            print(f"Failed to initialize DSPy: {str(e)}")
            raise
    
    def _test_configuration(self):
        """Test the DSPy configuration with a simple evaluation."""
        if self.config.mock_responses:
            return
        
        try:
            # Simple test to verify configuration
            test_signature = dspy.Signature(
                "test_input = dspy.InputField(desc='Test input')",
                "test_output = dspy.OutputField(desc='Test output')"
            )
            
            test_module = dspy.ChainOfThought(test_signature)
            result = test_module(test_input="Hello")
            
            # Track minimal cost for test
            self.cost_tracker.track_evaluation_cost(0.001)
            
        except Exception as e:
            print(f"Configuration test failed: {str(e)}")
            raise
    
    def get_language_model(self) -> dspy.LM:
        """Get the current language model."""
        if not self.initialized:
            self.initialize()
        return self.lm_manager.get_language_model()
    
    def get_optimizer(self):
        """Get the MIPROv2 optimizer."""
        return self.optimization_manager.create_miprov2_optimizer()
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return self.cost_tracker.get_cost_summary()
    
    def is_initialized(self) -> bool:
        """Check if DSPy is initialized."""
        return self.initialized
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "provider": self.lm_manager.provider,
            "model": self.config.default_model,
            "optimization_enabled": self.config.optimization_enabled,
            "ai_evaluation_enabled": self.config.ai_evaluation_enabled,
            "hybrid_evaluation_enabled": self.config.hybrid_evaluation_enabled,
            "cost_tracking_enabled": self.config.cost_tracking_enabled,
            "debug_mode": self.config.debug_mode
        }


# Global DSPy manager instance
_dspy_manager: Optional[DSPyManager] = None


def get_dspy_manager() -> DSPyManager:
    """Get the global DSPy manager instance."""
    global _dspy_manager
    if _dspy_manager is None:
        _dspy_manager = DSPyManager()
    return _dspy_manager


def initialize_dspy(config: Optional[DSPyConfig] = None) -> DSPyManager:
    """Initialize DSPy with the given configuration."""
    global _dspy_manager
    _dspy_manager = DSPyManager(config)
    _dspy_manager.initialize()
    return _dspy_manager


def configure_dspy():
    """Configure DSPy with default settings."""
    return initialize_dspy()


def get_language_model() -> dspy.LM:
    """Get the current language model."""
    manager = get_dspy_manager()
    return manager.get_language_model()


def get_optimizer():
    """Get the MIPROv2 optimizer."""
    manager = get_dspy_manager()
    return manager.get_optimizer()


def get_cost_summary() -> Dict[str, Any]:
    """Get cost summary."""
    manager = get_dspy_manager()
    return manager.get_cost_summary()


def reset_costs():
    """Reset cost tracking."""
    manager = get_dspy_manager()
    manager.cost_tracker.reset_costs()


def is_dspy_available() -> bool:
    """Check if DSPy is available and configured."""
    try:
        manager = get_dspy_manager()
        return manager.is_initialized()
    except:
        return False 