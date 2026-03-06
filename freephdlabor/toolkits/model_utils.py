"""
Model utilities for tools.

After the LangGraph migration, the agent-level model is a ChatLiteLLM instance.
Tools that need to make direct LLM calls use litellm.completion() with a model_id
string extracted from whatever model object is passed in.
"""


def get_raw_model(model):
    """
    Extract a model identifier string (or a raw callable) from whatever
    model object is passed.

    Pre-migration: received a smolagents LiteLLMModel; returned it as-is.
    Post-migration: receives a langchain ChatLiteLLM; returns the model_id
    string so tool internals can call litellm.completion(model=...) directly.

    Handles BudgetedLiteLLMModel wrappers by unwrapping one level.

    Args:
        model: ChatLiteLLM instance, BudgetedLiteLLMModel, a string model_id, or None.

    Returns:
        model_id string, or None if no model provided.
    """
    if model is None:
        return None

    if isinstance(model, str):
        return model

    # langchain_community ChatLiteLLM stores model name in .model attribute
    if hasattr(model, "model") and isinstance(model.model, str):
        return model.model

    # BudgetedLiteLLMModel: self.model is the wrapped ChatLiteLLM
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "model") and isinstance(inner.model, str):
        return inner.model

    # Fallback: return as-is (may be a legacy LiteLLMModel; tools handle it)
    return model
