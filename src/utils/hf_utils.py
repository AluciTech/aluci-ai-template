from typing import Any, Type
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from decorators.error_handler import catch_errors
from utils.log_utils import log
from hydra.utils import get_class


@catch_errors()
def get_huggingface_model(
    model_id: str,
    tokenizer_class: str | Type[PreTrainedTokenizerBase] = AutoTokenizer,
    model_class: str | Type[PreTrainedModel] = AutoModel,
    **hf_kwargs: Any,
):
    """
    Retrieves a model and its tokenizer from the Hugging Face Hub.

    Args:
        model_id (str): The name of the model to load (e.g., 'distilbert-base-uncased').
        tokenizer_class: The tokenizer class to use. Defaults to AutoTokenizer.
        model_class: The model class to use (e.g., AutoModelForSequenceClassification).
                     Defaults to AutoModel.
        hf_kwargs: Additional parameters.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.

    Raises:
        OSError: If the model name is not found on the Hugging Face Hub.
    """
    if isinstance(tokenizer_class, str):
        tokenizer_class = get_class(tokenizer_class)
    if isinstance(model_class, str):
        model_class = get_class(model_class)

    log(message=f"Loading tokenizer for '{model_id}'...", level="INFO")
    tok_or_proc = tokenizer_class.from_pretrained(model_id, **hf_kwargs)

    log(message=f"Loading model '{model_id}'...", level="INFO")
    model = model_class.from_pretrained(model_id, **hf_kwargs)

    log(message="Model and tokenizer loaded successfully!", level="SUCCESS")
    return tok_or_proc, model
