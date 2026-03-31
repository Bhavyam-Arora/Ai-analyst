"""
utils/token_counter.py — Count tokens before making LLM API calls

WHY THIS EXISTS:
OpenAI charges per token and enforces context window limits (e.g., GPT-4o has
a 128k token context window). If we blindly send 200 chunks to the LLM, we'll
either hit the limit and get an error, or waste money on tokens that don't fit.

This utility lets us:
1. Estimate cost before a call
2. Truncate/trim context to stay within limits
3. Log token usage for monitoring

TOKENIZATION:
We use `tiktoken`, OpenAI's own tokenizer library. Different models use
different tokenization schemes — GPT-4o uses "cl100k_base", same as GPT-4.
Always use the model's actual encoder for accurate counts (not word splits).
"""

import logging
import tiktoken

logger = logging.getLogger(__name__)

# ── Encoder cache ──────────────────────────────────────────────────────────────
# Loading an encoder is slightly expensive, so we cache it in a module-level dict.
# This is safe because encoders are stateless and thread-safe.
_encoder_cache: dict[str, tiktoken.Encoding] = {}


def get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    """
    Return (and cache) the tiktoken encoder for a given model.

    Args:
        model: The OpenAI model name (e.g., "gpt-4o", "text-embedding-3-small").

    Returns:
        tiktoken.Encoding instance.
    """
    if model not in _encoder_cache:
        try:
            # tiktoken.encoding_for_model maps model name → correct encoding
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models (covers GPT-4 family)
            logger.warning(
                "No tiktoken encoding found for model '%s'. Falling back to cl100k_base.",
                model,
            )
            _encoder_cache[model] = tiktoken.get_encoding("cl100k_base")

    return _encoder_cache[model]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string for a given model.

    Args:
        text: The string to tokenize.
        model: OpenAI model name used to select the correct tokenizer.

    Returns:
        Integer token count.
    """
    if not text:
        return 0
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def count_tokens_for_messages(messages: list[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens for a list of OpenAI chat messages (role + content dicts).

    WHY: The chat API adds a small overhead per message (role tokens, separator
    tokens). This matches OpenAI's own token counting logic for chat endpoints.

    Args:
        messages: List of {"role": str, "content": str} dicts.
        model: Model name for correct tokenizer.

    Returns:
        Total token count including per-message overhead.
    """
    encoder = get_encoder(model)
    # Per OpenAI docs: each message adds ~4 tokens for role/separator overhead
    tokens_per_message = 4
    total = 0

    for message in messages:
        total += tokens_per_message
        for value in message.values():
            total += len(encoder.encode(str(value)))

    # Add 3 tokens for the reply primer ("assistant" turn start)
    total += 3
    return total


def truncate_to_token_limit(
    text: str, max_tokens: int, model: str = "gpt-4o"
) -> str:
    """
    Truncate a text string to fit within a token limit.

    WHY: When assembling the context window for a prompt, we may need to
    trim the retrieved chunks to ensure the full prompt fits in the model's
    context window. Truncating by tokens (not characters) is accurate.

    Args:
        text: The text to potentially truncate.
        max_tokens: Maximum number of tokens allowed.
        model: Model name for correct tokenizer.

    Returns:
        Original text if within limit, otherwise truncated text.
    """
    encoder = get_encoder(model)
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    logger.warning(
        "Text exceeded token limit (%d > %d). Truncating.", len(tokens), max_tokens
    )
    # Decode back to string — this handles multi-byte characters correctly
    return encoder.decode(tokens[:max_tokens])
