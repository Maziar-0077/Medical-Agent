# retry_utils.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import logging

logger = logging.getLogger(__name__)

def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error (429)."""
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        return exception.response.status_code == 429
    if hasattr(exception, 'status_code'):
        return exception.status_code == 429
    return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(is_rate_limit_error),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number}/5)"
    )
)
def invoke_with_retry(llm, messages, **kwargs):
    """Invoke an LLM with automatic retry on rate limits."""
    return llm.invoke(messages, **kwargs)