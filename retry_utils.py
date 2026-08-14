# retry_utils.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import logging
import time
from threading import Semaphore, Lock
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============= REQUEST RATE LIMITING =============
class RateLimiter:
    """Token bucket rate limiter to prevent API overload"""

    def __init__(self, max_requests_per_minute=30, max_concurrent=2):
        self.max_requests = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.requests_window = deque()  # Track requests in last 60s
        self.concurrent_lock = Semaphore(max_concurrent)
        self.state_lock = Lock()
        self.last_rate_limit_time = None
        self.cooldown_seconds = 60

    def is_rate_limited(self):
        """Check if we're currently in cooldown"""
        if self.last_rate_limit_time:
            elapsed = time.time() - self.last_rate_limit_time
            if elapsed < self.cooldown_seconds:
                return True
            else:
                self.last_rate_limit_time = None
        return False

    def wait_if_needed(self):
        """Wait if rate limit cooldown is active"""
        if self.last_rate_limit_time:
            elapsed = time.time() - self.last_rate_limit_time
            if elapsed < self.cooldown_seconds:
                wait_time = self.cooldown_seconds - elapsed
                logger.warning(f"🚫 Rate limit cooldown: Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.last_rate_limit_time = None

    def acquire(self):
        """Acquire permission to make a request"""
        self.wait_if_needed()

        with self.state_lock:
            # Remove requests older than 60 seconds
            cutoff = time.time() - 60
            while self.requests_window and self.requests_window[0] < cutoff:
                self.requests_window.popleft()

            # Check if we're under the limit
            if len(self.requests_window) >= self.max_requests:
                oldest = self.requests_window[0]
                wait_time = 60 - (time.time() - oldest)
                if wait_time > 0:
                    logger.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    return self.acquire()  # Retry after waiting

            self.requests_window.append(time.time())

        # Also enforce max concurrent
        self.concurrent_lock.acquire()

    def release(self):
        """Release semaphore after request"""
        self.concurrent_lock.release()

    def mark_rate_limited(self):
        """Mark that we hit a rate limit"""
        self.last_rate_limit_time = time.time()
        logger.error(f"🛑 Rate limit triggered. Entering {self.cooldown_seconds}s cooldown...")


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests_per_minute=25, max_concurrent=2)


# ============= ERROR DETECTION =============
def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error (429)"""
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        return exception.response.status_code == 429
    if hasattr(exception, 'status_code'):
        return exception.status_code == 429
    error_str = str(exception).lower()
    return '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str


# ============= RETRY WITH RATE LIMITING =============
@retry(
    stop=stop_after_attempt(3),  # Reduced from 5 to 3
    wait=wait_exponential(multiplier=3, min=5, max=30),  # Longer waits
    retry=retry_if_exception(is_rate_limit_error),
    before_sleep=lambda retry_state: logger.warning(
        f"⚠️  Rate limit detected. Retrying in {retry_state.next_action.sleep:.1f}s... "
        f"(Attempt {retry_state.attempt_number}/3)"
    )
)
def invoke_with_retry(llm, messages, **kwargs):
    """Invoke LLM with smart rate limiting"""
    rate_limiter.acquire()
    try:
        result = llm.invoke(messages, **kwargs)
        return result
    except Exception as e:
        if is_rate_limit_error(e):
            rate_limiter.mark_rate_limited()
        raise
    finally:
        rate_limiter.release()


# ============= FALLBACK RESPONSES =============
class FallbackResponse:
    """Simulated LLM response for when API fails"""

    def __init__(self, content):
        self.content = content


def get_derailment_fallback():
    """Safe default for derailment detection"""
    return FallbackResponse(
        '{"is_derailed": false, "stabilization_message": ""}'
    )


def get_extraction_fallback():
    """Safe default for data extraction"""
    return FallbackResponse(
        '{"name": null, "age": null, "gender": null, "symptoms": [], '
        '"duration": null, "pain_severity": null, "vital_signs": null, '
        '"medical_history": null, "allergies": null, "medications": null, '
        '"clinical_notes": null, "pregnancy_status": null, "previous_visit_correlation": null}'
    )


def get_triage_fallback():
    """Safe default for ESI assessment"""
    return FallbackResponse(
        '{"reasoning": "Unable to assess due to system load. Defaulting to conservative ESI Level 3.", '
        '"esi_level": 3, "is_emergency": false}'
    )


def get_referral_fallback():
    """Safe default for referral"""
    return FallbackResponse(
        '{"referred_specialty": "General Medicine", '
        '"reasoning": "Conservative referral due to system constraints", '
        '"urgency": "Urgent", '
        '"recommended_timeline": "Within 24 hours"}'
    )


def get_paraclinical_fallback():
    """Safe default for diagnostic tests"""
    return FallbackResponse(
        '{"recommended_tests": ["CBC", "Basic Metabolic Panel"], '
        '"reasoning": "Standard initial workup", '
        '"priority": "Routine", '
        '"is_critical": false}'
    )


# ============= INVOKE WITH FALLBACK =============
def invoke_with_fallback(llm, messages, fallback_fn, node_name="", **kwargs):
    """
    Invoke LLM with intelligent fallback.

    Args:
        llm: Language model
        messages: Messages to send
        fallback_fn: Function that returns fallback response
        node_name: Name of the node (for logging)
        **kwargs: Additional args

    Returns:
        LLM response or fallback response
    """
    try:
        return invoke_with_retry(llm, messages, **kwargs)
    except Exception as e:
        logger.error(
            f"❌ {node_name} LLM failed after retries: {type(e).__name__}. "
            f"Using fallback response."
        )
        return fallback_fn()
