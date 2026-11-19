import asyncio
import re
import time
from groq import RateLimitError

from logging_setup import setup_logger

retry_logger = setup_logger('retry_logger')


def _retry_delay(error: RateLimitError, attempt: int) -> float:
    """Prefer Groq's retry headers, then its message, then short backoff."""
    response = getattr(error, 'response', None)
    headers = getattr(response, 'headers', {}) or {}

    retry_after_ms = headers.get('retry-after-ms')
    if retry_after_ms:
        try:
            return max(0.0, float(retry_after_ms) / 1000)
        except (TypeError, ValueError):
            pass

    retry_after = headers.get('retry-after')
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

    match = re.search(r'try again in ([\d.]+)(ms|s)', str(error), re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2).lower() == 'ms' else value

    message = str(error).lower()
    if 'per day' in message or 'daily' in message or 'rpd' in message or 'tpd' in message:
        raise error

    return min(30.0, 2.0 ** attempt)


def run_groq_with_retry(make_call, max_attempts: int = 5):
    """
    Runs a Groq API call and retries it if Groq sends a rate limit error.

    make_call: a function that takes no arguments and makes the actual
               Groq call. Example:
               run_groq_with_retry(lambda: client.chat.completions.create(...))
    max_attempts: how many times to try before giving up.
    """
    for attempt in range(max_attempts):
        try:
            return make_call()
        except RateLimitError as e:
            if attempt + 1 >= max_attempts:
                retry_logger.error('Groq rate limit persists after %d attempts', max_attempts)
                raise
            wait_time = _retry_delay(e, attempt)
            retry_logger.warning(
                'Groq rate limit hit; retrying in %.2f seconds (attempt %d/%d)',
                wait_time,
                attempt + 1,
                max_attempts,
            )
            time.sleep(wait_time)

    raise RuntimeError('Exceeded maximum number of retry attempts for a Groq call.')


async def stream_groq_with_retry(make_stream, max_attempts: int = 5):
    """Retry a rate-limited stream only before yielding output to avoid duplicates."""
    for attempt in range(max_attempts):
        yielded_chunk = False
        try:
            async for chunk in make_stream():
                yielded_chunk = True
                yield chunk
            return
        except RateLimitError as error:
            if yielded_chunk or attempt + 1 >= max_attempts:
                raise
            wait_time = _retry_delay(error, attempt)
            retry_logger.warning(
                'Groq stream rate limit hit; retrying in %.2f seconds (attempt %d/%d)',
                wait_time,
                attempt + 1,
                max_attempts,
            )
            await asyncio.sleep(wait_time)
