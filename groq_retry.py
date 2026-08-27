import re
import time
from pathlib import Path

from groq import RateLimitError

from logging_setup import setup_logger

log_path = Path(__file__).parent / 'stack_feed.log'
retry_logger = setup_logger('retry_logger', log_path)


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
            match = re.search(r"try again in ([\d.]+)(ms|s)", str(e))
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                wait_time = value / 1000 if unit == 'ms' else value
            else:
                wait_time = 60
            retry_logger.warning(
                f"Groq rate limit hit, retrying in {wait_time} seconds. Attempt #{attempt + 1}"
            )
            time.sleep(wait_time)
    raise RuntimeError("Exceeded maximum number of retry attempts for a Groq call.")
