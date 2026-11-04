import os
from openai import AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"      # Standard mode only: child data must never reach a -contributor model
assert not SPARK_MODEL.endswith("-contributor")

def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"], timeout=60, max_retries=1)

import functools
shared_client = functools.cache(make_client)   # one client per process, created on first use
