import time
from langchain_together import ChatTogether

from civic_chat.env import RATE_LIMIT_DELAY

class TogetherAPIWithDelay(ChatTogether):
    # The together.ai API rate-limits API access for some models like deepseek.
    # Use this wrapper for those models.
    def _generate(self, *args, **kwargs):
        print(f"GEN")
        result = super()._call(*args, **kwargs)  # Await the parent async method
        print("sleep...")
        time.sleep(RATE_LIMIT_DELAY)
        print("done sleeping")
        return result

    async def _agenerate(self, *args, **kwargs):
        print(f"AGEN!")
        result = await super()._agenerate(*args, **kwargs)  # Await the parent async method
        print("sleep...")
        await asyncio.sleep(RATE_LIMIT_DELAY)
        print("done sleeping")
        return result
