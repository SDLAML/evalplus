import os
from typing import List

import httpx
import openai

from evalplus.gen.util import openai_request
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import concurrent_call


class OpenAIChatDecoder(DecoderBase):
    def __init__(
        self, name: str, base_url=None, verify_certificate=True, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.verify_certificate = verify_certificate

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        user_message = (
            self.instruction_prefix + f"\n```\n{prompt.strip()}\n```"
        )
        assistant_prefix = (
            f"{self.response_prefix}\n```python\n"
            if self.response_prefix
            else "```python\n"
        )

        return self._codegen_batch_via_concurrency(user_message, assistant_prefix, num_samples)

    def _codegen_api_batch(self, user_message: str, assistant_prefix: str, batch_size: int) -> List[str]:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"),
            base_url=self.base_url,
            http_client=httpx.Client(verify=self.verify_certificate),
        )

        ret = openai_request.make_auto_request(
            client,
            user_message=user_message,
            assistant_prefix=assistant_prefix,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
        )

        outputs = []
        for item in ret.choices:
            content = item.message.content or ""
            # Strip the prefix if vLLM echoes it back in the response
            if content.startswith(assistant_prefix):
                content = content[len(assistant_prefix):]
            outputs.append(content)

        return outputs

    def _codegen_batch_via_concurrency(self, user_message: str, assistant_prefix: str, num_samples: int) -> List[str]:
        batches = concurrent_call(
            num_samples, self._codegen_api_batch, user_message, assistant_prefix, batch_size=1
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False
