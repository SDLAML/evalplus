import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    user_message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    assistant_prefix: str = None,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_tokens"] = max_tokens
    if model.startswith("o1-"):  # pop top-p and max_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_tokens")
        temperature = 1.0  # o1 models do not support temperature

    messages = [{"role": "user", "content": user_message}]
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})
        kwargs.setdefault("stop", ["\n```\n"])

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret
