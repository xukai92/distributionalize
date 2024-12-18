# distrilize

A package that turns LLM into primitive distributions with
- `sample` and `log_prob` methods for conditional distributions
- Proper handling of chat templates
- Support for both CLM and OAI models

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from distrilize.distrilize import CLMDist

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

dist = CLMDist(tokenizer, model)

messages = [
    {"role": "system", 
     "content": "You are a helpful assistant."},
    {"role": "user", 
     "content": "Hello, how are you?"},
]

s, lp = dist.sample(messages, do_sample=False, max_length=512)

dist.log_prob(s, messages)
```
