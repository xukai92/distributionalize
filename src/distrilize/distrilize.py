from abc import ABC, abstractmethod

import torch

class BaseLLMDist(ABC):
    def __init__(self, tokenizer):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.tokenizer = tokenizer

    @abstractmethod
    def sample(self, cond):
        '''
        Sample from the distribution conditioned on cond.
        Both cond and s are in the message format.
        '''
        pass
    
    @abstractmethod
    def log_prob(self, s, cond):
        '''
        Compute the log probability of s given cond.
        Both s and cond are in the message format.
        '''
        pass

class CLMDist(BaseLLMDist):

    def __init__(self, tokenizer, clm):
        super().__init__(tokenizer=tokenizer)
        self.clm = clm

    def sample(self, cond, length_normalization=True, **kwargs):
        '''
        Sample from the distribution conditioned on cond.

        Examples:
        cond:
        [
            {"role": "system", 
            "content": "You are a helpful assistant."},
            {"role": "user", 
            "content": "Hi, how are you?"},
        ]
        return:
        (
            {"role": "assistant", 
            "content": "Hi, I'm doing great! How can I assist you today?"},
            -10.23
        )
        '''
        kwargs = {"output_logits": True, "return_dict_in_generate": True} | kwargs

        messages = cond

        # encode the cond
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.clm.generate(**inputs, **kwargs)

        # generate an empty response to find the common prefix and suffix
        empty_outputs = self.tokenizer(
            self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": ""}], tokenize=False
            ), return_tensors="pt"
        )

        # find common prefix and suffix between empty_outputs and generated sequence
        empty_tokens = empty_outputs["input_ids"][0].tolist()
        gen_tokens = outputs.sequences[0].tolist()

        # find prefix length by comparing from start
        prefix_len = 0
        while prefix_len < len(empty_tokens) and prefix_len < len(gen_tokens):
            if empty_tokens[prefix_len] != gen_tokens[prefix_len]:
                break
            prefix_len += 1

        # find suffix length by comparing from end
        suffix_len = 0
        while (suffix_len < len(empty_tokens) - prefix_len and 
            suffix_len < len(gen_tokens) - prefix_len):
            if empty_tokens[-(suffix_len+1)] != gen_tokens[-(suffix_len+1)]:
                break
            suffix_len += 1

        # get the new tokens without the prefix and suffix in chat template
        new_tokens = gen_tokens[prefix_len:-suffix_len]

        # the scores are for the new tokens with prefix and suffix in chat template
        new_logits = outputs.logits # scores start from the first new token
        # we need to extract the scores for the new tokens only by first removing the suffix
        # new_logits = new_logits[:-suffix_len]
        # then we need to shift the scores to match the new tokens by using its length
        n_new_tokens = len(new_tokens)
        n_new_scores = len(new_logits)
        new_logits = [new_logits[n_new_scores-n_new_tokens+i] for i in range(n_new_tokens)]

        # compute and extract the log probabilities
        log_probs = torch.stack([torch.log_softmax(score, dim=-1)[0,token] 
                                 for (token, score) in zip(new_tokens, new_logits)])
        
        # decode the tokens
        sample = self.tokenizer.decode(new_tokens)

        if length_normalization:
            log_probs = log_probs / len(new_tokens)
        lp = log_probs.sum().item()
        
        return {"role": "assistant", "content": sample}, lp
    
    def log_prob(self, s, cond, length_normalization=True, **kwargs):
        messages = cond + [s]

        new_tokens = self.tokenizer(s['content'], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # encode the cond
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clm(**inputs, **kwargs)

        input_ids = inputs["input_ids"][0]

        # find where the new tokens start in the input sequence
        start_idx = None
        for i in range(len(input_ids) - len(new_tokens) + 1):
            if torch.equal(input_ids[i:i+len(new_tokens)], new_tokens):
                start_idx = i
                break

        new_logits = outputs.logits[:,start_idx:start_idx+len(new_tokens),:]
        new_logits = [new_logits[:,i,:] for i in range(new_logits.shape[1])]
        log_probs = torch.stack([torch.log_softmax(score, dim=-1)[0,token] 
                                 for (token, score) in zip(new_tokens, new_logits)])
        
        
        if length_normalization:
            log_probs = log_probs / len(new_tokens)
        lp = log_probs.sum().item()

        return lp



class VLLMDist(BaseLLMDist):

    def __init__(self, tokenizer, vllm):
        super().__init__(tokenizer=tokenizer)
        self.oai = oai
