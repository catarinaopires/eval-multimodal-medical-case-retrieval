import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM


class LlamaEncoder:
    def __init__(self, device, model_id="meta-llama/Meta-Llama-3-8B"):
        self.device = device

        self.model = LlamaForCausalLM.from_pretrained(
            model_id, output_hidden_states=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=8000)

    def encode_text(self, text: str, withTruncation=True) -> np.ndarray:
        if not isinstance(text, list) and not isinstance(text, str):
            text = ""
            print("text is not a string or list of strings:", text)

        with torch.inference_mode():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=withTruncation
            ).to(self.device)

            # Encode the text and generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract the embeddings
            hidden_states = outputs.hidden_states[-1]  # Taking the last hidden state
            embeddings = hidden_states.mean(dim=1)  # Taking the mean of the embeddings

            # Convert embeddings to numpy array
            embeddings = embeddings.cpu().detach().numpy()

        return embeddings
