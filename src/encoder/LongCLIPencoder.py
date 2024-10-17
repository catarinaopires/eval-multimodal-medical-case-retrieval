import numpy as np
import torch
from LongCLIP.model import longclip


class LongCLIPencoder:
    def __init__(self, device, model_id="./checkpoints/longclip-B.pt"):
        self.device = device

        self.model, self.preprocess = longclip.load(model_id, device=self.device)

    def encode_text(self, text: str, withTruncation=True) -> np.ndarray:
        if not isinstance(text, list) and not isinstance(text, str):
            text = ""
            print("text is not a string or list of strings:")

        with torch.inference_mode():
            text = longclip.tokenize(text, truncate=withTruncation).to(self.device)

            # Get the text embedding
            text_embedding = self.model.encode_text(text)
            # Convert to numpy array
            text_embedding = text_embedding.cpu().detach().numpy()
            text_embedding = text_embedding / np.linalg.norm(text_embedding)

        return text_embedding.astype("float32")

    def encode_image(self, batch_images) -> np.ndarray:
        batch_images = self.preprocess(batch_images).unsqueeze(0).to(self.device)

        # Get image embeddings
        with torch.inference_mode():
            batch_embeddings = self.model.encode_image(batch_images)
            # Convert to numpy array
            batch_embeddings = batch_embeddings.cpu().detach().numpy()
            batch_embeddings = batch_embeddings.T / np.linalg.norm(batch_embeddings)

        return batch_embeddings.astype("float32")
