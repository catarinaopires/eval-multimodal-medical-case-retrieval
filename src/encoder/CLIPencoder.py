import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


class CLIPencoder:
    def __init__(self, device, model_id="openai/clip-vit-base-patch32"):
        self.device = device

        self.model = CLIPModel.from_pretrained(model_id).to(device)

        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def tokenize(self, text: str):
        if not isinstance(text, list) and not isinstance(text, str):
            text = ""
            print("text is not a string or list of strings")

        if text == "nan":
            text = ""

        with torch.inference_mode():
            inputs = self.tokenizer.tokenize(text)

            return len(inputs)

    def encode_text(self, text: str, withTruncation=True) -> np.ndarray:
        if not isinstance(text, list) and not isinstance(text, str):
            text = ""
            print("text is not a string or list of strings")

        with torch.inference_mode():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=withTruncation,
                max_length=77,
            ).to(self.device)

            # Get the text embedding
            text_embedding = (
                self.model.get_text_features(**inputs).cpu().detach().numpy()
            )
            text_embedding = text_embedding / np.linalg.norm(text_embedding)

        return text_embedding

    def encode_image(self, batch_images) -> np.ndarray:
        # Process and resize
        batch = self.processor(images=batch_images, return_tensors="pt", padding=True)[
            "pixel_values"
        ].to(self.device)

        # Get image embeddings
        batch_embeddings = self.model.get_image_features(pixel_values=batch)

        # Convert to numpy array
        batch_embeddings = batch_embeddings.cpu().detach().numpy()
        batch_embeddings = batch_embeddings.T / np.linalg.norm(batch_embeddings)

        return batch_embeddings
