import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LlavaGenerator:
    def __init__(self, device, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = device

        self.model = LlavaForConditionalGeneration.from_pretrained(model_id).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # self.processor.image_processor.do_rescale = False # Disable image rescaling if images are previously resized using the datasets/resize_images.py script

    def generate_image_description(self, batch, max_generated_tokens=1024) -> str:
        prompt = "USER: <image>\nDescribe the image. ASSISTANT:"

        with torch.inference_mode():
            inputs = self.processor(
                text=[prompt] * len(batch),
                images=batch,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate
            generate_ids = self.model.generate(
                **inputs, max_new_tokens=max_generated_tokens
            )
            generated_text = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True
            )

            generated_descriptions = []
            for text in generated_text:
                generated_descriptions.append(text.split("ASSISTANT:")[-1])

            return generated_descriptions
