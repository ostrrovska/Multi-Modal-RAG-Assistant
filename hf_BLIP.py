import logging
from pathlib import Path
from typing import List
from llama_index.core import Document
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


# ===== GLOBAL MODEL LOAD (Load once, use everywhere) =====
blip_processor = None
blip_model = None


def load_blip_model():
    """Lazy-load BLIP model only when needed"""
    global blip_processor, blip_model
    if blip_processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


# ===== IMAGE TO DOCUMENT FUNCTION =====
def image_to_document(image_path: str) -> Document:
    """Process image to Document with caption"""
    try:
        img = Image.open(image_path).convert('RGB')

        # Generate caption
        load_blip_model()
        inputs = blip_processor(img, return_tensors="pt").to(blip_model.device)
        caption = blip_model.generate(**inputs, max_new_tokens=100)[0]
        caption_text = blip_processor.decode(caption, skip_special_tokens=True)

        return Document(
            text=f"Caption: {caption_text}",
            metadata={
                "file_path": image_path,
                "file_type": "IMAGE",
                "caption": caption_text
            }
        )
    except Exception as e:
        logging.error(f"IMAGE PROCESSING FAILED: {image_path} - {str(e)}")
        return Document(text="", metadata={"file_path": image_path, "error": str(e)})


# ===== TEST CAPTION GENERATION =====
def test_caption_generation():
    """Test function for caption generation"""
    test_images = ["data/image.jpeg", "data/image1.jpg", "data/image2.jpg", "data/image3.jpg", "data/image4.jpg"]

    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"Test image missing: {img_path}")
            continue

        doc = image_to_document(img_path)
        print(f"\nImage: {img_path}")
        print(f"Caption: {doc.metadata['caption']}")

# Call test in main() or separately
test_caption_generation()