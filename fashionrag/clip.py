import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from fashionrag.settings import BATCH_SIZE, CLIP_MODEL_NAME


clip_processor, clip_model, clip_device = None, None, None


def load_clip_model():
    global clip_processor, clip_model, clip_device

    if clip_processor is None or clip_model is None:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_model.to(clip_device)
        clip_model.eval()

    return clip_processor, clip_model, clip_device


def get_vector(features):
    if getattr(features, "pooler_output", None) is not None:
        return features.pooler_output

    if getattr(features, "text_embeds", None) is not None:
        return features.text_embeds

    if getattr(features, "image_embeds", None) is not None:
        return features.image_embeds

    return features


def normalize(features):
    features = get_vector(features)
    return features / features.norm(dim=1, keepdim=True)


def embed_clip_texts(texts, batch_size=BATCH_SIZE, desc="Making CLIP text embeddings"):
    processor, model, device = load_clip_model()
    vectors = []
    dataloader = DataLoader(texts, batch_size=batch_size)

    if desc:
        dataloader = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for batch in dataloader:
            inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_features = normalize(text_features)
            vectors.append(text_features.cpu().numpy())

    return np.vstack(vectors)


def embed_images(images, batch_size=BATCH_SIZE, desc="Making CLIP image embeddings", total=None):
    processor, model, device = load_clip_model()
    vectors = []
    batch = []

    if desc:
        images = tqdm(images, desc=desc, total=total)

    with torch.no_grad():
        for image in images:
            batch.append(image.convert("RGB"))

            if len(batch) == batch_size:
                image_vectors = embed_image_batch(batch, processor, model, device)
                vectors.append(image_vectors)
                batch = []

        if batch:
            image_vectors = embed_image_batch(batch, processor, model, device)
            vectors.append(image_vectors)

    return np.vstack(vectors)


def embed_image_batch(batch, processor, model, device):
    inputs = processor(images=batch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = normalize(image_features)

    return image_features.cpu().numpy()
