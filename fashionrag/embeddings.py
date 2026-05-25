import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from fashionrag.settings import BATCH_SIZE, MODEL_NAME


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    return tokenizer, model, device


def embed_texts(texts, batch_size=BATCH_SIZE, desc="Making text embeddings"):
    tokenizer, model, device = load_model()
    vectors = []
    dataloader = DataLoader(texts, batch_size=batch_size)

    if desc:
        dataloader = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1)
            text_features = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            vectors.append(text_features.cpu().numpy())

    return np.vstack(vectors)
