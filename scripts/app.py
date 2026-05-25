import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fashionrag.embeddings import load_model
from fashionrag.hybrid import hybrid_search

def search(query):
    if not query.strip():
        return None

    results = hybrid_search(query)
    rows = []

    for product in results:
        rows.append(
            {
                "name": product["name"],
                "color": product["color"],
                "gender": product["gender"],
                "article_type": product["article_type"],
                "semantic_score": product["semantic_score"],
                "keyword_score": product["keyword_score"],
                "final_score": product["final_score"],
            }
        )
    return pd.DataFrame(rows)

load_model()

demo = gr.Interface(
    fn = search,
    inputs = gr.Textbox(label="Search"),
    outputs = gr.Dataframe(label="Results"),
    title = "FashionRAG",
)

demo.launch(
    server_name = '127.0.0.1',
    server_port = 7860,
    share = True,
    inbrowser = True,
    show_error = True
)
