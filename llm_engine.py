import streamlit as st
from llama_cpp import Llama

@st.cache_resource
def load_llm():
    """Loads model Llama 3.2 locally in cache memory of the app"""
    return Llama(
        model_path="Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=-1
    )