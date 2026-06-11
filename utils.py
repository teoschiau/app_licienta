import os
import tempfile
import torch
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import streamlit as st

GENE_LIST_PATH = "selected_gene_list.txt"

@st.cache_data
def load_full_adata(file_input):
    """Loads data in formats .h5ad sau .h5."""
    is_uploaded = hasattr(file_input, 'read')
    file_name = file_input.name if is_uploaded else str(file_input)
    
    tmp_path = None
    target_path = file_input
    
    if is_uploaded:
        _, ext = os.path.splitext(file_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_input.getvalue())
            tmp_path = tmp.name
        target_path = tmp_path
        
    try:
        if file_name.endswith('.h5ad'):
            adata = anndata.read_h5ad(target_path)
            return adata
        elif file_name.endswith('.h5'):
            adata = sc.read_10x_h5(target_path)
            return adata
    except Exception as e:
        st.error(f"Error to reading file {file_name}: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return None

def load_gene_data(file_input):
    """Extract gene matrix from files .pt, .h5ad sau .h5."""
    file_name = file_input.name if hasattr(file_input, 'name') else str(file_input)
    
    try:
        if file_name.endswith('.pt'):
            tensor_data = torch.load(file_input, map_location="cpu")
            if len(tensor_data.shape) == 3:
                tensor_data = tensor_data.squeeze(1)
            return pd.DataFrame(tensor_data.numpy())
        
        elif file_name.endswith('.h5ad') or file_name.endswith('.h5'):
            adata = load_full_adata(file_input)
            
            if adata is not None:
                X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                return pd.DataFrame(X_data, columns=adata.var_names)
            
    except Exception as e:
        st.error(f"Eroare la extragerea datelor din {file_name}: {e}")
        
    return None

def load_local_gene_list(path):
    """Reads biomarkers list"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return None