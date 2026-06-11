import streamlit as st
import pandas as pd
import numpy as np
import torch
import anndata
from llama_cpp import Llama
import plotly.express as px
import os
import glob
import json
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import numpy as np
import tempfile
import psycopg2
from psycopg2.extras import RealDictCursor
import shutil
import time

GENE_LIST_PATH = "selected_gene_list.txt"
PATIENTS_DIR = "./patients"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "")
}

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn, conn.cursor(cursor_factory=RealDictCursor)

@st.cache_data(ttl=10)
def get_patient_list_from_db():
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT patient_identifier FROM patients ORDER BY patient_identifier")
        patients = cursor.fetchall()
        cursor.close()
        conn.close()
        return [p['patient_identifier'] for p in patients]
    except Exception as e:
        st.error(f"Eroare conectare BD: {e}")
        return []
    
def save_uploaded_file(uploaded_file, patient_id):
    if uploaded_file is None:
        return None
    
    save_dir = os.path.join(PATIENTS_DIR, patient_id)
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.abspath(file_path)

patient_list = get_patient_list_from_db()

st.set_page_config(page_title="HER2 Pathology Assistant", page_icon="🧬", layout="wide")

@st.cache_resource
def load_llm():
    return Llama(
        model_path="Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=-1
    )

llm = load_llm()

@st.cache_data
def load_full_adata(file_input):
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
        st.error(f"Eroare la citirea fișierului {file_name}: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return None


def load_gene_data(file_input):
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
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return None

@st.dialog("Add New Patient Profile")
def add_patient():
    st.write("Complete the fields below to register a new patient.")

    new_pat_id = st.text_input("Patient Identifier (ex: P01)*")
            
    st.markdown("**Transcriptomic Data (.h5ad/.h5/.pt)**")
    data_path_input = st.text_input("Absolute path:", key="data_path_input")
    data_upload = st.file_uploader("Or upload file here", type=['h5ad', 'h5', 'pt'], key="data_upload_file")
            
    st.markdown("**Image Preview (H&E .png/.jpg)**")
    img_path_input = st.text_input("Absolute path:", key="img_path_input")
    img_upload = st.file_uploader("Or upload file here", type=['png', 'jpg', 'jpeg'], key="img_upload_file")
            
    st.markdown("**High-Res Tissue Image (.tif)**")
    tif_path_input = st.text_input("Absolute path:", key="tif_path_input")
    tif_upload = st.file_uploader("Or upload file here", type=['tif', 'tiff'], key="tif_upload_file")
            
    st.markdown("**Clinical Data (.json)**")
    json_path_input = st.text_input("Absolute path:", key="json_path_input")
    json_upload = st.file_uploader("Or upload file here", type=['json'], key="json_upload_file")
                    
    if st.button("Save Patient Profile"):
        if not new_pat_id:
            st.error("Patient Identifier is required!")
        else:
            final_data_path = save_uploaded_file(data_upload, new_pat_id) or data_path_input or None
            final_img_path = save_uploaded_file(img_upload, new_pat_id) or img_path_input or None
            final_tif_path = save_uploaded_file(tif_upload, new_pat_id) or tif_path_input or None
            final_json_path = save_uploaded_file(json_upload, new_pat_id) or json_path_input or None
                    
            try:
                conn, cursor = get_db_connection()
                cursor.execute("""
                            INSERT INTO patients 
                            (patient_identifier, data_file_path, image_preview_path, highres_tif_path, clinical_json_path)
                            VALUES (%s, %s, %s, %s, %s)
                """, (new_pat_id, final_data_path, final_img_path, final_tif_path, final_json_path))
                conn.commit()
                cursor.close()
                conn.close()
                st.success(f"Patient {new_pat_id} added successfully!")
                time.sleep(1.5)
                st.rerun() 
            except Exception as e:
                st.error(f"Error to db: {e}")

@st.dialog("Update Patient Profile")
def update_patient_dialog(record):
    st.write(f"Update fields for **{record['patient_identifier']}**.")
    
    st.text_input("Patient Identifier", value=record['patient_identifier'], disabled=True)
    
    st.markdown("**Transcriptomic Data (.h5ad/.h5/.pt)**")
    data_path_input = st.text_input("Absolute path:", value=record.get('data_file_path') or "", key="upd_data_path")
    data_upload = st.file_uploader("Or upload new file here", type=['h5ad', 'h5', 'pt'], key="upd_data_upl")
    
    st.markdown("**Image Preview (H&E .png/.jpg)**")
    img_path_input = st.text_input("Absolute path:", value=record.get('image_preview_path') or "", key="upd_img_path")
    img_upload = st.file_uploader("Or upload new file here", type=['png', 'jpg', 'jpeg'], key="upd_img_upl")
    
    st.markdown("**High-Res Tissue Image (.tif)**")
    tif_path_input = st.text_input("Absolute path:", value=record.get('highres_tif_path') or "", key="upd_tif_path")
    tif_upload = st.file_uploader("Or upload new file here", type=['tif', 'tiff'], key="upd_tif_upl")
    
    st.markdown("**Clinical Data (.json)**")
    json_path_input = st.text_input("Absolute path:", value=record.get('clinical_json_path') or "", key="upd_json_path")
    json_upload = st.file_uploader("Or upload new file here", type=['json'], key="upd_json_upl")
    
    if st.button("Update in Database"):
        pat_id = record['patient_identifier']
        final_data_path = save_uploaded_file(data_upload, pat_id) if data_upload else data_path_input
        final_img_path = save_uploaded_file(img_upload, pat_id) if img_upload else img_path_input
        final_tif_path = save_uploaded_file(tif_upload, pat_id) if tif_upload else tif_path_input
        final_json_path = save_uploaded_file(json_upload, pat_id) if json_upload else json_path_input
        
        try:
            conn, cursor = get_db_connection()
            cursor.execute("""
                UPDATE patients 
                SET data_file_path = %s, image_preview_path = %s, highres_tif_path = %s, clinical_json_path = %s
                WHERE patient_identifier = %s
            """, (final_data_path, final_img_path, final_tif_path, final_json_path, pat_id))
            conn.commit()
            cursor.close()
            conn.close()
            
            st.rerun()
        except Exception as e:
            st.error(f"Error updating database: {e}")

gene_names = load_local_gene_list(GENE_LIST_PATH)

with st.sidebar:
    st.header("Hospital Database")
    
    if st.button("Add New Patient"):
        add_patient()

    st.divider()

    if gene_names:
        with st.expander(f"View Validated Biomarker Panel ({len(gene_names)} genes)"):
            st.caption("Targeted oncology panel used for spatial hotspot analysis.")
            df_genes = pd.DataFrame(gene_names, columns=["Gene Symbol"])
            st.dataframe(df_genes, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Select Patient")
    options = ["-- Upload Custom File --"] + patient_list
    selected_mode = st.selectbox("Choose action:", options)
    
    data_file = None 
    st.session_state.current_db_record = None
    
    if selected_mode == "-- Upload Custom File --":
        st.info("Manual Mode: Upload temporary data.")
        data_file = st.file_uploader("Upload patient spatial data", type=['pt', 'h5ad', 'h5'])
    else:
        try:
            conn, cursor = get_db_connection()
            cursor.execute("SELECT * FROM patients WHERE patient_identifier = %s", (selected_mode,))
            record = cursor.fetchone()
            cursor.close()
            conn.close()
            
            st.session_state.current_db_record = record
            
            if record and record['data_file_path'] and os.path.exists(record['data_file_path']):
                st.success(f"Loaded Profile: **{selected_mode}**")
                data_file = record['data_file_path']
            else:
                st.error(f"Calea către date ({record['data_file_path'] if record else 'N/A'}) este invalidă sau fișierul nu există.")
        except Exception as e:
            st.error(f"Eroare extragere date pacient: {e}")

    if data_file:
        current_filename = data_file.name if hasattr(data_file, 'name') else str(data_file)
    else:
        current_filename = ""
        
    if "current_file" not in st.session_state:
        st.session_state.current_file = ""
        
    if current_filename != st.session_state.current_file:
        st.session_state.current_file = current_filename
        st.session_state.medical_report = ""
        st.session_state.messages = []
        st.rerun()

prompt_medical = ""
NUM_GENES = 10
top_mean_str = ""
top_var_str = ""
df = None 

if data_file and gene_names:
    df = load_gene_data(data_file)
    if df is not None:
        if df.shape[1] == len(gene_names):
            df.columns = gene_names
            count_mtx_norm = df
        else:
            valid_genes = [g for g in gene_names if g in df.columns]
            df = df.loc[:, valid_genes]
            df_safe = df.clip(lower=0) 
            count_mtx_norm = np.log2(df_safe + 1)
            
        means = count_mtx_norm.mean(axis=0)
        variances = count_mtx_norm.var(axis=0)
        top_mean = means.nlargest(NUM_GENES)
        top_var = variances.nlargest(NUM_GENES)
            
        top_mean_str = ", ".join([f"{gene} ({val:.2f})" for gene, val in top_mean.items()])
        top_var_str = ", ".join([f"{gene} ({val:.2f})" for gene, val in top_var.items()])
            
        prompt_medical = (
            f"You are an expert oncologist and bioinformatician. Analyze the following spatial transcriptomic profile "
            f"extracted from the tissue of a patient.\n"
            f"Quantitative data:\n"
            f"- The top {NUM_GENES} most globally expressed genes are: {top_mean_str}.\n"
            f"- The top {NUM_GENES} genes with the highest variance (forming distinct spatial hotspots in the tumor) are: {top_var_str}.\n\n"
            f"Based on these biomarker networks, please provide a comprehensive analysis of the tumor microenvironment, "
            f"potential signaling pathways involved, and prognostic implications."
        )


if selected_mode != "-- Upload Custom File --":
    import time
    hdr_col1, hdr_col2, hdr_col3 = st.columns([3, 1, 1])
    
    with hdr_col1:
        st.header(f"Patient Profile: {selected_mode}")
        
    with hdr_col2:
        if st.button("Update", use_container_width=True):
            update_patient_dialog(st.session_state.current_db_record)
            
    with hdr_col3:
        if st.button("Delete", use_container_width=True):
            try:
                conn, cursor = get_db_connection()
                cursor.execute("DELETE FROM patients WHERE patient_identifier = %s", (selected_mode,))
                conn.commit()
                cursor.close()
                conn.close()
                
                get_patient_list_from_db.clear()
                
                st.success(f"Patient {selected_mode} was deleted successfully!")
                
                time.sleep(1.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error deleting patient: {e}")
    
    st.markdown("---")
    patient_folder = os.path.join(PATIENTS_DIR, selected_mode)
    
    prof_col1, prof_col2, prof_col3 = st.columns([1, 2, 1])
    
    with prof_col1:
        st.subheader("Downscaled H&E Tissue Image")
        
        record = st.session_state.get('current_db_record', None)

        if record and record.get('image_preview_path') and os.path.exists(record['image_preview_path']):
            st.image(record['image_preview_path'], caption="H&E Thumbnail (Preview)", use_container_width=True)
        else:
            st.info("No preview image available in Database.")

        if record and record.get('highres_tif_path') and os.path.exists(record['highres_tif_path']):
            tif_path = record['highres_tif_path']
            
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                with open(tif_path, "rb") as f:
                    st.download_button(
                        label="Download TIF",
                        data=f,
                        file_name=os.path.basename(tif_path),
                        mime="image/tiff",
                        help="Download file to open in QuPath"
                    )
            
            with btn_col2:
                if st.button("Copy Path"):
                    st.code(os.path.abspath(tif_path))
                    st.caption("Paste this path into QuPath File -> Open")
        else:
            st.warning("No TIF file found in Database for deep analysis.")
            
            
    with prof_col2:
        st.subheader("Clinical Metadata")
       
        record = st.session_state.get('current_db_record', None)
        if record and record.get('clinical_json_path') and os.path.exists(record['clinical_json_path']):
            try:
                with open(record['clinical_json_path'], 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                df_meta = pd.DataFrame(meta_data.items(), columns=["Attribute", "Value"])
                st.dataframe(df_meta, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading JSON: {e}")
        else:
            st.warning("No clinical metadata (.json) found for this patient in Database.")

    with prof_col3:
        record = st.session_state.get('current_db_record', None)
        
        db_her2_status = record.get('her2_status') if record else None
        
        if db_her2_status and db_her2_status not in ["Unknown", ""]:
            her2_status = db_her2_status
        else:
            her2_status = "Unknown"
            if df is not None:
                if "ERBB2" in df.columns:
                    if "ERBB2" in top_mean.index or "ERBB2" in top_var.index:
                        her2_status = "Positive (+)"
                    else:
                        her2_status = "Negative (-)"
                else:
                    her2_status = "N/A (ERBB2 missing)"
            
            if record and her2_status != "Unknown":
                try:
                    conn, cursor = get_db_connection()
                    cursor.execute("""
                        UPDATE patients 
                        SET her2_status = %s 
                        WHERE patient_identifier = %s
                    """, (her2_status, selected_mode))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    st.session_state.current_db_record['her2_status'] = her2_status
                    
                except Exception as e:
                    st.error(f"Eroare la actualizarea statusului HER2 în BD: {e}")

        st.metric(label="HER2 (ERBB2) Status", value=her2_status)
        st.metric(label="Tissue Spots", value=f"{df.shape[0]}" if df is not None else "N/A")

if df is not None:
    with st.expander("View Extracted Biomarkers", expanded=True):
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            df_mean = pd.DataFrame({'Gene': top_mean.index, 'Expression Level': top_mean.values})
            fig_mean = px.bar(df_mean, x='Expression Level', y='Gene', orientation='h',
                              title='Top 10 Global Expression (Mean)', color='Expression Level', color_continuous_scale='Viridis')
            fig_mean.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_mean, use_container_width=True)

        with col_chart2:
            df_var = pd.DataFrame({'Gene': top_var.index, 'Variance Level': top_var.values})
            fig_var = px.bar(df_var, x='Variance Level', y='Gene', orientation='h',
                             title='Top 10 Spatial Hotspots (Variance)', color='Variance Level', color_continuous_scale='Magma')
            fig_var.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_var, use_container_width=True)
    st.markdown("---")

Image.MAX_IMAGE_PIXELS = None

if selected_mode != "-- Upload Custom File --" and data_file:
    st.subheader("Heatmap generation for patient's tissue")
    
    record = st.session_state.get('current_db_record', None)
    img_path = None
    
    if record:
        if record.get('highres_tif_path') and os.path.exists(record['highres_tif_path']):
            img_path = record['highres_tif_path']
        elif record.get('image_preview_path') and os.path.exists(record['image_preview_path']):
            img_path = record['image_preview_path']
    
    if not img_path:
        st.warning("No image path found in the database for this patient!")
    else:
        selected_gene = st.selectbox("Select the gene: ", options=gene_names)
        
        if st.button(f"Generate the heatmap for {selected_gene}"):
            with st.spinner("Generating spatial heatmap, please wait..."):
                try:
                    img_raw = Image.open(img_path)
                    adata_spatial = load_full_adata(data_file)
                    
                    if adata_spatial is not None and 'spatial' in adata_spatial.obsm:
                        x_original = adata_spatial.obsm["spatial"][:, 0]
                        y_original = adata_spatial.obsm["spatial"][:, 1]
                        
                        img_width, img_height = img_raw.size
                        max_x_spot = np.max(x_original)
                        
                        scale_factor = 1.0
                        if max_x_spot > img_width:
                            scale_factor = (img_width / max_x_spot) * 0.96 # 0.96 adds a tiny padding
                            
                        x_scaled = x_original * scale_factor
                        y_scaled = y_original * scale_factor
                        
                        num_spots = len(x_original)
                        dynamic_size = max(0.5, min(20.0, 10000.0 / num_spots))
                        
                        if selected_gene in adata_spatial.var_names:
                            color_gt = adata_spatial[:, selected_gene].X
                            
                            if hasattr(color_gt, "toarray"):
                                color_gt = color_gt.toarray()
                            color_gt = color_gt.flatten()
                            
                            color_gt_norm = (color_gt - np.min(color_gt)) / (np.max(color_gt) - np.min(color_gt) + 1e-8)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            fig.patch.set_facecolor('none')
                            
                            ax.imshow(img_raw)
                            
                            im = ax.scatter(
                                x_scaled, y_scaled, 
                                c=color_gt_norm, 
                                cmap='viridis', 
                                s=dynamic_size, 
                                alpha=0.8, 
                                edgecolors='none'
                            )
                            
                            ax.set_title(f"Relative gene expression: {selected_gene} (Spots: {num_spots})", fontsize=16)
                            ax.axis('off')
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity of gene")
                            
                            st.pyplot(fig)
                            
                        else:
                            st.error(f"Gene '{selected_gene}' was not found in patient's data")
                    else:
                        st.error("Spatial coordinates could not be loaded")
                except Exception as e:
                    st.error(f"Error generating the graph: {e}")
    st.markdown("---")

col1, col2 = st.columns([1, 1])
CONTAINER_HEIGHT = 700 

if "medical_report" not in st.session_state:
    st.session_state.medical_report = ""

with col1:
    st.header("Medical Report")
    report_box = st.container(height=CONTAINER_HEIGHT)
    
    with report_box:
        if st.button("Generate Medical Report"):
            if prompt_medical:
                final_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt_medical}\n\n### Response:\n"
                with st.spinner("AI is analyzing the tumor microenvironment..."):
                    response = llm(final_prompt, max_tokens=2048, stop=["<|eot_id|>"])
                    st.session_state.medical_report = response['choices'][0]['text']
                    st.rerun()
            else:
                st.error("Please load patient data first!")
        
        if st.session_state.medical_report:
            st.markdown(st.session_state.medical_report)

with col2:
    st.header("AI Copilot Chat")
    prompt_chat = st.chat_input("Ask a question about the patient's results...")
    chat_box = st.container(height=CONTAINER_HEIGHT - 75)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if prompt_chat:
        st.session_state.messages.append({"role": "user", "content": prompt_chat})
        st.rerun()

    with chat_box:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with col2:
        with chat_box:
            with st.chat_message("assistant"):
                context = f"Patient Data Context: Top Means: {top_mean_str}. Top Variances: {top_var_str}.\n" if prompt_medical else ""
                
                if st.session_state.medical_report:
                    context += f"Previous Medical Report summary generated by you: {st.session_state.medical_report[:500]}...\n"

                user_question = st.session_state.messages[-1]["content"]
                chat_full_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{context}The user asks: {user_question}\n\n### Response:\n"
                
                with st.spinner("Thinking..."):
                    response = llm(chat_full_prompt, max_tokens=1024, stop=["<|eot_id|>"])
                    answer = response['choices'][0]['text'].strip()
                    st.markdown(answer)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})