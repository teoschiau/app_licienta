import os
import time
import streamlit as st
from database import get_db_connection

from dotenv import load_dotenv

load_dotenv()

PATIENTS_DIR = os.getenv("PATIENTS_DIRECTORY",".")

def save_uploaded_file(uploaded_file, patient_id):
    """Saves an uploaded file to a patient-specific directory and returns the absolute path."""
    if uploaded_file is None:
        return None
    
    save_dir = os.path.join(PATIENTS_DIR, patient_id)
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.abspath(file_path)

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