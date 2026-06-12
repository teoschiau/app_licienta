import subprocess
import streamlit as st
import re
import os
from database import get_db_connection

@st.dialog("Start Sampling on a Patient")
def start_sampling_dialog(db_patients): 
    st.write("Select a valid patient to extract the genes:")
    
    valid_patients = []
    patient_paths = {}
    
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT patient_identifier, data_file_path, highres_tif_path FROM patients")
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        for record in records:
            pat_id = record['patient_identifier']
            st_path = record['data_file_path']
            tif_path = record['highres_tif_path']
            
            if st_path and tif_path and os.path.exists(st_path) and os.path.exists(tif_path):
                valid_patients.append(pat_id)
                patient_paths[pat_id] = {
                    "st_path": st_path,
                    "tif_path": tif_path
                }
    except Exception as e:
        st.error(f"Database error: {e}")
        return
        
    if not valid_patients:
        st.warning("No ready patients found. Ensure .tif and .h5ad files are uploaded via the Patient Profile.")
        return 
        
    slide_id = st.selectbox(
        "Select Patient*", 
        options=valid_patients,
        help="Only patients with complete raw data (.tif & .h5ad) in the database are shown."
    )
    
    if st.button("Run AI Pipeline", type="primary"):
        st.markdown(f"**Initializing Pipeline for {slide_id}...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        selected_st_path = patient_paths[slide_id]["st_path"]
        selected_tif_path = patient_paths[slide_id]["tif_path"]
        
        try:
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "")
            
            command = [
                "python","-u", "processing_sampling.py", 
                "--slide_out", slide_id,
                "--save_path", downloads_path,
                "--tif_path", selected_tif_path,
                "--st_path", selected_st_path
            ]
            
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            
            full_logs = ""
            status_text.markdown("Analyzing data state... (Checking for embeddings)")
            
            for line in process.stdout:
                full_logs += line 
                
                if "Starting on-the-fly extraction" in line:
                    status_text.warning("Pre-computed embeddings missing. Extracting features from TIF (This will take a few minutes)...")
                    
                match_extract = re.search(r"Extracting patches for (\d+) spots", line)
                if match_extract:
                    status_text.info(f"Processing image patches... Found {match_extract.group(1)} spatial spots.")
                    
                if "Saved embeddings to disk" in line:
                    status_text.success("Extraction complete! Cleared memory.")
                    
                if "Loading checkpoint" in line:
                    status_text.markdown("Loading Diffusion model weights into memory...")

                match_progress = re.search(r"Batch (\d+)/(\d+) DONE", line)
                if match_progress:
                    current = int(match_progress.group(1))
                    total = int(match_progress.group(2))
                    progress_pct = float(current) / float(total)
                    progress_pct = max(0.0, min(1.0, progress_pct))
                    progress_bar.progress(progress_pct)
                    status_text.markdown(f"**Generating samples:** Batch **{current}** out of **{total}** completed...")
                
                match_save = re.search(r"Successfully saved samples to:\s*(.+)", line)
                if match_save:
                    generated_file_path = match_save.group(1).strip()
            
            process.wait()
            
            if process.returncode == 0:
                progress_bar.progress(1.0)
                status_text.success(" Spatial inference completed successfully!")
                
                if generated_file_path and os.path.exists(generated_file_path):
                    with open(generated_file_path, "rb") as file:
                        st.divider()
                        st.download_button(
                            label="Download Generated Samples (.pt)",
                            data=file, file_name=os.path.basename(generated_file_path),
                            mime="application/octet-stream", type="primary", use_container_width=True
                        )
                else:
                    st.warning("Sampling finished, but couldn't locate the output file.")
            else:
                st.error("An error occurred during PyTorch execution.")
                with st.expander("View Error Details"):
                    st.code(full_logs, language="text")
                
        except Exception as e:
            st.error(f"Internal UI Error: {e}")

        finally:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
                process.kill()
                print(f"Procesul PyTorch pentru {slide_id} a fost oprit forțat pentru a elibera memoria.")