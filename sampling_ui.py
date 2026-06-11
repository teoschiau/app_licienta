import subprocess
import streamlit as st
import re
import os

BASE_DATA_PATH = "/Users/teoschiau/Documents/Licienta/Stem/Stem/hest1k_datasets/her2st/"
EMBEDDINGS_DIR = os.path.join(BASE_DATA_PATH, "processed_data", "1spot_uni_ebd")

@st.dialog("Start Sampling on a Patient")
def start_sampling_dialog(db_patients): 
    st.write("Select a valid patient to run the diffusion model.")
    
    if os.path.exists(EMBEDDINGS_DIR):
        local_patients = [f.name.replace("_uni.pt", "") for f in os.scandir(EMBEDDINGS_DIR) if f.is_file() and f.name.endswith("_uni.pt")]
    else:
        local_patients = []
        st.error(f"Cannot find the directory: {EMBEDDINGS_DIR}")
        
    valid_patients = [patient for patient in db_patients if patient in local_patients]
    
    if not valid_patients:
        st.warning("No ready patients found. Ensure patients are added to the DB and their embedding files exist in the dataset folder.")
        return 
        
    slide_id = st.selectbox(
        "Select Patient*", 
        options=valid_patients,
        help="Only patients present in both the database and local dataset are shown."
    )
    
    if st.button("Run Diffusion Sampling", type="primary"):
        st.markdown(f"**Initializing AI Engine for {slide_id}...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_expander = st.expander("Live Backend Errors", expanded=False)
        generated_file_path = None 
        
        try:
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "")
            
            command = [
                "python","-u", "sample.py", 
                "--slide_out", slide_id,
                "--save_path", downloads_path,
                "--data_path", BASE_DATA_PATH
            ]
            
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            
            full_logs = ""
            status_text.markdown("⏳ Loading model weights into memory... (This may take a minute)")
            
            for line in process.stdout:
                full_logs += line 
                
                match_progress = re.search(r"Batch (\d+)/(\d+) DONE", line)
                if match_progress:
                    current = int(match_progress.group(1))
                    total = int(match_progress.group(2))
                    progress_pct = float(current) / float(total)
                    progress_pct = max(0.0, min(1.0, progress_pct))
                    progress_bar.progress(progress_pct)
                    status_text.markdown(f" **Generating samples:** Batch **{current}** out of **{total}** completed...")
                
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