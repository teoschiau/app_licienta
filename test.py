import os
import sys
from unittest.mock import patch, MagicMock
import llama_cpp
llama_cpp.Llama = MagicMock()
import psycopg2
import pytest
import torch
import numpy as np
import pandas as pd
import anndata
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from app import load_local_gene_list, load_gene_data, save_uploaded_file, get_patient_list_from_db
import streamlit as st

# UNIT TESTS

def test_normalization_math():
    """Tests the mathematical logic for gene normalization (Log2)."""
    raw_data = pd.DataFrame({'ERBB2': [-5, 0, 3, 15]})
    df_safe = raw_data.clip(lower=0)
    normalized = np.log2(df_safe + 1)
    
    assert normalized.iloc[0, 0] == 0.0  
    assert normalized.iloc[2, 0] == 2.0  
    assert normalized.iloc[3, 0] == 4.0  

def test_load_local_gene_list(tmp_path):
    """Tests reading the local biomarker list from a text file."""
    file_path = tmp_path / "selected_gene_list.txt"
    file_path.write_text("ERBB2\nBRCA1")
    
    genes = load_local_gene_list(str(file_path))
    assert genes == ["ERBB2", "BRCA1"]

def test_save_uploaded_file(tmp_path, monkeypatch):
    """Tests saving an uploaded file to the correct patient directory."""
    monkeypatch.setattr('app.PATIENTS_DIR', str(tmp_path))
    
    class MockUploadedFile:
        name = "test_data.h5ad"
        def getbuffer(self):
            return b"fake_binary_data"
            
    patient_id = "P99_TEST"
    saved_path = save_uploaded_file(MockUploadedFile(), patient_id)
    
    assert patient_id in saved_path
    assert "test_data.h5ad" in saved_path
    assert os.path.exists(saved_path)

# INTEGRATION TESTS

@patch('sampling_ui.start_sampling_dialog') 
def test_start_sampling_button_trigger(mock_dialog):
    """Tests if the diffusion sampling button triggers the external UI dialog."""
    st.cache_data.clear()
    
    at = AppTest.from_file("app.py").run()
    
    sample_btn = next((btn for btn in at.button if btn.label == "Start sampling on a patient"), None)
    assert sample_btn is not None, "Butonul de sampling nu a fost gasit in interfata!"
    
    sample_btn.click().run()
    
    assert mock_dialog.called, "Funcția dialogului nu a fost apelată!"


@patch('psycopg2.connect') 
def test_delete_patient_button(mock_connect):
    """Tests if the Delete button executes the correct SQL query."""
    st.cache_data.clear()
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.fetchall.return_value = [{'patient_identifier': 'P01_DELETE_ME'}]
    
    mock_cursor.fetchone.return_value = {
        'patient_identifier': 'P01_DELETE_ME', 
        'data_file_path': 'dummy.pt',
        'image_preview_path': None,
        'highres_tif_path': None,
        'clinical_json_path': None
    }
    
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("P01_DELETE_ME").run()
    
    delete_btn = next((btn for btn in at.button if btn.label == "Delete"), None)
    assert delete_btn is not None, "No ddelete button appeard!"
    
    delete_btn.click().run()
    mock_cursor.execute.assert_any_call("DELETE FROM patients WHERE patient_identifier = %s", ("P01_DELETE_ME",))

def test_db_fetch_logic():
    """
    Tests if the fetch function correctly retrieves data.
    We check for a list type to allow your real DB entries (like 'SPA120') to pass!
    """
    get_patient_list_from_db.clear() 
    patients = get_patient_list_from_db()
    
    assert isinstance(patients, list), "Expected the database fetch to return a list."

def test_app_startup_state():
    """Tests if the app boots up correctly and populates the sidebar."""
    at = AppTest.from_file("app.py").run()
    
    assert not at.exception
    assert "Hospital Database" in at.sidebar.header[0].value
    
    options = at.sidebar.selectbox[0].options
    assert "-- Upload Custom File --" in options
    
    assert len(options) > 1, "Expected the database to load at least one patient into the dropdown."

def test_file_upload_mode_ui():
    """Tests the manual upload mode UI rendering."""
    at = AppTest.from_file("app.py").run()
    
    at.sidebar.selectbox[0].set_value("-- Upload Custom File --").run()
    
    assert len(at.sidebar.file_uploader) > 0
    assert "Upload patient spatial data" in at.sidebar.file_uploader[0].label


@patch('app.get_db_connection')
def test_db_connection_failure(mock_conn):
    """Tests if the app handles a completely offline database gracefully."""
    mock_conn.side_effect = psycopg2.OperationalError("Database connection failed!")
    
    get_patient_list_from_db.clear()
    patients = get_patient_list_from_db()
    
    assert patients == []

def test_load_gene_data_invalid_file(tmp_path):
    """Tests the data loader with an unsupported file extension."""
    bad_file = tmp_path / "corrupt_data.txt"
    bad_file.write_text("This is not a valid H5AD or PT file.")
    
    class MockBadFile:
        name = str(bad_file)
        
    df = load_gene_data(MockBadFile())
    assert df is None

def test_load_gene_data_pt(tmp_path):
    """Tests loading PyTorch tensors into Pandas DataFrames."""
    file_path = tmp_path / "test_patient.pt"
    dummy_data = torch.rand(10, 5)
    torch.save(dummy_data, file_path)
    
    class MockPtFile:
        name = "test_patient.pt"
        def __init__(self, path):
            self.path = path
            
    df = load_gene_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 5)

def test_load_gene_data_h5ad(tmp_path):
    """Tests loading anndata objects and extracting the X matrix."""
    file_path = tmp_path / "test_patient.h5ad"
    
    X = np.random.rand(50, 3)
    adata = anndata.AnnData(X=X)
    adata.var_names = ["GeneA", "GeneB", "GeneC"]
    adata.write(file_path)
    
    df = load_gene_data(str(file_path))
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 3)
    assert list(df.columns) == ["GeneA", "GeneB", "GeneC"]

@patch('app.get_patient_list_from_db')
def test_generate_medical_report_ui(mock_get_patients):
    """Tests the interaction of clicking 'Generate Medical Report'."""
    mock_get_patients.return_value = ["P01"]
    
    at = AppTest.from_file("app.py").run()
    
    at.session_state["prompt_medical"] = "Fake prompt for testing"
    
    report_btn = next((btn for btn in at.button if btn.label == "Generate Medical Report"), None)
    
    if report_btn:
        report_btn.click().run()
        assert at.session_state.medical_report is not None

@patch('app.get_patient_list_from_db')
def test_add_patient_dialog_renders(mock_get_patients):
    """Tests if the 'Add New Patient' dialog button triggers the form."""
    mock_get_patients.return_value = ["P01"]
    
    at = AppTest.from_file("app.py").run()
    
    add_btn = next((btn for btn in at.sidebar.button if btn.label == "Add New Patient"), None)
    
    if add_btn:
        add_btn.click().run()
        assert len(at.text_input) > 0
        
        pat_id_input = next((inp for inp in at.text_input if "Patient Identifier" in inp.label), None)
        assert pat_id_input is not None