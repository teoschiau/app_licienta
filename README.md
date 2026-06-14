# AI Pathology Assistant & Spatial Transcriptomics Copilot

A clinical-grade, End-to-End (E2E) bioinformatics application designed to bridge the gap between advanced spatial transcriptomics and practical oncological diagnostics. Built with a unified Streamlit architecture, this system enables pathologists to manage patient data, generate spatial heatmaps, run predictive molecular inference from routine histology, and interact with a privacy-preserving AI Copilot.


## Key Features

*   **Comprehensive Patient Management:** Robust PostgreSQL-backed registry for full CRUD operations, ensuring persistent and ACID-compliant storage of clinical metadata and local file paths.
*   **Generative Spatial Inference:** An automated AI pipeline that synthesizes localized spatial transcriptomic profiles directly from standard H&E stained biopsy images (`.tif`). Utilizes vision foundation models (CONCH & UNI) and a Diffusion Image Transformer (DiT).
*   **Biomarker Extraction & Statistical Profiling:** Automated ingestion of `.h5ad` and `.pt` matrices to calculate global expression (mean) and spatial variance hotspots, rendered interactively via Plotly.
*   **Interactive Spatial Heatmaps:** Dynamically overlays normalized gene expression data directly onto high-resolution H&E tissue images with dynamic coordinate scaling and matplotlib-driven visual rendering.
*   **Automated Medical Reporting:** Integrates a locally hosted, quantized Large Language Model (Llama-3.2-3B) to autonomously generate comprehensive, biologically accurate medical reports regarding the tumor microenvironment.
*   **Interactive AI Copilot:** A conversational chat interface that retains the context of the patient's quantitative data and the generated report, enabling ad-hoc clinical questions with zero data leakage.


## System Architecture

The application adopts a unified, Python-based pipeline orchestrated through **Streamlit**, avoiding traditional frontend/backend separation to ensure fluid data state management. 

*   **UI / Controller:** Streamlit
*   **Database:** PostgreSQL
*   **Bioinformatics Core:** AnnData, Pandas, NumPy
*   **Machine Learning / Vision:** PyTorch, Hugging Face models (CONCH, UNI), Diffusion Models
*   **Local LLM:** `llama-cpp-python` (GGUF Quantized Llama-3.2-3B-Instruct)
*   **Visualization:** Plotly Express, Matplotlib


## Prerequisites

Before installing and running the application, ensure your system meets the following requirements:

1.  **Python 3.10+**
2.  **PostgreSQL** installed and running locally.
3.  **Hardware Acceleration:** An Apple Silicon Mac (M1/M2/M3 with MPS support) or a machine with an NVIDIA GPU (CUDA) is highly recommended for running the AI models.
4.  **Hugging Face Account:** You must have an access token with permissions to download gated models (specifically `MahmoodLab/conch`).


## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/pathology-assistant.git](https://github.com/your-username/pathology-assistant.git)
cd pathology-assistant

## How to Run the Application

Once your environment is set up and the database is initialized, follow these steps to launch the Pathology Assistant:

**1. Ensure PostgreSQL is Running**
Make sure your local PostgreSQL server is active. The application will need to connect to it immediately upon startup to load the patient registry.

**2. Activate Your Virtual Environment**
Open your terminal and activate the environment where your dependencies are installed:
```bash
# On macOS/Linux:
source env/bin/activate

# On Windows:
env\Scripts\activate
