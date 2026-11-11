# AutoLab 🔬
**An AI-Assisted Scientific Experiment Planning, Execution and Analysis Platform**

AutoLab is a desktop AI research laboratory built for students and researchers. It guides researchers through an evidence-grounded scientific inquiry cycle:
**Research Question ➔ Literature Evidence Retrieval ➔ Structured Experiment Plan ➔ Safe Predefined Run ➔ Real Metrics & Charts ➔ Scientific Interpretation & Next Steps.**

---

## 🚀 Key Features

1. **🧭 Research Workspace:**
   - Define project titles, research questions, operational hypotheses, objectives, constraints, and research disciplines.
   - Persistent SQLite metadata storage with multi-project switching.
   - Built-in "Load Iris Demo Workspace" for one-click faculty review.

2. **📚 Knowledge Library (Local RAG):**
   - Ingest PDF (`PyMuPDF`), TXT, and Markdown files.
   - Character-level sliding chunking (default: 800 chars, 120 overlap).
   - Local dense embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
   - Vector indexing and cosine similarity search via `faiss-cpu` (with seamless NumPy/scikit-learn fallback).
   - Strict citation provenance: maps every chunk to source filename and page number. Zero fabricated citations.

3. **📝 Experiment Planner:**
   - Generates 12-section structured empirical protocols (Objective, Testable Hypothesis, Independent/Dependent/Controlled Variables, Methodology, Baseline, Evaluation Metrics, Expected Outputs, Risks & Limitations, Next Steps, Verified Citations, Assumptions).
   - Grounded in retrieved literature context when documents are present.
   - Powered offline by local **Ollama** (`qwen2.5:7b-instruct` default, or switchable `qwen2.5:3b-instruct`) with strict JSON Schema output validation via Pydantic (and optional Google Gemini cloud support).
   - Explicit offline template button and deterministic fallback if local Ollama is stopped.
   - In-place plan customization and revision history.

4. **⚡ Safe Predefined Experiment Runner:**
   - **Zero Arbitrary Code Execution:** Strictly forbids `eval`, `exec`, or shell code execution.
   - Runs tested, reproducible scikit-learn pipelines with fixed random seeds and train/test splits.
   - Automatic missing-value imputation (median for numeric, mode for categorical) and standard scaling.
   - Built-in datasets:
     - Fisher Iris (Classification)
     - Diabetes Progression (Regression)
     - Synthetic Binary Classification (`make_classification`)
     - Synthetic Continuous Regression (`make_regression`)
   - Custom CSV dataset upload with shape/missing-value summary.
   - Model types supported:
     - **Classification:** Logistic Regression, Random Forest (Accuracy, Precision, Recall, Weighted F1, Confusion Matrix Heatmap, Feature Importances).
     - **Regression:** Linear Regression, Random Forest Regressor (MAE, RMSE, R², Actual vs. Predicted scatter, Feature Importances).
     - **Clustering:** K-Means with Silhouette coefficient and 2D PCA projection scatter.
     - **Parameter Sweep:** Controlled sweeps across model hyperparameters (e.g. Random Forest `n_estimators`, Logistic Regression `C`, tree `max_depth`) with comparative performance curves.

5. **📊 Results & Scientific Insights:**
   - Plain-language synthesis of empirical findings.
   - Explicit hypothesis evaluation: **Supported**, **Not Supported**, or **Inconclusive**.
   - Identifies underlying mechanisms and lists empirical caveats (sample size limits, non-causality notes).

6. **💡 Actionable Next-Experiment Recommendations:**
   - Recommends 1 to 3 concrete follow-up experiments (What to change, What to keep constant, What to measure, Scientific rationale, Hypothesis impact).
   - One-click launcher into the predefined runner for immediate execution.

7. **📜 Experiment History & Cross-Run Comparison:**
   - Complete audit trail stored in SQLite.
   - Side-by-side comparative table and Plotly bar charts contrasting metrics across models (e.g. Logistic Regression vs. Random Forest).
   - CSV export of all experimental runs.

---

## 🛠️ Technology Stack

- **Language:** Python 3.11+ (Compatible with Python 3.11 – 3.14)
- **Application Framework:** Streamlit
- **Local LLM Runtime (Default):** Ollama (`http://localhost:11434`) running `qwen2.5:7b-instruct` or `qwen2.5:3b-instruct` with JSON Schema enforcement (`requests`)
- **Optional Cloud LLM:** Google Gemini via `google-genai` (disabled unless `LLM_PROVIDER=gemini`)
- **Document Extraction:** PyMuPDF (`pymupdf`)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Retrieval:** `faiss-cpu` (with cosine fallback)
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`
- **Data Visualizations:** `plotly`
- **Persistence:** `sqlite3`
- **Configuration:** `python-dotenv`, `pydantic`
- **Testing:** `pytest`

---

## 📦 Installation & Setup

### 1. Install Ollama & Pull Local Qwen Models (Offline Default)
1. Install Ollama for Windows/macOS/Linux from <https://ollama.com/download> (or `winget install Ollama.Ollama` on Windows).
2. Download the default 7B model (and optional faster 3B model for 4 GB VRAM laptops):
```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
```
3. Ensure the local Ollama server is running:
```bash
ollama serve
```
Verify loaded models at any time with:
```bash
ollama list
ollama ps
```

### 2. Clone or Open Project
```bash
cd autolab
```

### 3. Configure Python Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration (`.env`)
No API key is required for default offline mode! Optionally copy `.env.example` to `.env` to customize local settings:
```bash
cp .env.example .env
```
Default local Ollama settings in `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_NUM_CTX=4096
OLLAMA_TEMPERATURE=0.2
OLLAMA_TIMEOUT_SECONDS=360
OLLAMA_KEEP_ALIVE=5m
```
> **Optional Cloud Gemini:** Only used if you explicitly set `LLM_PROVIDER=gemini` and provide `GEMINI_API_KEY`. When `LLM_PROVIDER=ollama` (default), AutoLab makes zero cloud AI requests and never silently switches to Gemini.

> **16 GB RAM / RTX 3050 4 GB VRAM Tip:** `qwen2.5:7b-instruct` (~4.7 GB Q4_K_M) splits across GPU VRAM and system RAM. If 7B generation feels slow or tight alongside large datasets, switch cleanly to `qwen2.5:3b-instruct` directly in the **Dashboard → Local LLM Settings & Diagnostics** panel (which automatically unloads the previous model).

---

## 💻 Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```
Or on Windows:
```cmd
run.bat
```
Then open your browser at `http://localhost:8501`.

---

## 🧪 Faculty Demonstration Walkthrough

Follow this step-by-step path for an end-to-end evaluation:

1. **Step 1: Workspace Setup**
   - Navigate to **Research Workspace** in the sidebar.
   - Click the button **🧪 Load Iris Demo Workspace**.
   - AutoLab pre-populates:
     - **Title:** Iris Species Predictability Benchmark
     - **Question:** *Can a Random Forest classifier predict Iris species with higher accuracy and F1-score than Logistic Regression on standardized botanical measurements?*
     - **Hypothesis:** *Nonlinear decision boundaries in Random Forest will yield superior multi-class classification metrics compared to linear Logistic Regression.*

2. **Step 2: Literature Ingestion (RAG)**
   - Navigate to **Knowledge Library**.
   - Click **📄 Load Sample Research PDF** to ingest the pre-compiled 2-page botanical study (`iris_classification_study.pdf`).
   - Switch to the **Evidence Search** tab and test querying `"Random Forest and Logistic Regression"`.
   - Observe the retrieved chunks with real source filenames, page numbers (Page 1, Page 2), and similarity scores.

3. **Step 3: Generate Experiment Plan**
   - Navigate to **Experiment Planner**.
   - Click **⚡ Generate Structured Plan**.
   - Review the 12-section plan grounded in the retrieved literature. Note the verified citation pills linking back to the uploaded paper.

4. **Step 4: Run Baseline (Logistic Regression)**
   - Navigate to **Run Experiment**.
   - Dataset: Select `Iris Species (Classification)`.
   - Experiment Type: `Classification`.
   - Model: `Logistic Regression`. Split: `0.2`. Seed: `42`.
   - Click **🚀 Run Predefined Experiment**.
   - Observe real computed metrics (Accuracy ~96.7%, F1-score ~0.967) and the interactive Plotly Confusion Matrix.

5. **Step 5: Run Proposed Model (Random Forest)**
   - In **Run Experiment**, change Model to `Random Forest`.
   - Click **🚀 Run Predefined Experiment**.
   - Inspect the Confusion Matrix and Feature Importance chart (highlighting Petal Length/Width dominance).

6. **Step 6: Interpret Results**
   - Click **🧠 Interpret Results & Generate Next Steps** (or open **Results & Insights**).
   - Click **Generate Scientific Interpretation**.
   - AutoLab assesses whether the hypothesis is **Supported** based on empirical metrics, explains reasons, and details threats to validity (e.g. small sample size n=150).

7. **Step 7: Execute Recommended Follow-Up**
   - Under **Concrete Follow-Up Experiment Recommendations**, view the suggested parameter sweep on Random Forest tree estimators.
   - Click **⚡ Configure & Run**.
   - In **Run Experiment**, select `Parameter Sweep` with `n_estimators` across `[10, 25, 50, 100, 200]`.
   - Click **Run Predefined Experiment** and view the parameter sensitivity curve.

8. **Step 8: Audit Trail & History Comparison**
   - Navigate to **Experiment History**.
   - Select the Logistic Regression and Random Forest runs.
   - View the side-by-side comparative table and Plotly benchmark bar chart.
   - Download the experiment log as CSV.

---

## 🔒 Scientific Integrity & Safety Guarantees

- **No Code Execution:** AutoLab does not execute user-provided or LLM-generated code. All experiments run within fixed, parameterized scikit-learn pipelines.
- **Zero Hallucination of Metrics:** All confusion matrices, R² values, F1-scores, and silhouette coefficients are computed directly from executed data.
- **Strict Evidence Attribution:** Every displayed citation references an actual uploaded document and page.
- **Scientific Humility:** AutoLab never asserts that a statistical model "proves" a scientific theory; it explicitly communicates correlation, caveats, and sample size limitations.

---

## 🧪 Automated Test Suite

Run the full pytest test suite:
```bash
python -m pytest tests/ -v
```

Tests included:
- `test_ollama_provider.py`: Ollama connectivity, JSON schema validation, offline RAG citation checks, comparative/interpretation routing, and no-silent-Gemini-fallback enforcement.
- `test_comparative_analysis.py`: Cross-run comparative analysis, hypothesis assessment, and duplicate recommendation prevention.
- `test_quota_handling.py`: Optional Gemini error detection and deterministic rule-based fallback.
- `test_database.py`: Workspaces, documents, plans, and runs CRUD lifecycle.
- `test_chunking.py`: Chunk boundary splitting and overlap preservation.
- `test_retrieval.py`: Dense embedding generation, FAISS indexing, and cosine retrieval.
- `test_experiments.py`: Classification, regression, clustering, and parameter sweep pipelines.
- `test_faculty_demo.py`: Complete 10-step end-to-end faculty benchmark scenario.

---

## 📝 Known Limitations & Future Work

- **Hardware Scope:** Experiments are CPU-bounded single-machine workflows designed for tabular and tabular-convertible data.
- **Model Palette:** Currently implements fundamental discriminative and generative baselines (Linear Models, Random Forests, K-Means). Future expansions can introduce Gradient Boosting (LightGBM/XGBoost) and ANOVA significance tests.
- **Document Formats:** Supports PDF, Markdown, and TXT; DOCX support can be added with `python-docx`.
