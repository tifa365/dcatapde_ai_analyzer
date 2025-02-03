# **How to Use This Notebook**

This notebook analyzes open government dataset metadata with a Large Language Model (LLM) and saves the enriched results to an Excel file. Below is a brief guide on how to run it.

---

### 1. Prerequisites

1. **Environment Setup**  
   - Make sure you have a Python 3 environment with the necessary libraries installed. You can install all required packages from `requirements.txt` using:
     ```bash
     pip install -r requirements.txt
     ```

2. **OpenAI API Key**  
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY=YOUR_OPENAI_KEY
     ```
   - The notebook will check for `OPENAI_API_KEY` and will raise an error if it’s not found.

---

### 2. Running the Notebook

1. **Navigate to the `/notebooks` directory**  
   ```bash
   cd notebooks
   ```
2. **Open the `llm_assessment.ipynb` (or similarly named) notebook**  
   - Select the notebook to run it step by step.

---

### 3. What the Notebook Does

1. **Retrieve Metadata**  
   - Pulls a full list of datasets from the Berlin open data API (CKAN) and saves it as a Parquet file (`metadata.parquet`).

2. **Filter or Inspect Data (Optional)**  
   - You can optionally filter the metadata by tags, publisher, or any other criteria before analysis.

3. **Semantic Analysis with LLM**  
   - For each dataset (title, description, etc.), the notebook calls an OpenAI model to generate semantic insights:
     - `content_score`, `context_score`, `quality_score`, `spacial_score`
     - Human-readable text assessments for content, context, quality, and spatial coverage.

4. **Combine and Save**  
   - The original metadata is combined with the new LLM-generated columns into a single DataFrame.
   - This final enriched DataFrame is saved to an Excel file in the `/_results` folder, named with a current date stamp (e.g., `metadata_analysis_YYYYMMDD.xlsx`).

---

### 4. Adjusting the Number of Datasets

- If you want to test only on a small subset to save time or API credits, update the line:
  ```python
  num_datasets_to_analyze = 10  # or None for all datasets
  ```
  This slices the DataFrame so only a certain number of rows are sent to the LLM.

---

### 5. Output

- **Location:**  
  An Excel file (e.g., `metadata_analysis_20250101.xlsx`) in `../_results`.
- **Contents:**  
  Original metadata columns + LLM scores (`content_score`, `context_score`, `quality_score`, `spacial_score`) + textual summaries.

