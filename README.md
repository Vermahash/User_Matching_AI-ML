
#  AI Real Estate Agent (Hybrid Recommendation System)

> **Context:** Technical Case Study Implementation  
> **Author:** Harsh Verma  
> **Date:** November 2025  
> **License:** Provided solely for evaluation purposes.

-----

##  Overview

This project implements an intelligent **Real Estate Matchmaker** that goes beyond simple SQL filters. It uses a **Hybrid Recommendation Engine** to balance quantitative constraints (Budget, Bedrooms) with qualitative desires ("Vibe," Architectural Style, Atmosphere).

The system allows users to search for properties using natural language (e.g., *"I want a modern minimalist home with a quiet backyard"*) and receive curated, explainable recommendations powered by a local Large Language Model (LLM).

*(Note: You can add the flowchart image we generated here)*

-----

##  Key Features

  *  Semantic Search (The "Vibe" Check):** Uses `Sentence-Transformers` and **FAISS** vector search to find properties that match the *meaning* of a user's description, not just keyword overlaps.
  *  Logic Layer (The "Wallet" Check):** A deterministic Python layer that applies strict penalties for budget violations and missing bedrooms, ensuring results are realistic.
  *  Smart Scoring System:** Decoupled scoring logic that weighs Vibe (60%), Hard Constraints (20%), and Feature Matches (20%) to surface the best "Trade-off" candidates.
  *  Agentic Reasoning:** Integrates **Llama 3.1** (via Ollama) to act as a virtual consultant. It analyzes the data to provide human-like justifications, identifying "Upsell" opportunities (e.g., *"It's $20k over budget, but the extra bedroom justifies the cost"*).
  *  Real-time Visualization:** Generates dynamic Bar Charts and Heatmaps to visualize the "Why" behind every recommendation.

-----

## Tech Stack

  * Backend:** Python 3.10+, FastAPI
  * Vector Database:** FAISS (Facebook AI Similarity Search)
  * ML Models:**
      * Embeddings: `all-MiniLM-L6-v2` (HuggingFace)
      * LLM: `llama3.1:8b-instruct-q4_K_M` (via Ollama)
  * Data Processing: Pandas, NumPy
  * Visualization: Matplotlib, Seaborn
  * Frontend: Vanilla HTML/JS (Embedded for simplicity)

-----

## Installation & Setup

### 1\. Prerequisites

  * Python 3.12+ installed.
  * **Ollama** installed and running locally. [Download Ollama](https://ollama.com/)

### 2\. Clone Repository

```bash
git clone https://github.com/Vermahash/User_Matching_AI-ML.git
cd ai-real-estate-agent
```

### 3\. Install Dependencies

```bash
pip install -r requirements2
```

### 4\. Setup the LLM (Llama 3.1)

Pull the optimized Llama 3.1 model to your local machine:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

*Note: Ensure Ollama is running in the background (`ollama serve`).*

### 5\. Prepare Data

Ensure your Excel dataset is placed correctly:

  * Path: `data/Case Study 2 Data (1).xlsx`
  * *The system will automatically load and index this file on startup.*

-----

## Usage

1.  **Start the Server:**
    ```bash
    python app.py
    ```
2.  **Access the Application:**
    Open your browser and navigate to: `http://localhost:8000`
3.  **Test a Query:**
      * **Budget:** `$850k`
      * **Bedrooms:** `3`
      * **Description:** *"I am looking for a peaceful home with natural light and a renovated kitchen."*
4.  **View Results:**
    The agent will return top 3 properties with specific justifications and generate a Market Analysis dashboard.

-----

## System Architecture

The recommendation engine operates in **4 Distinct Layers**:

1.  **Retrieval Layer (FAISS):** Instantly narrows down the dataset from N to top 50 candidates based on semantic vector similarity.
2.  **Filter Layer (Python):** Applies business logic.
3.  **Scoring Layer (Weighted Avg):** Calculates a composite `System Score` (0-100) combining Vibe, Logic, and Topic keywords.
4.  **Cognitive Layer (Llama 3.1):** The top 5 scored candidates are fed into the LLM with a "Truth Protocol" prompt. The LLM acts as the final decision-maker, generating the persuasive text displayed to the user.

-----

## License

This code is strictly for **educational and evaluation purposes** as part of a technical case study.
Copyright (c) 2025 Harsh Verma. All rights reserved.


