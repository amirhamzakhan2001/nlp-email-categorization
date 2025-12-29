# NLP-Based Email Categorization System (Production-Grade)

A full end-to-end **NLP + MLOps** system that automatically categorizes emails using
unsupervised learning, LLM-based labeling, supervised classification, and incremental
updates — designed for real-world, long-running inboxes.


## 📌 Project Motivation

Email inboxes grow continuously and contain diverse, unstructured text.

Traditional rule-based filters and static classifiers fail because:

- Categories evolve over time
- Manual labeling does not scale
- Models drift as new email types arrive

This project solves these problems by building a **self-adapting email categorization pipeline**
that discovers topics automatically, labels them intelligently, and learns to classify new emails efficiently.



## 🎯 Project Objectives

- Automatically discover email categories (no manual labels)
- Create human-readable cluster labels using LLMs
- Train a supervised model for fast inference
- Support incremental email arrival
- Enable safe periodic retraining
- Maintain production-grade project structure and logging



## 🧠 High-Level Approach

1. **Fetch emails** from Gmail using Gmail API  
2. **Clean & normalize text** (subject + body)  
3. **Generate embeddings** using a transformer model  
4. **Reduce dimensionality** using PCA  
5. **Discover clusters** via hierarchical bisecting K-Means  
6. **Generate labels & summaries** using an LLM  
7. **Train a supervised classifier** using AutoML  
8. **Run incremental inference** for new emails  
9. **Periodically retrain** to handle concept drift  



## 🏗️ System Architecture
```
Gmail API
↓
Text Cleaning & Normalization
↓
Embeddings (Qwen)
↓
PCA (256-d)
↓
Hierarchical Clustering
↓
LLM-based Labeling
↓
Supervised MLP (AutoML)
↓
Incremental Inference
↓
Master Email Dataset
```


## 📁 Complete Project Structure

```
NLP/
│
├── airflow/                                # Airflow DAGs
│   └── dags/
│       └── gmail_pipeline_dag.py
│
├── artifacts/                              # ML artifacts (gitignored)
│   ├── clustering/
│   ├── embeddings/
│   │   └── qwen/
│   ├── pca/
│   └── supervised/
│
├── data/                                   # Local data (gitignored )
│   ├── raw/
│   │   ├── gmail_cleaned.csv
│   │   └── gmail_new_mail_buffer.csv
│   │
│   ├── processed/
│   │   ├── gmail_cluster_snapshot.csv
│   │   └── gmail_master.csv
│   │
│   └── schema/
│       └── csv_schema.md
│
├── docker/                                # Docker setup
│   ├── .dockerignore
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── gmail_api_secret/                      # gmail auth credentials ( gitignore )
│   ├── gmail_api.json
│   └── token.json
│
├── hf_models/                             # cache 
│
├── models/                                # model defining
│   ├── embedding_models/
│   │   ├── __init__.py
│   │   └── qwen.py
│   │
│   ├── supervised_models/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── mlp.py
│   │
│   └── __init__.py
│
│
├── outputs/                                # Logs, metrics, visualizations
│   ├── logs/
│   ├── metrics/
│   └── visualizations/
│
├── scripts/                                # linux script
│   ├── run_full_pipeline.sh
│   └── run_incremental_pipeline.sh
│
├── src/                                    # main source file
│   │
│   ├── clustering/                         # Unsupervised learning
│   │   ├── __init__.py
│   │   ├── cluster_full.py
│   │   ├── cluster_tree_builder.py
│   │   ├── data_loader.py
│   │   └── kmeans_recursive.py
│   │
│   ├── common/                           	  # Shared utilities
│   │   ├── logging.py
│   │   └── paths.py
│   │
│   ├── config/								  # central configeration
│   │   ├── __init__.py
│   │   ├── clustering_config.py
│   │   ├── embedding_config.py
│   │   ├── fetch_config.py
│   │   ├── inference_config.py
│   │   ├── pipeline_config.py
│   │   └── supervised_config.py
│   │
│   ├── data_ops/								# merging csv file
│   │   └── csv_merge.py
│   │
│   ├── embedding/								# Embeddings & PCA
│   │   ├── __init__.py
│   │   ├── csv_io.py
│   │   ├── embed_full.py
│   │   ├── embed_incremental.py
│   │   ├── embedder.py
│   │   ├── model_loader.py
│   │   └── pca_manager.py
│   │
│   ├── fetch_gmail/							# Gmail ingestion
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── body_cleaner.py
│   │   ├── csv_writer.py
│   │   ├── email_parser.py
│   │   ├── fetch.py
│   │   ├── fetch_and_clean_pipeline.py
│   │   ├── fetch_latest_pipeline.py
│   │   ├── incremental.py
│   │   └── subject_cleaner.py
│   │
│   ├── graph/							 		# root to leaf cluster path
│   │   ├── __init__.py
│   │   └── label_path.py
│   │
│   ├── inference/								# Incremental inference
│   │   ├── __init__.py
│   │   ├── inf_data_loader.py
│   │   ├── infer_supervised.py
│   │   ├── inferencer.py
│   │   ├── label_path.py
│   │   └── model_loader.py
│   │
│   ├── labeling/							# LLM-based labeling
│   │   ├── __init__.py
│   │   ├── checkpoint_store.py
│   │   ├── label_freezer.py
│   │   ├── label_generator.py
│   │   ├── llm_client.py
│   │   └── prompt_builder.py
│   │
│   ├── outputs/							# outputs function
│   │   ├── metrics_writer.py
│   │   └── visualization_utils.py
│   │
│   ├── pipelines/								# Orchestration
│   │   ├── full_pipeline.py
│   │   └── incremental_pipeline.py
│   │
│   ├── supervised/								# Supervised learning
│   │   ├── __init__.py
│   │   ├── automl.py
│   │   ├── dataset_builder.py
│   │   ├── evaluator.py
│   │   ├── sup_data_loader.py
│   │   ├── train_supervised_full.py
│   │   └── trainer.py
│   │
│   ├── utils/								 # deleting saved artifacts
│   │   └── reset_artifacts.py
│   │
│   └── visualization/							# visualizing metrices
│       ├── cluster_depths.py
│       ├── cluster_sizes.py
│       ├── embedding_map.py
│       ├── generate_visualizations.py
│       └── sse_metrics.py
│
├── .env
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt


```

## 📊 Data Flow

### Full Pipeline (First Run / Retraining)

1. Fetch all emails → `gmail_cleaned.csv`
2. Embed & PCA → `gmail_cluster_snapshot.csv`
3. Hierarchical clustering + LLM labeling
4. Supervised AutoML training
5. Merge → `gmail_master.csv`


### Incremental Pipeline (Regular Updates)

1. Fetch new emails → `gmail_new_mail_buffer.csv`
2. Embed + PCA transform
3. Supervised inference
4. Append to `gmail_master.csv`
5. Delete buffer CSV



## 🔁 Pipeline Modes

Controlled using environment variable `PIPELINE_MODE`:

| Mode          |       Purpose             |
|---------------|---------------------------|
| `full`        | First-time execution      |
| `incremental` | Regular updates           |
| `retrain`     | Full ML reset and rebuild |

Example:

```
PIPELINE_MODE=retrain python -m src.pipelines.full_pipeline
```



## 🤖 Models Used
	•	Embeddings: Transformer-based (Qwen)
	•	Dimensionality Reduction: PCA (256 dimensions)
	•	Clustering: Hierarchical Bisecting K-Means
	•	Labeling: LLM-based summarization
	•	Classifier: MLP with AutoML architecture search



## 📈 MLOps Principles Applied
	•	Immutable artifacts
	•	Atomic CSV writes
	•	Incremental-safe pipelines
	•	Artifact reset for retraining
	•	Unified logging
	•	AutoML-based model selection



## 🔐 Environment Setup

	1.	Copy environment file:
	```
    	cp .env.example .env
	```
    2.	Add required keys:
	```
		•	GEMINI_API_KEY
	```


## 🛠️ Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 👨‍🎓 Academic Context

This project was developed as part of an **MSc in Artificial Intelligence & Machine Learning**
with emphasis on **NLP, clustering, MLOps, and production systems.**

⸻

### 👤 Author

Amir Hamza Khan
MSc AI & ML
Jamia Millia Islamia

⸻

### 📜 License

For academic and research use only.


#### 🎯 Final Status

✅ README finalized  
✅ No edits required  
✅ Matches your full implementation  
✅ Suitable for GitHub + submission  

If you want next:
- README diagrams (PNG / SVG)
- Architecture flowcharts
- Interview explanation script
- Viva defense Q&A
- 
Just tell me 👍
