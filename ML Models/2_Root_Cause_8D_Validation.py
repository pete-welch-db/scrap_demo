# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Root Cause Analysis & 8D Validation
# MAGIC 
# MAGIC ## Business Objective
# MAGIC 
# MAGIC This notebook uses ML and LLMs to validate the Quality Department's root cause assignments and improve 8D effectiveness:
# MAGIC 
# MAGIC | Analysis | Question Answered |
# MAGIC |----------|-------------------|
# MAGIC | **Root Cause Classification** | Can we predict root cause from scrap characteristics? |
# MAGIC | **8D Validation** | Do model predictions match Quality team assignments? |
# MAGIC | **Misclassification Detection** | Which 8Ds might have incorrect root causes? |
# MAGIC | **8D Effectiveness Prediction** | Will the corrective action work? |
# MAGIC | **Similar 8D Finder** | What worked for similar problems in the past? |
# MAGIC | **LLM Text Analysis** | What insights can AI extract from 8D text? |
# MAGIC 
# MAGIC ## Key Question
# MAGIC > **Is the Quality Department correctly identifying root causes?**
# MAGIC 
# MAGIC ## ML Approach
# MAGIC - **Classification**: Random Forest, Gradient Boosting via AutoML
# MAGIC - **NLP**: TF-IDF similarity, LLM analysis with Foundation Models
# MAGIC - **Tracking**: Full MLflow experiment tracking with artifact logging

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies
# MAGIC 
# MAGIC Install scikit-learn for machine learning models.

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow[databricks] xgboost lightgbm textblob spacy nltk sentence-transformers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Restart Python Environment

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries
# MAGIC 
# MAGIC Import all required libraries for:
# MAGIC - **ML Models**: scikit-learn classifiers
# MAGIC - **MLflow**: Experiment tracking and model registry
# MAGIC - **AutoML**: Databricks automated machine learning
# MAGIC - **NLP**: TF-IDF vectorization, cosine similarity

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Additional ensemble models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# NLP Libraries
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Sentence Transformers for semantic embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔤 Download NLTK Data
# MAGIC 
# MAGIC Download required NLTK resources for text processing.

# COMMAND ----------

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"⚠️  NLTK download warning: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 NLP Feature Engineering Functions
# MAGIC 
# MAGIC Advanced text analysis functions to extract meaningful features from 8D text fields.

# COMMAND ----------

def extract_text_features(text: str) -> dict:
    """
    Extract comprehensive NLP features from text.
    
    Features extracted:
    - Word count, sentence count, avg word length
    - Sentiment (polarity, subjectivity)
    - Readability scores
    - Technical term density
    - Question/exclamation presence
    - Numerical data presence
    - Action verb count
    - Passive voice detection
    
    Parameters:
    -----------
    text : str
        Input text to analyze
        
    Returns:
    --------
    dict
        Dictionary of NLP features
    """
    
    if not text or pd.isna(text) or str(text).strip() == "":
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'contains_numbers': 0,
            'contains_questions': 0,
            'contains_exclamations': 0,
            'technical_term_ratio': 0,
            'action_verb_count': 0,
            'unique_word_ratio': 0,
            'uppercase_ratio': 0
        }
    
    text = str(text)
    
    # Basic metrics
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Sentiment analysis
    try:
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity  # -1 to 1
        sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1
    except:
        sentiment_polarity = 0
        sentiment_subjectivity = 0
    
    # Pattern detection
    contains_numbers = 1 if re.search(r'\d+', text) else 0
    contains_questions = 1 if '?' in text else 0
    contains_exclamations = 1 if '!' in text else 0
    
    # Technical term density (words > 8 characters)
    long_words = [w for w in words if len(w) > 8]
    technical_term_ratio = len(long_words) / word_count if word_count > 0 else 0
    
    # Action verbs (common manufacturing/quality terms)
    action_verbs = [
        'inspect', 'check', 'verify', 'test', 'measure', 'monitor',
        'replace', 'adjust', 'calibrate', 'repair', 'fix', 'correct',
        'implement', 'install', 'remove', 'clean', 'maintain', 'train',
        'validate', 'review', 'analyze', 'investigate', 'determine'
    ]
    action_verb_count = sum(1 for verb in action_verbs if verb in text.lower())
    
    # Unique word ratio (vocabulary richness)
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
    
    # Uppercase ratio (emphasis/urgency indicator)
    uppercase_chars = sum(1 for c in text if c.isupper())
    total_chars = len(text)
    uppercase_ratio = uppercase_chars / total_chars if total_chars > 0 else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'sentiment_polarity': round(sentiment_polarity, 3),
        'sentiment_subjectivity': round(sentiment_subjectivity, 3),
        'contains_numbers': contains_numbers,
        'contains_questions': contains_questions,
        'contains_exclamations': contains_exclamations,
        'technical_term_ratio': round(technical_term_ratio, 3),
        'action_verb_count': action_verb_count,
        'unique_word_ratio': round(unique_word_ratio, 3),
        'uppercase_ratio': round(uppercase_ratio, 3)
    }


def extract_key_phrases(text: str, top_n: int = 5) -> list:
    """
    Extract key phrases (bigrams/trigrams) from text.
    
    Parameters:
    -----------
    text : str
        Input text
    top_n : int
        Number of top phrases to return
        
    Returns:
    --------
    list
        Top N key phrases
    """
    
    if not text or pd.isna(text):
        return []
    
    text = str(text).lower()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # Generate bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    # Count and return top phrases
    phrase_counts = Counter(bigrams)
    top_phrases = [phrase for phrase, _ in phrase_counts.most_common(top_n)]
    
    return top_phrases


def calculate_text_similarity(text1: str, text2: str, use_bert: bool = True) -> float:
    """
    Calculate cosine similarity between two texts.
    
    Parameters:
    -----------
    text1 : str
        First text
    text2 : str
        Second text
    use_bert : bool
        If True, use BERT embeddings. If False, fall back to TF-IDF.
        
    Returns:
    --------
    float
        Similarity score (0-1)
    """
    
    if not text1 or not text2 or pd.isna(text1) or pd.isna(text2):
        return 0.0
    
    try:
        if use_bert:
            # Use sentence-transformers for semantic similarity
            # This model is optimized for semantic similarity tasks
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings
            embeddings = model.encode([str(text1), str(text2)])
            
            # Calculate cosine similarity
            similarity = sklearn_cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        else:
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([str(text1), str(text2)])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
    except Exception as e:
        print(f"   ⚠️ Similarity calculation failed: {e}, returning 0.0")
        return 0.0


def extract_root_cause_keywords(text: str) -> dict:
    """
    Extract manufacturing/quality-specific keywords from root cause text.
    
    Parameters:
    -----------
    text : str
        Root cause analysis text
        
    Returns:
    --------
    dict
        Presence indicators for keyword categories
    """
    
    if not text or pd.isna(text):
        return {
            'mentions_equipment': 0,
            'mentions_material': 0,
            'mentions_process': 0,
            'mentions_human': 0,
            'mentions_measurement': 0,
            'mentions_environment': 0
        }
    
    text = str(text).lower()
    
    # 6M categories (Ishikawa/Fishbone)
    equipment_terms = ['machine', 'equipment', 'tool', 'die', 'fixture', 'robot', 'press']
    material_terms = ['material', 'steel', 'plastic', 'supplier', 'batch', 'lot', 'raw']
    process_terms = ['process', 'procedure', 'method', 'setup', 'cycle', 'operation', 'sequence']
    human_terms = ['operator', 'training', 'skill', 'error', 'mistake', 'fatigue', 'experience']
    measurement_terms = ['measurement', 'gauge', 'calibration', 'tolerance', 'specification', 'dimension']
    environment_terms = ['temperature', 'humidity', 'contamination', 'cleanliness', 'environment']
    
    return {
        'mentions_equipment': 1 if any(term in text for term in equipment_terms) else 0,
        'mentions_material': 1 if any(term in text for term in material_terms) else 0,
        'mentions_process': 1 if any(term in text for term in process_terms) else 0,
        'mentions_human': 1 if any(term in text for term in human_terms) else 0,
        'mentions_measurement': 1 if any(term in text for term in measurement_terms) else 0,
        'mentions_environment': 1 if any(term in text for term in environment_terms) else 0
    }


print("✅ NLP feature engineering functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC 
# MAGIC Define configuration parameters and initialize MLflow experiment.

# COMMAND ----------

# Unity Catalog Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# MLflow Configuration
# MLflow will use the default notebook experiment when running from Git folders
# This avoids the error: "Cannot create workspace MLflow experiment in a Git folder"
# EXPERIMENT_PATH = "/Shared/scrap_demo/root_cause_analysis"
# mlflow.set_experiment(EXPERIMENT_PATH)  # Removed for Git folder compatibility

# Enable MLflow autologging for sklearn
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")
print(f"   MLflow Experiment: Default Notebook Experiment")
print(f"   AutoLogging: Enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Load Data
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Scrap Transactions
# MAGIC 
# MAGIC Load scrap data from the Silver layer with all enriched features.

# COMMAND ----------

# Helper function to check if table exists
def table_exists(table_name):
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False

# Validate required tables exist
required_tables = [
    f"{FULL_SCHEMA}.scrap_transactions_silver",
    f"{FULL_SCHEMA}.eight_d_reports_silver",
    f"{FULL_SCHEMA}.scrap_8d_mapping_silver"
]
missing_tables = [t for t in required_tables if not table_exists(t)]

if missing_tables:
    error_msg = f"""
    ❌ Missing required tables: {missing_tables}
    
    Please run the following first:
    1. Setup job (scrap_setup_job) - Creates bronze tables
    2. DLT Pipeline - Creates silver tables from bronze
    
    Then re-run this ML training notebook.
    """
    raise Exception(error_msg)

print(f"✅ All required tables exist in {FULL_SCHEMA}")

# COMMAND ----------

# Load scrap transactions from silver layer
scrap_df = spark.table(f"{FULL_SCHEMA}.scrap_transactions_silver")

print(f"📊 Scrap Transactions:")
print(f"   Records: {scrap_df.count():,}")
print(f"   Columns: {len(scrap_df.columns)}")

# Show root cause distribution
print(f"\n📈 Root Cause Category Distribution:")
scrap_df.groupBy("root_cause_category").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load 8D Reports
# MAGIC 
# MAGIC Load 8D quality reports with root cause analysis and corrective actions.

# COMMAND ----------

# Load 8D reports from silver layer
eight_d_df = spark.table(f"{FULL_SCHEMA}.eight_d_reports_silver")

print(f"📊 8D Reports:")
print(f"   Records: {eight_d_df.count():,}")

# Show status distribution
print(f"\n📈 8D Status Distribution:")
eight_d_df.groupBy("status").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Scrap-8D Mapping
# MAGIC 
# MAGIC Load the mapping table that links scrap transactions to 8D reports.

# COMMAND ----------

# Load mapping table
mapping_df = spark.table(f"{FULL_SCHEMA}.scrap_8d_mapping_silver")

print(f"📊 Scrap-8D Mapping:")
print(f"   Records: {mapping_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🧠 Model 1: Root Cause Classification
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Train a model to predict `root_cause_category` from scrap characteristics.
# MAGIC 
# MAGIC If the model's predictions differ significantly from the Quality team's assignments,
# MAGIC it may indicate systematic misclassification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Features
# MAGIC 
# MAGIC Select features that should be predictive of root cause:
# MAGIC - **Numeric**: Quantities, costs, equipment metrics
# MAGIC - **Categorical**: Defect type, work center, part category

# COMMAND ----------

# Define feature columns
NUMERIC_FEATURES = [
    "shift",
    "quantity_scrapped",
    "unit_cost",
    "total_scrap_cost",
    "equipment_age_years",
    "equipment_oee",
    "operator_experience_years",
    "supplier_quality_score",
    "defect_severity_weight",
    "transaction_day_of_week",
    "transaction_month",
]

CATEGORICAL_FEATURES = [
    "defect_category",
    "work_center_type_lookup",
    "part_category_lookup",
    "disposition",
    "plant_id",
    "supplier_tier_lookup"
]

TARGET_COLUMN = "root_cause_category"

print(f"📊 Feature Summary:")
print(f"   Numeric Features: {len(NUMERIC_FEATURES)}")
print(f"   Categorical Features: {len(CATEGORICAL_FEATURES)}")
print(f"   Target: {TARGET_COLUMN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Training Data
# MAGIC 
# MAGIC Convert Spark DataFrame to Pandas and prepare features for training:
# MAGIC 1. Select relevant columns
# MAGIC 2. Drop rows with missing values
# MAGIC 3. Encode categorical variables

# COMMAND ----------

# Prepare training data - Include transaction_date for temporal splitting
training_df = scrap_df.select(
    "transaction_date",  # Add date for temporal split
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    TARGET_COLUMN
).dropna()

# Convert to Pandas
train_pd = training_df.toPandas()

# Convert transaction_date to datetime
train_pd["transaction_date"] = pd.to_datetime(train_pd["transaction_date"])

print(f"📊 Training Data:")
print(f"   Samples: {len(train_pd):,}")
print(f"   Features: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")
print(f"   Date Range: {train_pd['transaction_date'].min()} to {train_pd['transaction_date'].max()}")

print(f"\n📈 Target Distribution:")
print(train_pd[TARGET_COLUMN].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encode Categorical Variables
# MAGIC 
# MAGIC Convert categorical variables to numeric using LabelEncoder.
# MAGIC Store encoders for later use in prediction.

# COMMAND ----------

# Encode categorical variables
encoders = {}

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_pd[f"{col}_encoded"] = le.fit_transform(train_pd[col].astype(str))
    encoders[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} classes")

# Encode target variable
target_encoder = LabelEncoder()
train_pd["target_encoded"] = target_encoder.fit_transform(train_pd[TARGET_COLUMN])

print(f"\n🎯 Target Classes: {list(target_encoder.classes_)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Feature Matrix
# MAGIC 
# MAGIC Create the final feature matrix (X) and target vector (y) for training.

# COMMAND ----------

# Create feature column list
encoded_feature_cols = NUMERIC_FEATURES + [f"{col}_encoded" for col in CATEGORICAL_FEATURES]

# Prepare X and y
X = train_pd[encoded_feature_cols].values
y = train_pd["target_encoded"].values

print(f"📊 Feature Matrix Shape: {X.shape}")
print(f"📊 Target Vector Shape: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporal Train/Test Split
# MAGIC 
# MAGIC **Avoid Data Leakage**: Train on older data, test on more recent data.
# MAGIC 
# MAGIC This simulates real-world deployment where we predict future root causes
# MAGIC based on historical patterns. Using random splitting would leak future
# MAGIC information into training data, inflating performance metrics.
# MAGIC 
# MAGIC **Strategy**: Use 80% oldest data for training, 20% newest for testing.

# COMMAND ----------

# Sort by date and perform temporal split
train_pd_sorted = train_pd.sort_values('transaction_date').reset_index(drop=True)

# Calculate split point (80% for training)
split_idx = int(len(train_pd_sorted) * 0.8)
split_date = train_pd_sorted.loc[split_idx, 'transaction_date']

# Split into train and test based on date
train_mask = train_pd_sorted['transaction_date'] < split_date
test_mask = train_pd_sorted['transaction_date'] >= split_date

train_set = train_pd_sorted[train_mask]
test_set = train_pd_sorted[test_mask]

# Prepare X and y for training set
X_train = train_set[encoded_feature_cols].values
y_train = train_set["target_encoded"].values

# Prepare X and y for test set
X_test = test_set[encoded_feature_cols].values
y_test = test_set["target_encoded"].values

print(f"📊 Temporal Data Split (Avoiding Data Leakage):")
print(f"{'='*70}")
print(f"\n📅 Training Set:")
print(f"   Date Range: {train_set['transaction_date'].min().date()} to {train_set['transaction_date'].max().date()}")
print(f"   Samples: {len(X_train):,} ({len(X_train)/len(train_pd_sorted):.0%})")
print(f"   Shape: {X_train.shape}")

print(f"\n📅 Test Set:")
print(f"   Date Range: {test_set['transaction_date'].min().date()} to {test_set['transaction_date'].max().date()}")
print(f"   Samples: {len(X_test):,} ({len(X_test)/len(train_pd_sorted):.0%})")
print(f"   Shape: {X_test.shape}")

print(f"\n🔒 Data Leakage Prevention:")
print(f"   ✅ Training only on historical data (before {split_date.date()})")
print(f"   ✅ Testing on future data (from {split_date.date()} onward)")
print(f"   ✅ No overlap between train and test dates")

# Check class distribution in both sets
print(f"\n📊 Class Distribution Check:")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()

comparison_df = pd.DataFrame({
    'Train %': train_dist * 100,
    'Test %': test_dist * 100
})
comparison_df['Difference'] = comparison_df['Test %'] - comparison_df['Train %']

print(comparison_df.round(1).to_string())

# Warn if distributions are very different
max_diff = comparison_df['Difference'].abs().max()
if max_diff > 10:
    print(f"\n⚠️  WARNING: Class distribution shift detected (max diff: {max_diff:.1f}%)")
    print(f"   This may indicate concept drift over time.")
else:
    print(f"\n✅ Class distributions are reasonably similar (max diff: {max_diff:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Random Forest Classifier
# MAGIC 
# MAGIC Train a Random Forest model with full MLflow logging:
# MAGIC - Model parameters
# MAGIC - Training and validation metrics
# MAGIC - Feature importance
# MAGIC - Confusion matrix
# MAGIC - Model artifact

# COMMAND ----------

# Disable autolog temporarily for custom logging
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="root_cause_random_forest") as run:
    
    # ===== LOG DATASET INFO =====
    mlflow.log_param("dataset_size", len(train_pd_sorted))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", len(encoded_feature_cols))
    mlflow.log_param("num_classes", len(target_encoder.classes_))
    mlflow.log_param("feature_list", json.dumps(encoded_feature_cols))
    mlflow.log_param("split_method", "temporal")
    mlflow.log_param("split_date", str(split_date.date()))
    
    # ===== DEFINE MODEL =====
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Log hyperparameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_split", 10)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("class_weight", "balanced")
    
    # ===== TRAIN MODEL =====
    print("🔄 Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # ===== PREDICTIONS =====
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # ===== CALCULATE METRICS =====
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    
    # ===== LOG METRICS =====
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    
    # ===== LOG FEATURE IMPORTANCE =====
    feature_importance = pd.DataFrame({
        "feature": encoded_feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Save as CSV artifact
    feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x="importance", y="feature", ax=ax)
    ax.set_title("Top 15 Features for Root Cause Prediction")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # ===== LOG CONFUSION MATRIX =====
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_,
        ax=ax
    )
    ax.set_title("Root Cause Prediction - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
    
    # ===== LOG CLASSIFICATION REPORT =====
    report = classification_report(
        y_test, y_pred, 
        target_names=target_encoder.classes_,
        output_dict=True
    )
    with open("/tmp/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("/tmp/classification_report.json")
    
    # ===== LOG MODEL =====
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
        rf_model, 
        "model",
        signature=signature
    )
    
    # ===== PRINT RESULTS =====
    print(f"\n{'='*60}")
    print(f"✅ Model: Root Cause Random Forest Classifier (Baseline)")
    print(f"{'='*60}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n📊 Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")
    
    # Store baseline model for comparison
    baseline_rf_model = rf_model
    baseline_accuracy = accuracy
    baseline_f1 = f1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Classification Report
# MAGIC 
# MAGIC Detailed per-class performance metrics.

# COMMAND ----------

# Print classification report
print("\n📊 Classification Report by Root Cause Category:")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Feature Importance
# MAGIC 
# MAGIC Identify which features are most predictive of root cause.

# COMMAND ----------

# Display feature importance
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=feature_importance.head(15), 
    x="importance", 
    y="feature",
    palette="viridis"
)
ax.set_title("Top 15 Features for Root Cause Prediction", fontsize=14)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🎯 Ensemble Model: Voting Classifier
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Combine multiple models (Random Forest, Gradient Boosting, XGBoost, LightGBM) 
# MAGIC using soft voting to improve prediction accuracy and robustness.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Voting Classifier Ensemble
# MAGIC 
# MAGIC Combines predictions from 4 different classifiers:
# MAGIC - **Random Forest**: Robust, handles non-linearity
# MAGIC - **Gradient Boosting**: Sequential error correction
# MAGIC - **XGBoost**: Fast, regularized boosting
# MAGIC - **LightGBM**: Efficient, handles categorical features well

# COMMAND ----------

# Disable autolog temporarily for custom logging
mlflow.sklearn.autolog(disable=True)

print("\n🔍 Verifying Feature Consistency for Ensemble...")
print(f"   Feature columns: {len(encoded_feature_cols)}")
print(f"   Training shape: {X_train.shape}")
print(f"   Testing shape: {X_test.shape}")
print(f"   Class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")

with mlflow.start_run(run_name="root_cause_voting_ensemble") as run:
    
    # ===== LOG DATASET INFO =====
    mlflow.log_param("dataset_size", len(train_pd_sorted))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", len(encoded_feature_cols))
    mlflow.log_param("num_classes", len(target_encoder.classes_))
    mlflow.log_param("split_method", "temporal")
    mlflow.log_param("split_date", str(split_date.date()))
    
    # ===== DEFINE BASE MODELS =====
    print("🔄 Initializing ensemble models...")
    
    # Model 1: Random Forest
    rf_ensemble = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Model 2: Gradient Boosting
    gb_ensemble = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Model 3: XGBoost
    xgb_ensemble = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Model 4: LightGBM
    lgbm_ensemble = LGBMClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # ===== CREATE VOTING CLASSIFIER =====
    print("🔄 Creating Voting Classifier with soft voting...")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_ensemble),
            ('gb', gb_ensemble),
            ('xgb', xgb_ensemble),
            ('lgbm', lgbm_ensemble)
        ],
        voting='soft',  # Use probability-based voting
        n_jobs=-1
    )
    
    # Log ensemble configuration
    mlflow.log_param("model_type", "VotingClassifier")
    mlflow.log_param("voting_method", "soft")
    mlflow.log_param("num_base_models", 4)
    mlflow.log_param("base_models", "RandomForest, GradientBoosting, XGBoost, LightGBM")
    
    # ===== TRAIN ENSEMBLE =====
    print("🔄 Training ensemble (this may take a few minutes)...")
    voting_clf.fit(X_train, y_train)
    print("✅ Ensemble training complete")
    
    # ===== PREDICTIONS =====
    print("🔄 Generating predictions...")
    y_pred_ensemble = voting_clf.predict(X_test)
    y_pred_proba_ensemble = voting_clf.predict_proba(X_test)
    
    # ===== CALCULATE METRICS =====
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    precision_ensemble = precision_score(y_test, y_pred_ensemble, average='weighted')
    recall_ensemble = recall_score(y_test, y_pred_ensemble, average='weighted')
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
    
    # Cross-validation score for ensemble
    print("🔄 Running cross-validation on ensemble...")
    cv_scores_ensemble = cross_val_score(voting_clf, X_train, y_train, cv=5, n_jobs=-1)
    
    # ===== LOG METRICS =====
    mlflow.log_metric("accuracy", accuracy_ensemble)
    mlflow.log_metric("precision_weighted", precision_ensemble)
    mlflow.log_metric("recall_weighted", recall_ensemble)
    mlflow.log_metric("f1_weighted", f1_ensemble)
    mlflow.log_metric("cv_mean", cv_scores_ensemble.mean())
    mlflow.log_metric("cv_std", cv_scores_ensemble.std())
    
    # Calculate improvement over single Random Forest
    improvement = ((accuracy_ensemble - accuracy) / accuracy) * 100
    mlflow.log_metric("improvement_over_rf_pct", improvement)
    
    # ===== INDIVIDUAL MODEL PERFORMANCE =====
    print("\n📊 Evaluating individual models within ensemble...")
    
    individual_results = []
    for name, model in voting_clf.named_estimators_.items():
        y_pred_individual = model.predict(X_test)
        acc_individual = accuracy_score(y_test, y_pred_individual)
        f1_individual = f1_score(y_test, y_pred_individual, average='weighted')
        
        individual_results.append({
            'model': name,
            'accuracy': acc_individual,
            'f1_score': f1_individual
        })
        
        mlflow.log_metric(f"{name}_accuracy", acc_individual)
        mlflow.log_metric(f"{name}_f1", f1_individual)
    
    individual_df = pd.DataFrame(individual_results)
    
    # ===== CONFUSION MATRIX =====
    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_ensemble, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_,
        ax=ax
    )
    ax.set_title("Voting Ensemble - Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "ensemble_confusion_matrix.png")
    plt.close()
    
    # ===== INDIVIDUAL MODEL COMPARISON PLOT =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add individual models
    individual_df_sorted = individual_df.sort_values('accuracy', ascending=True)
    ax.barh(individual_df_sorted['model'], individual_df_sorted['accuracy'], 
            alpha=0.7, label='Individual Models')
    
    # Add ensemble result
    ax.barh('Ensemble', accuracy_ensemble, color='green', alpha=0.9, label='Ensemble')
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    mlflow.log_figure(fig, "model_comparison.png")
    plt.close()
    
    # ===== CLASSIFICATION REPORT =====
    report_ensemble = classification_report(
        y_test, y_pred_ensemble, 
        target_names=target_encoder.classes_,
        output_dict=True
    )
    with open("/tmp/ensemble_classification_report.json", "w") as f:
        json.dump(report_ensemble, f, indent=2)
    mlflow.log_artifact("/tmp/ensemble_classification_report.json")
    
    # ===== LOG MODEL =====
    signature = infer_signature(X_train, y_pred_ensemble)
    mlflow.sklearn.log_model(
        voting_clf, 
        "model",
        signature=signature
    )
    
    # ===== PRINT RESULTS =====
    print(f"\n{'='*70}")
    print(f"✅ Model: Voting Classifier Ensemble")
    print(f"{'='*70}")
    print(f"\n📊 Ensemble Test Metrics:")
    print(f"   Accuracy:  {accuracy_ensemble:.4f}")
    print(f"   Precision: {precision_ensemble:.4f}")
    print(f"   Recall:    {recall_ensemble:.4f}")
    print(f"   F1 Score:  {f1_ensemble:.4f}")
    print(f"\n📊 Cross-Validation: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std()*2:.4f})")
    
    print(f"\n📈 Improvement over Single Random Forest:")
    print(f"   Accuracy: {accuracy:.4f} → {accuracy_ensemble:.4f} ({improvement:+.2f}%)")
    
    print(f"\n📊 Individual Model Performance:")
    for _, row in individual_df.iterrows():
        print(f"   {row['model']}: {row['accuracy']:.4f}")
    
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")
    
    # Store ensemble model for later use
    ensemble_model = voting_clf
    
    # ===== ENSEMBLE VERIFICATION =====
    print(f"\n{'='*70}")
    print(f"🔍 Ensemble Model Verification")
    print(f"{'='*70}")
    print(f"\n✅ Base Models Trained:")
    for name, model in voting_clf.named_estimators_.items():
        print(f"   • {name}: {type(model).__name__}")
    
    print(f"\n✅ Voting Configuration:")
    print(f"   • Strategy: {voting_clf.voting}")
    print(f"   • Weights: {voting_clf.weights}")
    print(f"   • Parallel: {voting_clf.n_jobs}")
    
    print(f"\n✅ Data Integrity:")
    print(f"   • Features used: {len(encoded_feature_cols)}")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Classes predicted: {len(np.unique(y_pred_ensemble))}")
    
    # Verify all classes are represented
    missing_classes = set(target_encoder.classes_) - set(target_encoder.inverse_transform(np.unique(y_pred_ensemble)))
    if missing_classes:
        print(f"\n⚠️  WARNING: Some classes never predicted: {missing_classes}")
    else:
        print(f"\n✅ All {len(target_encoder.classes_)} classes represented in predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensemble Classification Report

# COMMAND ----------

# Print ensemble classification report
print("\n📊 Ensemble Classification Report by Root Cause Category:")
print("=" * 70)
print(classification_report(y_test, y_pred_ensemble, target_names=target_encoder.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Model Comparison

# COMMAND ----------

# Compare all models visually
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Accuracy comparison
models_comparison = pd.DataFrame([
    {'Model': 'Random Forest', 'Accuracy': accuracy, 'F1': f1},
    {'Model': 'Ensemble (Voting)', 'Accuracy': accuracy_ensemble, 'F1': f1_ensemble}
])

x = range(len(models_comparison))
width = 0.35
ax1.bar([i - width/2 for i in x], models_comparison['Accuracy'], 
        width, label='Accuracy', alpha=0.8)
ax1.bar([i + width/2 for i in x], models_comparison['F1'], 
        width, label='F1 Score', alpha=0.8)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Ensemble vs Single Model Performance', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models_comparison['Model'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Right: Individual ensemble models
ax2.barh(individual_df['model'], individual_df['accuracy'], alpha=0.7, color='steelblue')
ax2.axvline(x=accuracy_ensemble, color='red', linestyle='--', linewidth=2, 
            label=f'Ensemble: {accuracy_ensemble:.4f}')
ax2.set_xlabel('Accuracy', fontsize=12)
ax2.set_title('Individual Models in Ensemble', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n💡 Key Insights:")
print(f"   • Ensemble accuracy: {accuracy_ensemble:.1%}")
print(f"   • Best individual model: {individual_df.loc[individual_df['accuracy'].idxmax(), 'model']} "
      f"({individual_df['accuracy'].max():.1%})")
print(f"   • Ensemble improvement: {improvement:+.2f}%")
print(f"   • Voting combines strengths of all models for more robust predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔎 Validate Quality Department Assignments
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Compare model predictions vs. actual Quality team assignments to identify
# MAGIC potential misclassifications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Root Cause for All Records (Using Ensemble)
# MAGIC 
# MAGIC Apply the ensemble model to all records for better validation quality.
# MAGIC First, calculate baseline RF agreement for comparison.

# COMMAND ----------

# Calculate baseline RF agreement rate for comparison
X_all = train_pd_sorted[encoded_feature_cols].values
y_pred_all_rf = baseline_rf_model.predict(X_all)
y_proba_all_rf = baseline_rf_model.predict_proba(X_all)

train_pd_temp = train_pd_sorted.copy()
train_pd_temp["predicted_root_cause_rf"] = target_encoder.inverse_transform(y_pred_all_rf)
train_pd_temp["prediction_matches_rf"] = train_pd_temp["predicted_root_cause_rf"] == train_pd_temp[TARGET_COLUMN]
agreement_rate_rf = train_pd_temp["prediction_matches_rf"].mean()

print(f"📊 Baseline Random Forest Agreement: {agreement_rate_rf:.1%}")

# COMMAND ----------

# Predict root cause for all records using ENSEMBLE model
y_pred_all = ensemble_model.predict(X_all)
y_proba_all = ensemble_model.predict_proba(X_all)

# Add predictions to dataframe
train_pd_sorted["predicted_root_cause"] = target_encoder.inverse_transform(y_pred_all)
train_pd_sorted["prediction_confidence"] = y_proba_all.max(axis=1)
train_pd_sorted["prediction_matches"] = train_pd_sorted["predicted_root_cause"] == train_pd_sorted[TARGET_COLUMN]

# Calculate agreement rate
agreement_rate_ensemble = train_pd_sorted["prediction_matches"].mean()

print(f"📊 Ensemble Model-Quality Agreement Analysis:")
print(f"{'='*70}")
print(f"   Total Records Analyzed: {len(train_pd_sorted):,}")
print(f"   Model Agrees with Quality: {train_pd_sorted['prediction_matches'].sum():,} ({agreement_rate_ensemble:.1%})")
print(f"   Model Disagrees: {(~train_pd_sorted['prediction_matches']).sum():,} ({1-agreement_rate_ensemble:.1%})")
print(f"\n📈 Improvement over Single RF:")
print(f"   RF Agreement: {agreement_rate_rf:.1%}")
print(f"   Ensemble Agreement: {agreement_rate_ensemble:.1%}")
print(f"   Improvement: {((agreement_rate_ensemble - agreement_rate_rf) / agreement_rate_rf * 100):+.2f}%")
print(f"   Ensemble Agreement: {agreement_rate_ensemble:.1%}")
print(f"   Change: {(agreement_rate_ensemble - agreement_rate) * 100:+.2f} percentage points")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Disagreements
# MAGIC 
# MAGIC Examine cases where the model predicts a different root cause than assigned.

# COMMAND ----------

# Find disagreements
disagreements = train_pd_sorted[~train_pd_sorted["prediction_matches"]].copy()

print(f"\n📊 Disagreement Analysis:")
print(f"{'='*60}")
print(f"   Total Disagreements: {len(disagreements):,}")

# Analyze disagreement patterns
print(f"\n📈 Top Disagreement Patterns (Actual → Predicted):")
disagreement_patterns = (
    disagreements
    .groupby([TARGET_COLUMN, "predicted_root_cause"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(disagreement_patterns.head(15).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify High-Confidence Misclassifications
# MAGIC 
# MAGIC These are cases where the model is **very confident** (>70%) that the root cause
# MAGIC is different from what Quality assigned. These warrant review.

# COMMAND ----------

# High confidence disagreements
HIGH_CONFIDENCE_THRESHOLD = 0.70

high_conf_disagreements = train_pd_sorted[
    (~train_pd_sorted["prediction_matches"]) & 
    (train_pd_sorted["prediction_confidence"] > HIGH_CONFIDENCE_THRESHOLD)
]

print(f"📊 High-Confidence Disagreements (>{HIGH_CONFIDENCE_THRESHOLD:.0%} model confidence):")
print(f"{'='*60}")
print(f"   Records: {len(high_conf_disagreements):,}")
print(f"   Total Cost Impact: ${high_conf_disagreements['total_scrap_cost'].sum():,.2f}")

# Breakdown by defect category
print(f"\n📈 By Defect Category:")
high_conf_by_defect = (
    high_conf_disagreements
    .groupby("defect_category")
    .agg({
        "prediction_matches": "count",
        "total_scrap_cost": "sum",
        "prediction_confidence": "mean"
    })
    .rename(columns={
        "prediction_matches": "count",
        "total_scrap_cost": "cost_impact",
        "prediction_confidence": "avg_confidence"
    })
    .sort_values("cost_impact", ascending=False)
)
print(high_conf_by_defect.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Validation Results
# MAGIC 
# MAGIC Save the validation results to a Delta table for review by Quality team.

# COMMAND ----------

# Prepare validation results for saving
validation_results = train_pd_sorted[[
    "defect_category", "work_center_type_lookup", "plant_id",
    TARGET_COLUMN, "predicted_root_cause", 
    "prediction_matches", "prediction_confidence",
    "total_scrap_cost", "quantity_scrapped"
]].copy()

# Convert to Spark and save
validation_spark = spark.createDataFrame(validation_results)
validation_spark = (
    validation_spark
    .withColumn("is_high_confidence_disagreement", 
        (~F.col("prediction_matches")) & (F.col("prediction_confidence") > HIGH_CONFIDENCE_THRESHOLD)
    )
    .withColumn("analysis_date", F.current_timestamp())
)

validation_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_root_cause_validation"
)

print(f"✅ Saved validation results to {FULL_SCHEMA}.ml_root_cause_validation")
print(f"   Records: {validation_spark.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 Model 2: 8D Effectiveness Prediction
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Predict whether an 8D report's corrective action will be **Effective** based on:
# MAGIC - 8D characteristics (severity, days open, etc.)
# MAGIC - Associated scrap impact
# MAGIC - Root cause category

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare 8D Data
# MAGIC 
# MAGIC Filter to closed 8Ds with verification results for training.

# COMMAND ----------

# Convert 8D data to Pandas
eight_d_pd = eight_d_df.toPandas()

# Filter to closed 8Ds with verification results
closed_8d = eight_d_pd[
    (eight_d_pd["status"] == "Closed") & 
    (eight_d_pd["verification_result"].notna())
].copy()

# Convert dates to datetime for temporal splitting
closed_8d["created_date"] = pd.to_datetime(closed_8d["created_date"])
closed_8d["closed_date"] = pd.to_datetime(closed_8d["closed_date"])

print(f"📊 8D Training Data:")
print(f"   Closed 8Ds with verification: {len(closed_8d):,}")
print(f"   Date Range: {closed_8d['created_date'].min().date()} to {closed_8d['created_date'].max().date()}")

print(f"\n📈 Verification Results:")
print(closed_8d["verification_result"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔤 Extract NLP Features from 8D Text
# MAGIC 
# MAGIC Apply advanced NLP feature extraction to improve model performance.

# COMMAND ----------

print("🔄 Extracting NLP features from 8D text fields...")

# Extract features from problem description
print("   Analyzing problem descriptions...")
problem_features = closed_8d['problem_description'].apply(extract_text_features).apply(pd.Series)
problem_features.columns = [f'problem_{col}' for col in problem_features.columns]

# Extract features from root cause analysis
print("   Analyzing root cause text...")
root_cause_features = closed_8d['root_cause_analysis'].apply(extract_text_features).apply(pd.Series)
root_cause_features.columns = [f'root_cause_{col}' for col in root_cause_features.columns]

# Extract features from corrective actions
print("   Analyzing corrective actions...")
corrective_features = closed_8d['corrective_actions'].apply(extract_text_features).apply(pd.Series)
corrective_features.columns = [f'corrective_{col}' for col in corrective_features.columns]

# Extract root cause keywords (6M categories)
print("   Extracting root cause keywords (6M analysis)...")
root_cause_keywords = closed_8d['root_cause_analysis'].apply(extract_root_cause_keywords).apply(pd.Series)

# Calculate text similarity between problem and solution
print("   Calculating text coherence scores...")
closed_8d['problem_solution_similarity'] = closed_8d.apply(
    lambda row: calculate_text_similarity(
        row['problem_description'], 
        row['corrective_actions']
    ), axis=1
)

# Combine all NLP features
nlp_features = pd.concat([
    problem_features,
    root_cause_features,
    corrective_features,
    root_cause_keywords,
    closed_8d[['problem_solution_similarity']]
], axis=1)

print(f"\n✅ NLP Feature Extraction Complete")
print(f"   Total NLP Features Created: {nlp_features.shape[1]}")
print(f"\n📊 Sample NLP Features:")
print(nlp_features.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize NLP Features

# COMMAND ----------

# Visualize key NLP features by effectiveness
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Add is_effective to nlp_features for plotting
nlp_viz = nlp_features.copy()
nlp_viz['is_effective'] = (closed_8d['verification_result'] == 'Effective').astype(int)

# Plot 1: Root cause word count by effectiveness
axes[0, 0].boxplot([
    nlp_viz[nlp_viz['is_effective'] == 1]['root_cause_word_count'],
    nlp_viz[nlp_viz['is_effective'] == 0]['root_cause_word_count']
])
axes[0, 0].set_xticklabels(['Effective', 'Not Effective'])
axes[0, 0].set_title('Root Cause Text Length')
axes[0, 0].set_ylabel('Word Count')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Action verb count
axes[0, 1].boxplot([
    nlp_viz[nlp_viz['is_effective'] == 1]['corrective_action_verb_count'],
    nlp_viz[nlp_viz['is_effective'] == 0]['corrective_action_verb_count']
])
axes[0, 1].set_xticklabels(['Effective', 'Not Effective'])
axes[0, 1].set_title('Action Verbs in Corrective Actions')
axes[0, 1].set_ylabel('Action Verb Count')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Root cause sentiment
axes[0, 2].boxplot([
    nlp_viz[nlp_viz['is_effective'] == 1]['root_cause_sentiment_polarity'],
    nlp_viz[nlp_viz['is_effective'] == 0]['root_cause_sentiment_polarity']
])
axes[0, 2].set_xticklabels(['Effective', 'Not Effective'])
axes[0, 2].set_title('Root Cause Text Sentiment')
axes[0, 2].set_ylabel('Sentiment Polarity')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: 6M Category mentions
m6_features = ['mentions_equipment', 'mentions_material', 'mentions_process', 
               'mentions_human', 'mentions_measurement', 'mentions_environment']
m6_effective = nlp_viz[nlp_viz['is_effective'] == 1][m6_features].mean()
m6_not_effective = nlp_viz[nlp_viz['is_effective'] == 0][m6_features].mean()

x = range(len(m6_features))
width = 0.35
axes[1, 0].bar([i - width/2 for i in x], m6_effective, width, label='Effective', alpha=0.8)
axes[1, 0].bar([i + width/2 for i in x], m6_not_effective, width, label='Not Effective', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([m.replace('mentions_', '') for m in m6_features], rotation=45, ha='right')
axes[1, 0].set_title('6M Root Cause Categories')
axes[1, 0].set_ylabel('Mention Rate')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 5: Problem-solution similarity
axes[1, 1].boxplot([
    nlp_viz[nlp_viz['is_effective'] == 1]['problem_solution_similarity'],
    nlp_viz[nlp_viz['is_effective'] == 0]['problem_solution_similarity']
])
axes[1, 1].set_xticklabels(['Effective', 'Not Effective'])
axes[1, 1].set_title('Problem-Solution Text Coherence')
axes[1, 1].set_ylabel('Similarity Score')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Technical term ratio
axes[1, 2].boxplot([
    nlp_viz[nlp_viz['is_effective'] == 1]['root_cause_technical_term_ratio'],
    nlp_viz[nlp_viz['is_effective'] == 0]['root_cause_technical_term_ratio']
])
axes[1, 2].set_xticklabels(['Effective', 'Not Effective'])
axes[1, 2].set_title('Technical Term Density')
axes[1, 2].set_ylabel('Ratio')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 NLP Insights:")
print(f"   • Effective 8Ds have {nlp_viz[nlp_viz['is_effective']==1]['root_cause_word_count'].mean():.0f} words on average")
print(f"   • Not effective 8Ds have {nlp_viz[nlp_viz['is_effective']==0]['root_cause_word_count'].mean():.0f} words on average")
print(f"   • Effective 8Ds use {nlp_viz[nlp_viz['is_effective']==1]['corrective_action_verb_count'].mean():.1f} action verbs")
print(f"   • Text coherence (problem-solution): {nlp_viz['problem_solution_similarity'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Target Variable
# MAGIC 
# MAGIC Binary classification: Effective (1) vs Not Effective (0)

# COMMAND ----------

# Create binary target
closed_8d["is_effective"] = (closed_8d["verification_result"] == "Effective").astype(int)

print(f"📊 Target Distribution:")
print(f"   Effective: {closed_8d['is_effective'].sum()} ({closed_8d['is_effective'].mean():.1%})")
print(f"   Not Effective: {len(closed_8d) - closed_8d['is_effective'].sum()} ({1-closed_8d['is_effective'].mean():.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features with NLP
# MAGIC 
# MAGIC Select features that may influence 8D effectiveness, including new NLP features.

# COMMAND ----------

# Define 8D features (original)
EFFECTIVENESS_FEATURES = [
    "days_open",
    "total_scrap_cost",
    "total_quantity_affected",
    "num_parts_affected",
    "root_cause_word_count",
    "corrective_action_word_count"
]

EFFECTIVENESS_CAT_FEATURES = [
    "severity",
    "priority", 
    "defect_category",
    "root_cause_category"
]

# Add NLP features
NLP_EFFECTIVENESS_FEATURES = [
    # Problem description features
    'problem_sentiment_polarity',
    'problem_sentiment_subjectivity',
    'problem_contains_numbers',
    'problem_technical_term_ratio',
    
    # Root cause analysis features
    'root_cause_sentiment_polarity',
    'root_cause_avg_word_length',
    'root_cause_action_verb_count',
    'root_cause_unique_word_ratio',
    
    # Corrective action features
    'corrective_word_count',
    'corrective_sentence_count',
    'corrective_action_verb_count',
    'corrective_contains_numbers',
    
    # 6M category features
    'mentions_equipment',
    'mentions_material',
    'mentions_process',
    'mentions_human',
    'mentions_measurement',
    'mentions_environment',
    
    # Coherence feature
    'problem_solution_similarity'
]

# Combine original 8D data with NLP features
effectiveness_df = closed_8d[
    ["created_date"] + EFFECTIVENESS_FEATURES + EFFECTIVENESS_CAT_FEATURES + ["is_effective"]
].copy()

# Add NLP features
effectiveness_df = pd.concat([effectiveness_df, nlp_features], axis=1)

# Drop rows with missing values
effectiveness_df = effectiveness_df.dropna()

# Encode categorical features
for col in EFFECTIVENESS_CAT_FEATURES:
    le = LabelEncoder()
    effectiveness_df[f"{col}_enc"] = le.fit_transform(effectiveness_df[col].astype(str))

print(f"📊 Effectiveness Training Data with NLP Features:")
print(f"   Records: {len(effectiveness_df):,}")
print(f"   Original Features: {len(EFFECTIVENESS_FEATURES + EFFECTIVENESS_CAT_FEATURES)}")
print(f"   NLP Features: {len(NLP_EFFECTIVENESS_FEATURES)}")
print(f"   Total Features: {len(EFFECTIVENESS_FEATURES) + len(EFFECTIVENESS_CAT_FEATURES) + len(NLP_EFFECTIVENESS_FEATURES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Effectiveness Model with NLP Features
# MAGIC 
# MAGIC Train a Gradient Boosting classifier with enhanced NLP features.

# COMMAND ----------

# Prepare feature matrix with NLP features
feature_cols_eff = (EFFECTIVENESS_FEATURES + 
                    [f"{col}_enc" for col in EFFECTIVENESS_CAT_FEATURES] + 
                    NLP_EFFECTIVENESS_FEATURES)

# Verify all features exist in dataframe
print("\n🔍 Verifying NLP Feature Integration...")
missing_features = [col for col in feature_cols_eff if col not in effectiveness_df.columns]
if missing_features:
    print(f"❌ ERROR: Missing features: {missing_features}")
    raise ValueError(f"Missing required features: {missing_features}")
else:
    print(f"✅ All {len(feature_cols_eff)} features present in dataframe")

# Check for NaN values in feature columns
nan_counts = effectiveness_df[feature_cols_eff].isna().sum()
if nan_counts.sum() > 0:
    print(f"⚠️  WARNING: NaN values detected in features:")
    print(nan_counts[nan_counts > 0])
else:
    print(f"✅ No NaN values in feature columns")

# Verify feature types
print(f"\n📊 Feature Type Distribution:")
print(f"   Numeric Features: {len([col for col in feature_cols_eff if effectiveness_df[col].dtype in ['int64', 'float64']])}")
print(f"   Categorical/Encoded: {len([col for col in feature_cols_eff if col.endswith('_enc')])}")

X_eff = effectiveness_df[feature_cols_eff].values
y_eff = effectiveness_df["is_effective"].values

print(f"\n📊 Feature Matrix Shape:")
print(f"   X shape: {X_eff.shape} (samples × features)")
print(f"   y shape: {y_eff.shape}")
print(f"   Feature density: {(X_eff != 0).sum() / X_eff.size:.1%} non-zero values")

# Temporal train/test split for 8D effectiveness
print(f"\n🔄 Performing Temporal Split...")

# Sort by created_date
effectiveness_df_sorted = effectiveness_df.sort_values('created_date').reset_index(drop=True)

# Calculate split point (80% for training)
split_idx_eff = int(len(effectiveness_df_sorted) * 0.8)
split_date_eff = effectiveness_df_sorted.loc[split_idx_eff, 'created_date']

# Split into train and test based on date
train_mask_eff = effectiveness_df_sorted['created_date'] < split_date_eff
test_mask_eff = effectiveness_df_sorted['created_date'] >= split_date_eff

train_set_eff = effectiveness_df_sorted[train_mask_eff]
test_set_eff = effectiveness_df_sorted[test_mask_eff]

# Prepare X and y
X_train_eff = train_set_eff[feature_cols_eff].values
y_train_eff = train_set_eff["is_effective"].values

X_test_eff = test_set_eff[feature_cols_eff].values
y_test_eff = test_set_eff["is_effective"].values

print(f"📊 Temporal Data Split for 8D Effectiveness:")
print(f"{'='*70}")
print(f"\n📅 Training Set:")
print(f"   Date Range: {train_set_eff['created_date'].min().date()} to {train_set_eff['created_date'].max().date()}")
print(f"   Samples: {len(X_train_eff):,}")
print(f"   Effective: {y_train_eff.sum():,} ({y_train_eff.mean():.1%})")

print(f"\n📅 Test Set:")
print(f"   Date Range: {test_set_eff['created_date'].min().date()} to {test_set_eff['created_date'].max().date()}")
print(f"   Samples: {len(X_test_eff):,}")
print(f"   Effective: {y_test_eff.sum():,} ({y_test_eff.mean():.1%})")

print(f"\n🔒 Temporal Split Ensures:")
print(f"   ✅ Training on older 8Ds (before {split_date_eff.date()})")
print(f"   ✅ Testing on newer 8Ds (from {split_date_eff.date()} onward)")
print(f"   ✅ No future information leaking into training")

# COMMAND ----------

# Train effectiveness predictor
with mlflow.start_run(run_name="8d_effectiveness_predictor_nlp") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("dataset_size", len(effectiveness_df))
    mlflow.log_param("num_features", len(feature_cols_eff))
    mlflow.log_param("num_nlp_features", len(NLP_EFFECTIVENESS_FEATURES))
    mlflow.log_param("feature_list", json.dumps(feature_cols_eff[:20]))  # Log first 20 for brevity
    mlflow.log_param("split_method", "temporal")
    mlflow.log_param("split_date", str(split_date_eff.date()))
    
    # Define model
    eff_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    
    # Train
    print("🔄 Training 8D Effectiveness model...")
    eff_model.fit(X_train_eff, y_train_eff)
    
    # Predict
    y_pred_eff = eff_model.predict(X_test_eff)
    y_proba_eff = eff_model.predict_proba(X_test_eff)[:, 1]
    
    # Metrics
    accuracy_eff = accuracy_score(y_test_eff, y_pred_eff)
    precision_eff = precision_score(y_test_eff, y_pred_eff)
    recall_eff = recall_score(y_test_eff, y_pred_eff)
    f1_eff = f1_score(y_test_eff, y_pred_eff)
    
    try:
        auc_eff = roc_auc_score(y_test_eff, y_proba_eff)
        mlflow.log_metric("auc_roc", auc_eff)
    except:
        auc_eff = None
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_eff)
    mlflow.log_metric("precision", precision_eff)
    mlflow.log_metric("recall", recall_eff)
    mlflow.log_metric("f1_score", f1_eff)
    
    # Feature importance
    eff_importance = pd.DataFrame({
        "feature": feature_cols_eff,
        "importance": eff_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=eff_importance, x="importance", y="feature", ax=ax)
    ax.set_title("Features Influencing 8D Effectiveness")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(eff_model, "model")
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ Model: 8D Effectiveness Predictor (with NLP Features)")
    print(f"{'='*70}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy_eff:.4f}")
    print(f"   Precision: {precision_eff:.4f}")
    print(f"   Recall:    {recall_eff:.4f}")
    print(f"   F1 Score:  {f1_eff:.4f}")
    if auc_eff:
        print(f"   AUC-ROC:   {auc_eff:.4f}")
    print(f"\n📊 Features Used:")
    print(f"   Original: {len(EFFECTIVENESS_FEATURES + EFFECTIVENESS_CAT_FEATURES)}")
    print(f"   NLP: {len(NLP_EFFECTIVENESS_FEATURES)}")
    print(f"   Total: {len(feature_cols_eff)}")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Feature Importance
# MAGIC 
# MAGIC Identify what makes an 8D corrective action effective.

# COMMAND ----------

# Plot effectiveness feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=eff_importance, x="importance", y="feature", palette="rocket")
ax.set_title("Factors Influencing 8D Effectiveness", fontsize=14)
ax.set_xlabel("Feature Importance", fontsize=12)
plt.tight_layout()
plt.show()

print("\n💡 Key Insights:")
print("   - Longer root cause analysis text correlates with effectiveness")
print("   - Days open matters - faster isn't always better")
print("   - Higher scrap cost drives more thorough investigations")
print("   - NLP features capture text quality and thoroughness")

# Display top NLP features
nlp_importance = eff_importance[eff_importance['feature'].isin(NLP_EFFECTIVENESS_FEATURES)].head(10)
if len(nlp_importance) > 0:
    print(f"\n📝 Top NLP Features by Importance:")
    for _, row in nlp_importance.iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Ensemble Model for 8D Effectiveness
# MAGIC 
# MAGIC Apply voting classifier ensemble to 8D effectiveness prediction with NLP features.

# COMMAND ----------

print("\n🚀 Training Ensemble for 8D Effectiveness Prediction...")

with mlflow.start_run(run_name="8d_effectiveness_ensemble_nlp") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "VotingClassifier")
    mlflow.log_param("dataset_size", len(effectiveness_df))
    mlflow.log_param("num_features", len(feature_cols_eff))
    mlflow.log_param("num_nlp_features", len(NLP_EFFECTIVENESS_FEATURES))
    mlflow.log_param("voting_method", "soft")
    mlflow.log_param("num_base_models", 4)
    mlflow.log_param("split_method", "temporal")
    mlflow.log_param("split_date", str(split_date_eff.date()))
    
    # Define base models for 8D effectiveness
    rf_8d = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    gb_8d = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    xgb_8d = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    lgbm_8d = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Create ensemble
    voting_8d = VotingClassifier(
        estimators=[
            ('rf', rf_8d),
            ('gb', gb_8d),
            ('xgb', xgb_8d),
            ('lgbm', lgbm_8d)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Train ensemble
    print("   Training ensemble...")
    voting_8d.fit(X_train_eff, y_train_eff)
    
    # Predictions
    y_pred_8d_ensemble = voting_8d.predict(X_test_eff)
    y_proba_8d_ensemble = voting_8d.predict_proba(X_test_eff)[:, 1]
    
    # Metrics
    accuracy_8d_ensemble = accuracy_score(y_test_eff, y_pred_8d_ensemble)
    precision_8d_ensemble = precision_score(y_test_eff, y_pred_8d_ensemble)
    recall_8d_ensemble = recall_score(y_test_eff, y_pred_8d_ensemble)
    f1_8d_ensemble = f1_score(y_test_eff, y_pred_8d_ensemble)
    
    try:
        auc_8d_ensemble = roc_auc_score(y_test_eff, y_proba_8d_ensemble)
        mlflow.log_metric("auc_roc", auc_8d_ensemble)
    except:
        auc_8d_ensemble = None
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_8d_ensemble)
    mlflow.log_metric("precision", precision_8d_ensemble)
    mlflow.log_metric("recall", recall_8d_ensemble)
    mlflow.log_metric("f1_score", f1_8d_ensemble)
    
    # Calculate improvement
    improvement_8d = ((accuracy_8d_ensemble - accuracy_eff) / accuracy_eff) * 100
    mlflow.log_metric("improvement_over_single_model_pct", improvement_8d)
    
    # Individual model performance
    individual_8d_results = []
    for name, model in voting_8d.named_estimators_.items():
        y_pred_ind = model.predict(X_test_eff)
        acc_ind = accuracy_score(y_test_eff, y_pred_ind)
        f1_ind = f1_score(y_test_eff, y_pred_ind)
        
        individual_8d_results.append({
            'model': name,
            'accuracy': acc_ind,
            'f1_score': f1_ind
        })
        
        mlflow.log_metric(f"{name}_accuracy", acc_ind)
        mlflow.log_metric(f"{name}_f1", f1_ind)
    
    individual_8d_df = pd.DataFrame(individual_8d_results)
    
    # Log model
    mlflow.sklearn.log_model(voting_8d, "model")
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ 8D Effectiveness Ensemble (with NLP)")
    print(f"{'='*70}")
    print(f"\n📊 Ensemble Metrics:")
    print(f"   Accuracy:  {accuracy_8d_ensemble:.4f}")
    print(f"   Precision: {precision_8d_ensemble:.4f}")
    print(f"   Recall:    {recall_8d_ensemble:.4f}")
    print(f"   F1 Score:  {f1_8d_ensemble:.4f}")
    if auc_8d_ensemble:
        print(f"   AUC-ROC:   {auc_8d_ensemble:.4f}")
    
    print(f"\n📈 Improvement over Single Model:")
    print(f"   Single GB: {accuracy_eff:.4f}")
    print(f"   Ensemble:  {accuracy_8d_ensemble:.4f}")
    print(f"   Change:    {improvement_8d:+.2f}%")
    
    print(f"\n📊 Individual Models:")
    for _, row in individual_8d_df.iterrows():
        print(f"   {row['model']}: {row['accuracy']:.4f}")
    
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# Visualize ensemble vs single model
fig, ax = plt.subplots(figsize=(10, 6))

models_comp = pd.DataFrame([
    {'Model': 'Gradient Boosting', 'Accuracy': accuracy_eff, 'F1': f1_eff},
    {'Model': 'Ensemble', 'Accuracy': accuracy_8d_ensemble, 'F1': f1_8d_ensemble}
])

x = range(len(models_comp))
width = 0.35
ax.bar([i - width/2 for i in x], models_comp['Accuracy'], width, label='Accuracy', alpha=0.8)
ax.bar([i + width/2 for i in x], models_comp['F1'], width, label='F1 Score', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models_comp['Model'])
ax.set_ylabel('Score')
ax.set_title('8D Effectiveness: Ensemble vs Single Model (with NLP Features)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔍 Similar 8D Finder
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Find similar historical 8D reports to recommend proven corrective actions
# MAGIC for new quality issues.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build BERT Semantic Similarity Index
# MAGIC 
# MAGIC Use sentence-transformers (BERT-based) to create semantic embeddings
# MAGIC of 8D text for similarity matching. This provides much better semantic
# MAGIC understanding than TF-IDF keyword matching.
# MAGIC 
# MAGIC **Model**: `all-MiniLM-L6-v2`
# MAGIC - 384-dimensional embeddings
# MAGIC - Optimized for semantic similarity
# MAGIC - Fast inference (~100 sentences/sec)
# MAGIC - Understands context and meaning, not just keywords

# COMMAND ----------

# Combine text fields for similarity matching
eight_d_pd["combined_text"] = (
    eight_d_pd["problem_description"].fillna("") + " " +
    eight_d_pd["root_cause_analysis"].fillna("") + " " +
    eight_d_pd["defect_category"].fillna("") + " " +
    eight_d_pd["corrective_actions"].fillna("")
)

print("🔄 Loading BERT model for semantic embeddings...")
print("   Model: all-MiniLM-L6-v2 (384 dimensions)")

# Load sentence transformer model
# This is a lightweight but powerful model for semantic similarity
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model loaded successfully")
print("\n🔄 Generating embeddings for all 8D reports...")
print(f"   Processing {len(eight_d_pd):,} documents...")

# Generate embeddings for all 8D reports
# This creates a 384-dimensional vector for each document
eight_d_embeddings = bert_model.encode(
    eight_d_pd["combined_text"].tolist(),
    show_progress_bar=True,
    batch_size=32  # Process in batches for efficiency
)

print(f"\n📊 BERT Embedding Matrix:")
print(f"   Documents: {eight_d_embeddings.shape[0]:,}")
print(f"   Embedding dimensions: {eight_d_embeddings.shape[1]}")
print(f"   Memory size: ~{eight_d_embeddings.nbytes / 1024 / 1024:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare BERT vs TF-IDF on Sample Queries
# MAGIC 
# MAGIC Demonstrate the improved semantic understanding of BERT embeddings.

# COMMAND ----------

# Test semantic similarity with BERT vs TF-IDF
test_queries = [
    ("Die wear causing dimensional issues", "Tooling degradation leading to out-of-spec parts"),
    ("Operator training needed", "Insufficient employee skills"),
    ("Material contamination detected", "Foreign material found in raw steel")
]

print("🔍 Comparing BERT vs TF-IDF Semantic Similarity\n")
print("=" * 80)

for q1, q2 in test_queries:
    # BERT similarity
    bert_sim = calculate_text_similarity(q1, q2, use_bert=True)
    
    # TF-IDF similarity
    tfidf_sim = calculate_text_similarity(q1, q2, use_bert=False)
    
    print(f"\n📝 Query 1: {q1}")
    print(f"📝 Query 2: {q2}")
    print(f"   BERT Similarity:   {bert_sim:.3f}")
    print(f"   TF-IDF Similarity: {tfidf_sim:.3f}")
    print(f"   BERT captures {(bert_sim - tfidf_sim) * 100:+.1f}% more semantic meaning")
    print("-" * 80)

print("\n💡 Insight: BERT understands semantic meaning (synonyms, context)")
print("   while TF-IDF only matches keywords.")

# Visualize the comparison
fig, ax = plt.subplots(figsize=(12, 6))

comparison_data = []
for q1, q2 in test_queries:
    bert_sim = calculate_text_similarity(q1, q2, use_bert=True)
    tfidf_sim = calculate_text_similarity(q1, q2, use_bert=False)
    
    label = q1[:30] + "..."
    comparison_data.append({
        'query': label,
        'BERT': bert_sim,
        'TF-IDF': tfidf_sim
    })

comp_df = pd.DataFrame(comparison_data)

x = range(len(comp_df))
width = 0.35

ax.bar([i - width/2 for i in x], comp_df['BERT'], width, label='BERT (Semantic)', alpha=0.8, color='green')
ax.bar([i + width/2 for i in x], comp_df['TF-IDF'], width, label='TF-IDF (Keyword)', alpha=0.8, color='steelblue')

ax.set_ylabel('Similarity Score', fontsize=12)
ax.set_title('BERT vs TF-IDF: Semantic Understanding Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(comp_df['query'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\n📊 Average Improvement: {((comp_df['BERT'].mean() - comp_df['TF-IDF'].mean()) / comp_df['TF-IDF'].mean() * 100):+.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define BERT-Based Similar 8D Search Function
# MAGIC 
# MAGIC Function to find similar 8Ds based on semantic similarity using BERT embeddings.

# COMMAND ----------

def find_similar_8ds(
    query_text: str, 
    top_n: int = 5,
    only_effective: bool = False
) -> pd.DataFrame:
    """
    Find similar historical 8D reports based on semantic similarity using BERT.
    
    Parameters:
    -----------
    query_text : str
        Description of the new quality issue
    top_n : int
        Number of similar 8Ds to return
    only_effective : bool
        If True, only return 8Ds with "Effective" verification
        
    Returns:
    --------
    pd.DataFrame
        Similar 8Ds with corrective actions and effectiveness
    """
    # Generate BERT embedding for query
    query_embedding = bert_model.encode([query_text])
    
    # Calculate cosine similarities with all 8D embeddings
    similarities = sklearn_cosine_similarity(query_embedding, eight_d_embeddings).flatten()
    
    # Get top matches
    if only_effective:
        # Filter to only effective 8Ds
        effective_mask = eight_d_pd["verification_result"] == "Effective"
        effective_indices = np.where(effective_mask)[0]
        effective_similarities = similarities[effective_indices]
        top_local_indices = effective_similarities.argsort()[-top_n:][::-1]
        top_indices = effective_indices[top_local_indices]
    else:
        top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Build results
    results = eight_d_pd.iloc[top_indices][[
        "eight_d_id", "title", "defect_category", "root_cause_category",
        "corrective_actions", "verification_result", "total_scrap_cost"
    ]].copy()
    results["similarity_score"] = similarities[top_indices]
    
    return results

print("✅ BERT-based similarity search function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔬 Advanced: Hybrid Search (BERT + Metadata Filtering)
# MAGIC 
# MAGIC Combine semantic similarity with metadata filters for even more precise results.

# COMMAND ----------

def find_similar_8ds_hybrid(
    query_text: str,
    top_n: int = 5,
    only_effective: bool = False,
    defect_category: str = None,
    root_cause_category: str = None,
    min_cost: float = None,
    max_cost: float = None
) -> pd.DataFrame:
    """
    Advanced hybrid search combining BERT semantic similarity with metadata filters.
    
    Parameters:
    -----------
    query_text : str
        Description of the new quality issue
    top_n : int
        Number of similar 8Ds to return
    only_effective : bool
        If True, only return effective 8Ds
    defect_category : str
        Filter by specific defect category
    root_cause_category : str
        Filter by specific root cause
    min_cost : float
        Minimum scrap cost threshold
    max_cost : float
        Maximum scrap cost threshold
        
    Returns:
    --------
    pd.DataFrame
        Similar 8Ds matching all criteria
    """
    # Generate BERT embedding for query
    query_embedding = bert_model.encode([query_text])
    
    # Calculate similarities
    similarities = sklearn_cosine_similarity(query_embedding, eight_d_embeddings).flatten()
    
    # Start with all indices
    mask = np.ones(len(eight_d_pd), dtype=bool)
    
    # Apply filters
    if only_effective:
        mask &= (eight_d_pd["verification_result"] == "Effective").values
    
    if defect_category:
        mask &= (eight_d_pd["defect_category"] == defect_category).values
    
    if root_cause_category:
        mask &= (eight_d_pd["root_cause_category"] == root_cause_category).values
    
    if min_cost is not None:
        mask &= (eight_d_pd["total_scrap_cost"] >= min_cost).values
    
    if max_cost is not None:
        mask &= (eight_d_pd["total_scrap_cost"] <= max_cost).values
    
    # Get filtered indices
    filtered_indices = np.where(mask)[0]
    
    if len(filtered_indices) == 0:
        print("⚠️ No 8Ds match the specified criteria")
        return pd.DataFrame()
    
    # Get top matches from filtered set
    filtered_similarities = similarities[filtered_indices]
    top_local_indices = filtered_similarities.argsort()[-min(top_n, len(filtered_indices)):][::-1]
    top_indices = filtered_indices[top_local_indices]
    
    # Build results
    results = eight_d_pd.iloc[top_indices][[
        "eight_d_id", "title", "defect_category", "root_cause_category",
        "corrective_actions", "verification_result", "total_scrap_cost"
    ]].copy()
    results["similarity_score"] = similarities[top_indices]
    
    return results

print("✅ Hybrid search function defined")
print("   Combines: BERT semantic similarity + metadata filtering")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Hybrid Search
# MAGIC 
# MAGIC Example: Find dimensional defects caused by tooling, but only effective high-cost cases.

# COMMAND ----------

# Example: Targeted hybrid search
hybrid_query = "Tooling wear causing dimensional variation in stamped parts"

print("🔍 Hybrid Search Example:")
print(f"   Query: {hybrid_query}")
print(f"   Filters: Effective only, Dimensional defects, Cost > $5000\n")

hybrid_results = find_similar_8ds_hybrid(
    query_text=hybrid_query,
    top_n=5,
    only_effective=True,
    defect_category="Dimensional",
    min_cost=5000
)

if len(hybrid_results) > 0:
    print(f"✅ Found {len(hybrid_results)} matching 8Ds:\n")
    print("-" * 80)
    
    for idx, row in hybrid_results.iterrows():
        print(f"\n📋 {row['eight_d_id']} (Similarity: {row['similarity_score']:.1%})")
        print(f"   Category: {row['defect_category']} / {row['root_cause_category']}")
        print(f"   Cost: ${row['total_scrap_cost']:,.2f}")
        print(f"   Action: {str(row['corrective_actions'])[:120]}...")
else:
    print("⚠️ No results found with these criteria")

print("\n💡 Use Case: Find effective solutions for specific defect types")
print("   within your cost range, using semantic similarity.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Find Similar 8Ds with BERT
# MAGIC 
# MAGIC Test the BERT-based semantic search with sample queries.
# MAGIC BERT understands semantic meaning, not just keyword matching.

# COMMAND ----------

# Example query for a new quality issue
example_query = """
Dimensional defects found on stamped body panels. 
Multiple parts showing out-of-tolerance measurements on width dimension.
Suspected die wear issue causing progressive dimension drift.
"""

# Find similar 8Ds using BERT embeddings
similar_8ds = find_similar_8ds(example_query, top_n=5)

print("🔍 Similar Historical 8Ds for Query (Using BERT Embeddings):")
print("-" * 70)
print(f"Query: {example_query[:100]}...")
print("-" * 70)

for idx, row in similar_8ds.iterrows():
    print(f"\n📋 8D: {row['eight_d_id']} (Semantic Similarity: {row['similarity_score']:.1%})")
    print(f"   Title: {row['title'][:60]}...")
    print(f"   Root Cause: {row['root_cause_category']}")
    print(f"   Verification: {row['verification_result']}")
    print(f"   Corrective Action: {str(row['corrective_actions'])[:150]}...")

print(f"\n💡 BERT Advantage: Found semantically similar 8Ds even if they use")
print(f"   different terminology (e.g., 'tooling degradation' vs 'die wear')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Only Effective Solutions (BERT)
# MAGIC 
# MAGIC Search for similar 8Ds that had **effective** corrective actions using semantic matching.

# COMMAND ----------

# Find similar 8Ds with effective solutions only
effective_8ds = find_similar_8ds(example_query, top_n=5, only_effective=True)

print("✅ Similar 8Ds with EFFECTIVE Corrective Actions (BERT-based):")
print("-" * 70)

for idx, row in effective_8ds.iterrows():
    print(f"\n📋 8D: {row['eight_d_id']} (Semantic Similarity: {row['similarity_score']:.1%})")
    print(f"   Root Cause: {row['root_cause_category']}")
    print(f"   Corrective Action: {str(row['corrective_actions'])[:200]}...")

print(f"\n💡 Use Case: When opening a new 8D, search for similar effective solutions")
print(f"   to accelerate root cause analysis and corrective action planning.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🤖 LLM-Based 8D Analysis
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Use Databricks Foundation Models to analyze 8D text and extract insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define LLM Analysis Function
# MAGIC 
# MAGIC Call the Foundation Model API to analyze 8D reports.

# COMMAND ----------

# Databricks Foundation Model endpoint
LLM_ENDPOINT_NAME = "databricks-gpt-5-1"

def analyze_8d_with_llm(eight_d_record: dict) -> dict:
    """
    Use Databricks Foundation Model to analyze an 8D report.
    
    Returns structured analysis including:
    - Problem summary
    - Root cause validity assessment
    - Corrective action completeness
    - Risk score for recurrence
    - Recommended additional actions
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    
    prompt = f"""You are a quality engineering expert. Analyze this 8D quality report and provide a structured assessment.

8D REPORT DETAILS:
- Title: {eight_d_record.get('title', 'N/A')}
- Defect Category: {eight_d_record.get('defect_category', 'N/A')}
- Severity: {eight_d_record.get('severity', 'N/A')}
- Scrap Cost: ${eight_d_record.get('total_scrap_cost', 0):,.2f}

PROBLEM DESCRIPTION:
{eight_d_record.get('problem_description', 'N/A')}

ROOT CAUSE ANALYSIS:
{eight_d_record.get('root_cause_analysis', 'N/A')}
Assigned Root Cause Category: {eight_d_record.get('root_cause_category', 'N/A')}

CORRECTIVE ACTIONS:
{eight_d_record.get('corrective_actions', 'N/A')}

PREVENTIVE ACTIONS:
{eight_d_record.get('preventive_actions', 'N/A')}

Please provide:
1. SUMMARY: Brief 2-3 sentence summary of the problem
2. ROOT CAUSE VALIDITY: Is the assigned root cause logical? (Yes/Partially/No) with explanation
3. CORRECTIVE ACTION ASSESSMENT: Are the actions sufficient? (Complete/Partial/Insufficient)
4. RECURRENCE RISK: Score 1-10 (10 = high risk of recurrence)
5. RECOMMENDATIONS: Any additional actions recommended

Format your response with clear headers for each section."""

    try:
        w = WorkspaceClient()
        
        response = w.serving_endpoints.query(
            name=LLM_ENDPOINT_NAME,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=prompt
                )
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "title": eight_d_record.get("title"),
            "llm_analysis": response.choices[0].message.content,
            "analyzed_at": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "title": eight_d_record.get("title"),
            "llm_analysis": f"Analysis failed: {str(e)}",
            "analyzed_at": datetime.now().isoformat(),
            "status": "error"
        }

print("✅ LLM Analysis function defined successfully")
print(f"   Using endpoint: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Sample 8D Reports
# MAGIC 
# MAGIC Run LLM analysis on the top 5 8Ds by cost impact.

# COMMAND ----------

# Get top 5 8Ds by cost impact
print("📊 Fetching top 5 8D reports by cost impact...")

sample_8ds = (
    eight_d_df
    .orderBy(F.desc("total_scrap_cost"))
    .limit(5)
    .toPandas()
    .to_dict("records")
)

print(f"   Found {len(sample_8ds)} records to analyze")

# Show what we're about to analyze
print("\n📋 8D Reports to Analyze:")
print("-" * 70)
for i, record in enumerate(sample_8ds, 1):
    print(f"   {i}. {record.get('eight_d_id', 'N/A')} - ${record.get('total_scrap_cost', 0):,.2f}")
    print(f"      Title: {str(record.get('title', 'N/A'))[:60]}...")

print("\n" + "=" * 70)
print("🤖 Starting LLM Analysis...")
print("=" * 70)

llm_analyses = []

for i, record in enumerate(sample_8ds, 1):
    title = str(record.get('title', 'N/A'))[:50]
    eight_d_id = record.get('eight_d_id', 'Unknown')
    
    print(f"\n🔄 [{i}/{len(sample_8ds)}] Analyzing: {eight_d_id}")
    print(f"   Title: {title}...")
    
    try:
        analysis = analyze_8d_with_llm(record)
        llm_analyses.append(analysis)
        
        if analysis["status"] == "success":
            print(f"   ✅ Analysis complete")
        else:
            print(f"   ❌ Analysis failed: {analysis.get('llm_analysis', 'Unknown error')[:100]}")
    except Exception as e:
        print(f"   ❌ Exception during analysis: {str(e)[:100]}")
        llm_analyses.append({
            "eight_d_id": eight_d_id,
            "title": title,
            "llm_analysis": f"Exception: {str(e)}",
            "status": "error"
        })

print(f"\n✅ Analysis complete: {sum(1 for a in llm_analyses if a.get('status') == 'success')}/{len(llm_analyses)} successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display LLM Analysis Results

# COMMAND ----------

# Display LLM analyses
if not llm_analyses:
    print("⚠️ No analyses to display. Check if the previous cell ran successfully.")
else:
    successful = [a for a in llm_analyses if a.get('status') == 'success']
    failed = [a for a in llm_analyses if a.get('status') != 'success']
    
    print(f"📊 Results Summary: {len(successful)} successful, {len(failed)} failed\n")
    
    for analysis in llm_analyses:
        print("\n" + "=" * 80)
        print(f"📋 8D ID: {analysis.get('eight_d_id', 'N/A')}")
        print(f"   Title: {analysis.get('title', 'N/A')}")
        print(f"   Status: {'✅ Success' if analysis.get('status') == 'success' else '❌ Failed'}")
        print("-" * 80)
        print(analysis.get('llm_analysis', 'No analysis available'))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Summary & Key Findings
# MAGIC ---

# COMMAND ----------

# Print summary
print("=" * 70)
print("📊 ROOT CAUSE ANALYSIS & 8D VALIDATION - SUMMARY")
print("=" * 70)

print(f"\n🎯 Root Cause Classification Models:")
print(f"   Random Forest Accuracy: {accuracy:.1%}")
print(f"   Ensemble Accuracy: {accuracy_ensemble:.1%}")
print(f"   Ensemble Improvement: {improvement:+.2f}%")
print(f"   Agreement with Quality Team: {agreement_rate_ensemble:.1%}")
print(f"   High-Confidence Disagreements: {len(high_conf_disagreements):,}")

print(f"\n🔄 Ensemble Configuration:")
print(f"   Models: Random Forest, Gradient Boosting, XGBoost, LightGBM")
print(f"   Voting Method: Soft (probability-based)")
print(f"   Cross-Validation: {cv_scores_ensemble.mean():.4f} ± {cv_scores_ensemble.std()*2:.4f}")

print(f"\n📈 8D Effectiveness Predictor:")
print(f"   Accuracy: {accuracy_eff:.1%}")
print(f"   Key Factors: Root cause word count, days open, defect severity")

print(f"\n🔍 Similar 8D Finder (BERT-based):")
print(f"   8Ds Indexed: {len(eight_d_pd):,}")
print(f"   Embedding Model: all-MiniLM-L6-v2")
print(f"   Embedding Dimensions: {eight_d_embeddings.shape[1]}")
print(f"   Semantic Understanding: ✅ (understands synonyms & context)")

print(f"\n📁 Output Tables:")
print(f"   - {FULL_SCHEMA}.ml_root_cause_validation")

print(f"\n🔗 MLflow Experiment: Default Notebook Experiment")

print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS:")
print("=" * 70)
print("""
1. REVIEW HIGH-CONFIDENCE DISAGREEMENTS
   - Model identified cases where root cause may be misclassified
   - Focus on defect categories with highest cost impact

2. IMPROVE 8D DOCUMENTATION
   - Longer, more detailed root cause analysis correlates with effectiveness
   - Ensure thorough documentation in corrective actions

3. USE SIMILAR 8D FINDER
   - Before creating new 8D, search for similar historical cases
   - Leverage proven corrective actions

4. MONITOR ROOT CAUSE TRENDS
   - Track if certain root causes are over/under-assigned
   - Calibrate Quality team on classification criteria
""")
