!pip install textstat
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sentence_transformers import SentenceTransformer

# Load embeddings model (small and fast)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_extra_features(prompt, response, reference=None):
    """Extracts additional ML-friendly features from prompt/response pairs."""
    features = {}

    # Length features
    features['resp_word_count'] = len(response.split())
    features['resp_sentence_count'] = response.count('.') + response.count('!') + response.count('?')
    features['avg_sentence_length'] = (features['resp_word_count'] / features['resp_sentence_count']) if features['resp_sentence_count'] > 0 else 0

    # Readability features
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(response)
        features['flesch_kincaid'] = flesch_kincaid_grade(response)
    except:
        features['flesch_reading_ease'] = 0
        features['flesch_kincaid'] = 0

    # Structural cues (lists, enumerations)
    features['comma_count'] = response.count(',')
    features['numbered_list'] = len(re.findall(r"\d+\.", response))

    # Token overlap (prompt vs response)
    prompt_tokens = set(prompt.lower().split())
    resp_tokens = set(response.lower().split())
    features['jaccard_overlap'] = len(prompt_tokens & resp_tokens) / len(prompt_tokens | resp_tokens) if (prompt_tokens | resp_tokens) else 0

    # Embedding-based similarity
    try:
        embeddings = embedder.encode([prompt, response])
        features['embedding_similarity'] = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    except:
        features['embedding_similarity'] = 0.0

    # Reference embedding similarity (if available)
    if reference:
        try:
            embeddings = embedder.encode([reference, response])
            features['ref_embedding_similarity'] = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        except:
            features['ref_embedding_similarity'] = 0.0
    else:
        features['ref_embedding_similarity'] = 0.0

    return features

evaluator = AgenticEvaluator(demo, ref_key='reference')
report_df, leaderboard_df = evaluator.evaluate_all()

# Add extra features for each row
extra_features_list = []
for _, row in report_df.iterrows():
    feats = extract_extra_features(row['prompt'], row['response'], row.get('reference'))
    extra_features_list.append(feats)

extra_df = pd.DataFrame(extra_features_list)
full_features = pd.concat([report_df, extra_df], axis=1)

# Feature matrix (X) and labels (y) - using composite score as pseudo-label
X = full_features[['instruction_score', 'coherence_score', 'assumption_score',
                   'hallucination_risk', 'reference_agreement',
                   'resp_word_count', 'resp_sentence_count', 'avg_sentence_length',
                   'flesch_reading_ease', 'flesch_kincaid',
                   'comma_count', 'numbered_list', 'jaccard_overlap',
                   'embedding_similarity', 'ref_embedding_similarity']].fillna(0)

y = full_features['composite_score']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train ML model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Predict scores for all responses
full_features['ml_predicted_score'] = rf.predict(X)

# Compare heuristic vs ML scores
print(full_features[['id', 'composite_score', 'ml_predicted_score']])
