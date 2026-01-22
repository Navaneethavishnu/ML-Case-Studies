import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# Set working directory to script location to find the csv
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
try:
    df = pd.read_csv('youtube_shorts_performance_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: youtube_shorts_performance_dataset.csv not found.")
    exit()

# --- Enhanced Preprocessing & Feature Engineering ---

# 1. Outlier Removal (using IQR method)
print("Removing outliers...")
cols_to_check = ['views', 'likes', 'comments', 'shares']
Q1 = df[cols_to_check].quantile(0.25)
Q3 = df[cols_to_check].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering condition: Keep rows where ALL values are within bounds
# (Note: Using ANY logic as per typical cleaning, or strict ALL logic. 
# The notebook used: condition = ~((df < lower) | (df > upper)).any(axis=1) -> Remove row if ANY column is outlier
condition = ~((df[cols_to_check] < lower_bound) | (df[cols_to_check] > upper_bound)).any(axis=1)
df = df[condition].copy() 
print(f"Data shape after outlier removal: {df.shape}")

# 2. Advanced Feature Engineering
print("Generating advanced features...")
# Safe log transform (log1p handles 0s)
for col in ['views', 'likes', 'comments', 'shares']:
    df[f'log_{col}'] = np.log1p(df[col])

# Per-second metrics
df['likes_per_sec'] = df['likes'] / df['duration_sec']
df['comments_per_sec'] = df['comments'] / df['duration_sec']
df['shares_per_sec'] = df['views'] / df['duration_sec']

# Re-define Target (just in case filtering changed distribution)
if 'engagement_rate' not in df.columns:
    df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['views']

q33 = df['engagement_rate'].quantile(0.33)
q66 = df['engagement_rate'].quantile(0.66)

def getPercentClass(value):
    if value < q33: return "Low"
    elif value < q66: return "Medium"
    else: return "High"

df['performance_engagement_tertile'] = df['engagement_rate'].apply(getPercentClass)

# 3. Model Setup with Full Features
# Define features to include
numeric_features = [
    'duration_sec', 'hashtags_count',
    'views', 'likes', 'comments', 'shares',
    'engagement_rate', 'likes_per_sec',
    'comments_per_sec', 'shares_per_sec',
    'log_views', 'log_likes', 'log_comments', 'log_shares'
]
categorical_features = ['category', 'upload_hour']

# Explicitly ensure all used features exist in df
missing_cols = [c for c in numeric_features + categorical_features if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}")
    exit()

X = df[numeric_features + categorical_features]
y = df['performance_engagement_tertile']

# Encode Label for Classification
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")

# Define evaluate_models function
def evaluate_models(models, preprocessor, X_train, y_train, k=5):
    results = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('model', model)
        ])
        
        # Accuracy
        acc_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=skf, scoring='accuracy'
        )
        
        # F1-macro
        f1_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=skf, scoring='f1_macro'
        )
        
        results.append({
            'Model': name,
            'Accuracy': acc_scores.mean(),
            'F1 Score (Macro)': f1_scores.mean()
        })
        
    return pd.DataFrame(results)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_model(model, param_grid, preprocessor, X_train, y_train, search_type='grid', n_iter=10):
    """
    Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    """
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    if search_type == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_, search.best_score_

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Run evaluation
print("Running model evaluation...")
results_df = evaluate_models(models, preprocessor, X_train, y_train, k=5)
print("\nEvaluation Results:")
print(results_df)
