import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------
# BIAS BUSTER ML PIPELINE DEMONSTRATION
# "Zero Manual Encoding" Architecture
# ---------------------------------------------------------

# 1. Load your raw dataset exactly as it will be uploaded to BiasBuster
# Example mock data:
data = {
    'Checking account': ['little', 'moderate', 'rich', 'little'],
    'Housing': ['own', 'free', 'rent', 'own'],
    'Sex': ['male', 'female', 'male', 'female'],
    'Credit_amount': [1000, 2500, 500, 3000],
    'Risk': ['bad', 'good', 'good', 'bad']
}
df = pd.DataFrame(data)

# Strictly split RAW data. No pd.get_dummies()!!
X_raw = df.drop(columns=['Risk'])
y = df['Risk']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# 2. Define exactly which columns need text-encoding vs scaling mathematically
categorical_cols = ['Checking account', 'Housing', 'Sex']
numeric_cols = ['Credit_amount']

# 3. Create a ColumnTransformer to handle the data preprocessing INTERNALLY
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 4. Construct the formal Scikit-Learn Pipeline
# This guarantees structural enforcement over raw inferences. BiasBuster relies on this!
ml_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 5. Train the pipeline DIRECTLY on the raw string dataframe
ml_pipeline.fit(X_train_raw, y_train)

# 6. Test inference dynamically using raw testing text strings!
# Output confirms no "Feature mismatch" or "Missing Checking_account_little" errors.
print("Predicting against structurally raw DataFrame inference test:")
y_pred = ml_pipeline.predict(X_test_raw)
print(y_pred)

# 7. Save the fully isolated pipeline object.
# When BiasBuster loads this .pkl, it passes raw text columns directly to .fit() and .predict()
# and everything mathematically cascades internally perfectly.
joblib.dump(ml_pipeline, "structural_pipeline.pkl")
print("Pipeline saved successfully!")
