import pandas as pd
import joblib
df = pd.read_csv(r'C:\Users\Yash Gaonkar\Downloads\German\German\german_credit_risk.csv')
print('CSV columns:', df.columns.tolist())
model = joblib.load(r'C:\Users\Yash Gaonkar\Downloads\German\German\german_dt_baseline.pkl')
if hasattr(model, 'feature_names_in_'):
    print('Model expected columns:', list(model.feature_names_in_))
elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'feature_names_in_'):
    print('Model estimator expected columns:', list(model.estimator_.feature_names_in_))
else:
    print('No explicit feature_names_in_ found')
