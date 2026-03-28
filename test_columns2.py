import pandas as pd
import joblib

df = pd.read_csv(r'C:\Users\Yash Gaonkar\Downloads\German\German\german_credit_risk.csv')
content = ''
content += f'CSV columns: {df.columns.tolist()}\n'

try:
    model = joblib.load(r'C:\Users\Yash Gaonkar\Downloads\German\German\german_dt_baseline.pkl')
    if hasattr(model, 'feature_names_in_'):
        content += f'Model expected columns: {list(model.feature_names_in_)}\n'
    elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'feature_names_in_'):
        content += f'Model estimator expected columns: {list(model.estimator_.feature_names_in_)}\n'
    else:
        content += 'No explicit feature_names_in_ found\n'
except Exception as e:
    content += f'Error loading model: {e}\n'

with open('test_columns_out.txt', 'w', encoding='utf-8') as f:
    f.write(content)
