import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

xgb_rf = xgb.XGBRFRegressor(random_state=42)