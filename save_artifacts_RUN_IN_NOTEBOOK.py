# ==========================================================
# RUN THIS CELL IN YOUR JUPYTER NOTEBOOK (not standalone)
# It saves everything the web app needs: model, scaler, feature list
# Matches your actual pipeline: feature_cols, X_train, scaler, models dict
# ==========================================================

import joblib
import os

os.makedirs('model', exist_ok=True)

# 1) Save the trained Logistic Regression model
joblib.dump(models['Logistic Regression'], 'model/best_model.pkl')

# 2) Save the fitted scaler used to create X_train_sc / X_test_sc
#    (in your notebook this is `scaler = StandardScaler()` fit on X_train)
joblib.dump(scaler, 'model/scaler.pkl')

# 3) Save the exact list/order of feature columns the model expects
#    (in your notebook this is already defined as `feature_cols`)
joblib.dump(feature_cols, 'model/feature_cols.pkl')

print("Saved: model/best_model.pkl, model/scaler.pkl, model/feature_cols.pkl")
print("Feature columns (in order):", feature_cols)
