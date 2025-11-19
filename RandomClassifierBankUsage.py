import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('Financial_inclusion_dataset.csv')
print(data.info())
print(data.describe())

# 2. Preprocessing
# creating a profile data to get more insights on the data
from ydata_profiling import ProfileReport
profile = ProfileReport(data, title = 'Bank Account Usage across East Africa')
profile.to_file('BankAccount.html')

print(data.isnull().sum())
data.drop_duplicates(inplace=True)

# Drop ID
X = data.drop(['bank_account', 'uniqueid'], axis=1, errors='ignore')
y = data['bank_account']

# Define columns
cat_cols = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
            'relationship_with_head', 'marital_status', 'education_level', 'job_type']
num_cols = ['household_size', 'age_of_respondent', 'year']

# Encode Categorical Data
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# 3Random Forest Classifier
# ---------------------------------------------------------
print("Training Random Forest... this might take 5-10 seconds...")

rf_model = RandomForestClassifier(n_estimators=100,
                                  class_weight='balanced',
                                  random_state=42,
                                  max_depth=10)

rf_model.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

#  Save the "Brain" (The trained model)
joblib.dump(rf_model, 'random_forest_model.pkl')

# 2. Save the "Dictionary" (The exact column names used during training)
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')

