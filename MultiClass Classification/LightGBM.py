import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb  # Import LightGBM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Regular expressions for cleaning column names

# Load the dataset
data_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\4.csv"  # Update with your file path
data = pd.read_csv(data_path)

# Step 1: Inspect the dataset
print("Dataset Shape:", data.shape)
print(data.info())
print(data.head())

# Step 2: Clean column names by removing special characters
data.columns = data.columns.str.replace(r'\W', '_', regex=True)  # Replace any non-word character with '_'

# Step 3: Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'Time']  # Update based on your dataset
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Step 4: Handle missing values
# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns
for col in categorical_cols:
    if not data[col].isnull().all():
        data[col] = data[col].fillna(data[col].mode()[0])

# Step 5: Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 6: Multiclass Classification using LightGBM
if 'attack_type' in data.columns:
    print("\n### Multiclass Classification ###")
    
    # Features (X) and target (y)
    X = data.drop(columns=['attack_type'], errors='ignore')
    y = data['attack_type']
    
    # Redefine numeric_cols based on updated X
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    # Ensure no NaNs in X before scaling
    X[numeric_cols] = X[numeric_cols].fillna(0)  # Replace NaNs with 0 or a suitable value
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the LightGBM model
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm.diagonal().sum()) / cm.sum()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Define target names (custom labels)
    target_names = ['UC1', 'UC2', 'UC3', 'UC4']
    
    # Classification report with custom labels
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Step 7: Anomaly Detection using Isolation Forest
print("\n### Anomaly Detection ###")
X = data.drop(columns=['attack_type'], errors='ignore')  # Exclude target column if exists

# Ensure no NaNs in X for anomaly detection
X[numeric_cols] = X[numeric_cols].fillna(0)  # Replace NaNs with 0 or a suitable value

# Train Isolation Forest model
anomaly_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
anomaly_model.fit(X)

# Predict anomalies (-1: anomaly, 1: normal)
anomaly_labels = anomaly_model.predict(X)
data['anomaly'] = [1 if x == -1 else 0 for x in anomaly_labels]

# Print results
print("Anomalies Detected:", data['anomaly'].sum())
sns.countplot(x='anomaly', data=data)
plt.title('Anomaly Detection Results')
plt.show()

# Analyze anomalies
print("\nAnomaly Data Samples:")
print(data[data['anomaly'] == 1].head())

# Save the dataset with anomalies
output_path = r"C:\Users\ritha\OneDrive\Desktop\EOCS MITM\anomaly_detected_dataset.csv"
data.to_csv(output_path, index=False)
print(f"Dataset with anomalies saved to {output_path}")
