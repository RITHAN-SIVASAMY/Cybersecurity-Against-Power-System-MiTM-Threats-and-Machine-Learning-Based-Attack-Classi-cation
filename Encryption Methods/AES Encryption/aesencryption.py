import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

# Load the dataset
data_path = r"C:\Users\ritha\OneDrive\Desktop\EOCS MITM\encryption\encryption\aes\dataset.csv"
data = pd.read_csv(data_path)

# Drop unnecessary columns if they exist
data = data.drop(columns=[col for col in ['Unnamed: 0', 'Time'] if col in data.columns])

# Handle missing values
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

if not categorical_cols.empty:
    mode_values = {col: data[col].mode().iloc[0] for col in categorical_cols if not data[col].isnull().all()}
    data = data.fillna(value=mode_values)

# Encode categorical variables
for col in categorical_cols:
    if not data[col].isnull().all():
        data[col] = LabelEncoder().fit_transform(data[col])

# Ensure the target column exists
if 'snort_alert' not in data.columns:
    raise ValueError("The target column 'snort_alert' is missing in the dataset.")

# Identify features and target variable
X = data.drop(columns=['snort_alert'], errors='ignore')  # Features
y = data['snort_alert']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Filter rows classified as "normal" (snort_alert = 0)
normal_data = data[data['snort_alert'] == 0]

# Save normal data to a CSV file
normal_data_path = r"C:\Users\ritha\OneDrive\Desktop\EOCS MITM\encryption\encryption\aes\normal_data.csv"
normal_data.to_csv(normal_data_path, index=False)
print(f"Normal data saved to {normal_data_path}")

# AES Encryption setup
def aes_encrypt(data, key):
    backend = default_backend()
    iv = os.urandom(16)  # Generate a random IV (16 bytes for AES-128)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()

    # Add padding to data to make it AES-compatible
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()

    # Encrypt the data
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data, iv

# Generate a random AES key (16 bytes for AES-128)
aes_key = os.urandom(16)

# Encrypt the "normal" data
hashed_data = []
for _, row in normal_data.iterrows():
    row_str = row.to_json()  # Convert row to JSON string
    encrypted_row, iv = aes_encrypt(row_str, aes_key)
    hashed_data.append((encrypted_row, iv))

# Save hashed data
hashed_data_path = r"C:\Users\ritha\OneDrive\Desktop\EOCS MITM\encryption\encryption\aes\encrypt_normal_data.csv"
joblib.dump(hashed_data, hashed_data_path)
print(f"Hashed data saved to {hashed_data_path}")

# Save AES key
key_path = r"C:\Users\ritha\OneDrive\Desktop\EOCS MITM\encryption\encryption\aes\aes_key.key"
with open(key_path, "wb") as key_file:
    key_file.write(aes_key)
print(f"AES key saved to {key_path}")
