import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Crypto.Cipher import Salsa20
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Hash import SHA256
from scipy.stats import chisquare
from collections import Counter

# AES-Based Key and IV Generation
def generate_aes_key_iv(password, key_path, iv_path):
    """
    Generate AES-compatible key and IV using a password directly, and save them to the specified file paths.
    """
    # Derive a 256-bit key (32 bytes) directly from password using AES
    key = password.encode('utf-8')[:32]  # Use first 32 bytes of the password for the key
    if len(key) < 32:
        key = key.ljust(32, b'\0')  # If password is shorter, pad with null bytes

    # AES IV is 128 bits (16 bytes), use random IV generation
    iv = get_random_bytes(16)

    # Save key and IV to files
    with open(key_path, 'wb') as key_file:
        key_file.write(key)
    with open(iv_path, 'wb') as iv_file:
        iv_file.write(iv)

    print(f"AES Key saved to: {key_path}")
    print(f"AES IV saved to: {iv_path}")

    return key, iv

# Salsa20 Encryption and Decryption Functions
def encrypt_value_salsa20(value, key, iv):
    """
    Encrypt a single value using Salsa20.
    """
    if pd.isnull(value):  # Skip null values
        return value
    try:
        value_bytes = str(value).encode()  # Convert to bytes
        cipher = Salsa20.new(key=key, nonce=iv)
        encrypted = cipher.encrypt(value_bytes)
        return encrypted.hex()  # Store as hexadecimal string
    except Exception as e:
        print(f"Encryption Error - Value: {value}, Error: {e}")
        return None

def decrypt_value_salsa20(value, key, iv):
    """
    Decrypt a single value using Salsa20.
    """
    if pd.isnull(value) or not isinstance(value, str):  # Skip non-string or null values
        return value

    try:
        # Check if the value is a valid hexadecimal string
        if all(c in '0123456789abcdefABCDEF' for c in value):  # Simple check for hex chars
            encrypted_bytes = bytes.fromhex(value)  # Convert hex to bytes
            cipher = Salsa20.new(key=key, nonce=iv)
            decrypted = cipher.decrypt(encrypted_bytes)
            return decrypted.decode()  # Convert to string
        else:
            # If not a valid hex string, return the value as is
            return value
    except Exception as e:
        print(f"Decryption Error - Value: {value}, Error: {e}")
        return None

# Load Dataset
file_path =r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\Normal.csv"
cleaned_data = pd.read_csv(file_path)

# Key and IV Paths
key_path = "aes_key.key"
iv_path = "aes_iv.key"

# Password for AES Key and IV Generation
password = "your_secure_password_here"

# Generate or Load Key and IV
if not (os.path.exists(key_path) and os.path.exists(iv_path)):
    key, iv = generate_aes_key_iv(password, key_path, iv_path)
else:
    with open(key_path, 'rb') as key_file:
        key = key_file.read()
    with open(iv_path, 'rb') as iv_file:
        iv = iv_file.read()

# Derive Salsa20 key from AES key
salsa20_key = key[:32]  # Salsa20 key size is 256 bits (32 bytes)
salsa20_iv = iv[:8]     # Salsa20 IV size is 64 bits (8 bytes)

# Debugging Key and IV
print(f"Loaded AES Key: {key.hex()}")
print(f"Loaded AES IV: {iv.hex()}")
print(f"Derived Salsa20 Key: {salsa20_key.hex()}")
print(f"Derived Salsa20 IV: {salsa20_iv.hex()}")

# Step 1: Encrypt Dataset
encrypted_file_path = "encrypt_salsa20_from_aes.csv"
start_time = time.time()

# Encrypt all data in the dataset
encrypted_data = cleaned_data.apply(lambda col: col.map(lambda x: encrypt_value_salsa20(x, salsa20_key, salsa20_iv)))

encryption_time = time.time() - start_time
encrypted_data.to_csv(encrypted_file_path, index=False)
print(f"Encrypted dataset saved to: {encrypted_file_path}")

# Step 2: Decrypt Dataset
decrypted_file_path = "decrypt_salsa20_from_aes.csv"
start_time = time.time()

# Decrypt all data in the dataset
decrypted_data = encrypted_data.apply(lambda col: col.map(lambda x: decrypt_value_salsa20(x, salsa20_key, salsa20_iv)))

decryption_time = time.time() - start_time
decrypted_data.to_csv(decrypted_file_path, index=False)
print(f"Decrypted dataset saved to: {decrypted_file_path}")

# Visualization 1: File Size Comparison
def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # File size in KB

original_size = get_file_size(file_path)
encrypted_size = get_file_size(encrypted_file_path)
decrypted_size = get_file_size(decrypted_file_path)

file_sizes = [original_size, encrypted_size, decrypted_size]
file_labels = ['Original', 'Encrypted', 'Decrypted']


"""
plt.bar(file_labels, file_sizes, color=['blue', 'green', 'red'])
plt.ylabel('File Size (KB)')
plt.title('File Size Comparison Before and After Encryption/Decryption')
plt.show()


# Visualization 2: Column-wise Data Length Comparison
plt.figure(figsize=(12, 6))

columns = cleaned_data.columns
original_lengths = cleaned_data.astype(str).map(len).sum()
encrypted_lengths = encrypted_data.astype(str).map(len).sum()
decrypted_lengths = decrypted_data.astype(str).map(len).sum()


plt.plot(columns, original_lengths, marker='o', label="Original Data Length", color='blue')
plt.plot(columns, encrypted_lengths, marker='s', label="Encrypted Data Length", color='green')
plt.plot(columns, decrypted_lengths, marker='^', label="Decrypted Data Length", color='red')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.xlabel("Columns")
plt.ylabel("Total Length (Characters)")
plt.title("Comparison of Original, Encrypted, and Decrypted Data Length per Column")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
"""

# Function to get character frequency distribution from a dataset
def get_char_frequency(df, smoothing=1e-10):
    char_counts = Counter()
    total_chars = 0

    for col in df.columns:
        for value in df[col].dropna().astype(str):
            char_counts.update(value)  # Count characters in each value
            total_chars += len(value)

    # Convert to frequency distribution
    char_freq = {char: (count + smoothing) / (total_chars + len(char_counts) * smoothing) for char, count in char_counts.items()}
    return char_freq

# Load original and encrypted datasets
original_data = pd.read_csv(r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\Normal.csv")
encrypted_data = pd.read_csv("encrypt_salsa20_from_aes.csv")

# Get character frequency distributions
original_freq = get_char_frequency(original_data)
encrypted_freq = get_char_frequency(encrypted_data)

# Create a common set of characters
all_chars = set(original_freq.keys()).union(set(encrypted_freq.keys()))

# Prepare frequency lists for chi-square test
original_values = [original_freq.get(char, 1e-10) for char in all_chars]  # Apply smoothing
encrypted_values = [encrypted_freq.get(char, 1e-10) for char in all_chars]

# Perform Chi-Square Test
chi_stat, p_value = chisquare(encrypted_values, f_exp=original_values)

# Print Results
print(f"Chi-Square Statistic: {chi_stat}")
print(f"P-Value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Significant difference in character distributions after encryption (Reject H0).")
else:
    print("No significant difference in character distributions after encryption (Fail to Reject H0).")


