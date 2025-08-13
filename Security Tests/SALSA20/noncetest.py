import pandas as pd
import secrets
from Crypto.Cipher import Salsa20
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from scipy import stats
import numpy as np

# Function to generate a random nonce and encrypt a value
nonce_dict = defaultdict(list)  # Dictionary to track nonces

def encrypt_value_with_nonce_logging(value, key):
    nonce = secrets.token_bytes(8)
    nonce_dict[value].append(nonce)
    value_bytes = str(value).encode('utf-8')
    cipher = Salsa20.new(key=key, nonce=nonce)
    encrypted_value = cipher.encrypt(value_bytes)
    return encrypted_value.hex()

def test_nonce_uniqueness():
    all_nonces = [nonce for nonces in nonce_dict.values() for nonce in nonces]
    root = tk.Tk()
    root.withdraw()
    if len(all_nonces) == len(set(all_nonces)):
        print("Nonce Uniqueness Test Passed: All nonces are unique.")
        messagebox.showinfo("Test Passed", "Nonce Uniqueness Test Passed: All nonces are unique.")
    else:
        print("Nonce Uniqueness Test Failed: Duplicate nonces detected.")
        messagebox.showerror("Test Failed", "Nonce Uniqueness Test Failed: Duplicate nonces detected.")

file_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\normal.csv"
data = pd.read_csv(file_path)

encrypted_data = data.copy()

for col in encrypted_data.columns:
    encrypted_data[col] = encrypted_data[col].astype('object')

salsa20_key = secrets.token_bytes(32)

for index, row in data.iterrows():
    for col in data.columns:
        encrypted_data.at[index, col] = encrypt_value_with_nonce_logging(row[col], salsa20_key)

test_nonce_uniqueness()

output_file = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\encrypt_salsa20_with_nonce_test.csv"
encrypted_data.to_csv(output_file, index=False)

print(f"Encrypted dataset saved to: {output_file}")

original_data_sizes = [len(str(value)) for col in data.columns for value in data[col]]
encrypted_data_lengths = [
    len(str(value)) for col in encrypted_data.columns for value in encrypted_data[col]
]

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=[original_data_sizes, encrypted_data_lengths], palette=["green", "orange"])
plt.xticks([0, 1], ["Original Data", "Encrypted Data"])
plt.ylabel("Value Length (Characters)", fontsize=14)
plt.title("Box Plot of Original vs Encrypted Data Sizes", fontsize=16, pad=20)
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=[original_data_sizes, encrypted_data_lengths], palette=["blue", "red"])
plt.xticks([0, 1], ["Original Data", "Encrypted Data"])
plt.ylabel("Value Length (Characters)", fontsize=14)
plt.title("Violin Plot of Original vs Encrypted Data Sizes", fontsize=16, pad=20)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(original_data_sizes)), original_data_sizes, color="blue", alpha=0.6, label="Original Data")
plt.scatter(range(len(encrypted_data_lengths)), encrypted_data_lengths, color="red", alpha=0.6, label="Encrypted Data")
plt.xlabel("Index", fontsize=14)
plt.ylabel("Value Length (Characters)", fontsize=14)
plt.title("Scatter Plot of Original vs Encrypted Data Sizes", fontsize=16, pad=20)
plt.legend()
plt.show()

# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(original_data_sizes, color="green", label="Original Data", shade=True)
sns.kdeplot(encrypted_data_lengths, color="purple", label="Encrypted Data", shade=True)
plt.xlabel("Value Length (Characters)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("KDE Plot of Original vs Encrypted Data Sizes", fontsize=16, pad=20)
plt.legend()
plt.show()

# QQ Plot
plt.figure(figsize=(10, 6))
stats.probplot(original_data_sizes, dist="norm", plot=plt)
plt.title("QQ Plot of Original Data", fontsize=16, pad=20)
plt.show()

plt.figure(figsize=(10, 6))
stats.probplot(encrypted_data_lengths, dist="norm", plot=plt)
plt.title("QQ Plot of Encrypted Data", fontsize=16, pad=20)
plt.show()

# Statistical Analysis
t_stat, p_val = stats.ttest_rel(original_data_sizes, encrypted_data_lengths)
print(f"T-statistic: {t_stat}, p-value: {p_val}")

ks_stat, ks_p_val = stats.ks_2samp(original_data_sizes, encrypted_data_lengths)
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_val}")

nonce_counts = [len(nonce_dict[col]) for col in nonce_dict]
expected = [sum(nonce_counts) / len(nonce_counts)] * len(nonce_counts)
chi2_stat, chi2_p_val = stats.chisquare(nonce_counts, expected)
print(f"Chi-Square Statistic: {chi2_stat}, p-value: {chi2_p_val}")

nonce_distribution = np.array(nonce_counts) / sum(nonce_counts)
entropy = -np.sum(nonce_distribution * np.log2(nonce_distribution))
print(f"Nonce Entropy: {entropy}")
