import pandas as pd
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import secrets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, ks_2samp, probplot
import numpy as np
import pylab

# Function to count differing bits between two byte sequences
def count_differing_bits(data1, data2):
    return sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(data1, data2))

# Perform Avalanche Effect Analysis
def avalanche_test_aes(csv_file, text_column):
    data = pd.read_csv(csv_file)
    differing_bits_plaintext, differing_bits_key = [], []
    
    for _, row in data.iterrows():
        plaintext = str(row[text_column]).encode('utf-8')
        key = secrets.token_bytes(16)
        
        # Encrypt with original plaintext
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext1 = cipher.encrypt(pad(plaintext, AES.block_size))

        # Modify one bit in the plaintext
        modified_plaintext = bytearray(plaintext)
        modified_plaintext[0] ^= 0b00000001  # Flip 1st bit
        ciphertext2 = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(bytes(modified_plaintext), AES.block_size))
        differing_bits_plaintext.append(count_differing_bits(ciphertext1, ciphertext2))
        
        # Modify one bit in the key
        modified_key = bytearray(key)
        modified_key[0] ^= 0b00000001  # Flip 1st bit
        ciphertext3 = AES.new(bytes(modified_key), AES.MODE_CBC).encrypt(pad(plaintext, AES.block_size))
        differing_bits_key.append(count_differing_bits(ciphertext1, ciphertext3))

    perform_statistical_tests(differing_bits_plaintext, differing_bits_key)
    plot_avalanche_results(differing_bits_plaintext, differing_bits_key)

# Perform Statistical Tests
def perform_statistical_tests(differing_bits_plaintext, differing_bits_key):
    chi2_plaintext, p_plaintext = chisquare(differing_bits_plaintext)
    chi2_key, p_key = chisquare(differing_bits_key)
    print(f"Chi-Square Test (Plaintext): chi2 = {chi2_plaintext}, p-value = {p_plaintext}")
    print(f"Chi-Square Test (Key): chi2 = {chi2_key}, p-value = {p_key}")
    
    ks_plaintext_stat, ks_plaintext_p = ks_2samp(differing_bits_plaintext, np.random.uniform(min(differing_bits_plaintext), max(differing_bits_plaintext), len(differing_bits_plaintext)))
    ks_key_stat, ks_key_p = ks_2samp(differing_bits_key, np.random.uniform(min(differing_bits_key), max(differing_bits_key), len(differing_bits_key)))
    print(f"Kolmogorov-Smirnov Test (Plaintext): KS Statistic = {ks_plaintext_stat}, p-value = {ks_plaintext_p}")
    print(f"Kolmogorov-Smirnov Test (Key): KS Statistic = {ks_key_stat}, p-value = {ks_key_p}")

# Plotting Function
def plot_avalanche_results(differing_bits_plaintext, differing_bits_key):
    rows = range(1, len(differing_bits_plaintext) + 1)
    
    # Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[differing_bits_plaintext, differing_bits_key], palette=["blue", "red"])
    plt.xticks([0, 1], ["Plaintext Change", "Key Change"])
    plt.ylabel("Number of Differing Bits")
    plt.title("Box Plot of Differing Bits Distribution")
    plt.show()
    
    # Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=[differing_bits_plaintext, differing_bits_key], palette=["blue", "red"])
    plt.xticks([0, 1], ["Plaintext Change", "Key Change"])
    plt.ylabel("Number of Differing Bits")
    plt.title("Violin Plot of Differing Bits Distribution")
    plt.show()
    
    # Scatter Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(rows, differing_bits_plaintext, color='blue', label="Plaintext Change", alpha=0.6)
    plt.scatter(rows, differing_bits_key, color='red', label="Key Change", alpha=0.6)
    plt.xlabel("Row Index")
    plt.ylabel("Number of Differing Bits")
    plt.title("Scatter Plot: Row-wise Avalanche Effect")
    plt.legend()
    plt.show()
    
    # KDE Plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(differing_bits_plaintext, color="blue", label="Plaintext Change", fill=True)
    sns.kdeplot(differing_bits_key, color="red", label="Key Change", fill=True)
    plt.xlabel("Number of Differing Bits")
    plt.ylabel("Density")
    plt.title("KDE Plot of Differing Bits Distribution")
    plt.legend()
    plt.show()
    
    # QQ Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    probplot(differing_bits_plaintext, dist="norm", plot=pylab)
    plt.title("QQ Plot: Plaintext Change")
    
    plt.subplot(1, 2, 2)
    probplot(differing_bits_key, dist="norm", plot=pylab)
    plt.title("QQ Plot: Key Change")
    plt.show()

# Example usage
csv_file_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\Normal.csv"  # Replace with your CSV file path
text_column_name = "tcp_rtt"  # Replace with your column name
avalanche_test_aes(csv_file_path, text_column_name)
