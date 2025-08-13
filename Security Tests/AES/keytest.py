from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import tkinter as tk
from tkinter import messagebox

def aes_key_length_test_aes128():
    try:
        # Define a key (Change this to test various lengths)
        key = b"thiskeyisnot16jj"  # Example invalid key: 14 bytes
        print(f"Key length: {len(key)} bytes")

        # Validate key length for AES-128
        if len(key) != 16:
            raise ValueError(f"Invalid key length: {len(key)} bytes. Key must be 16 bytes for AES-128.")

        # Define plaintext
        plaintext = b"This is a test message for AES encryption"

        # Encrypt the plaintext
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv  # Initialization vector
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        print(f"Ciphertext: {ciphertext}")

        # Decrypt the ciphertext
        decipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_text = unpad(decipher.decrypt(ciphertext), AES.block_size)
        print(f"Decrypted text: {decrypted_text.decode()}")

        # Test successful
        if plaintext == decrypted_text:
            print("AES-128 key length test successful: Encryption and decryption worked as expected.")
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showinfo("Test Passed", "AES-128 key length test successful: Encryption and decryption worked as expected.")

    except ValueError as e:
        print(f"ValueError: {e}")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Test Failed", f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Test Failed", f"An error occurred: {e}")

# Run the test
aes_key_length_test_aes128()
