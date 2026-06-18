import os
from dotenv import load_dotenv

# Membaca file .env yang berisi variabel rahasia (API Key dan Connection String)
load_dotenv()

# Mengambil variabel GROQ_API_KEY dari .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Mengambil variabel MONGO_CONNECTION_STRING dari .env
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

# Mengambil model Groq yang digunakan (default: llama-3.3-70b-versatile)
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# Validasi untuk memastikan variabel tidak kosong
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY tidak ditemukan di .env file! Silakan isi terlebih dahulu.")

if not MONGO_CONNECTION_STRING:
    raise ValueError("MONGO_CONNECTION_STRING tidak ditemukan di .env file! Silakan isi terlebih dahulu.")

