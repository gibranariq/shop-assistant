from langchain_huggingface import HuggingFaceEmbeddings

# Inisialisasi model embedding SentenceTransformers menggunakan HuggingFace lewat LangChain.
# Model 'sentence-transformers/all-mpnet-base-v2' menghasilkan embedding dengan 768 dimensi.
# Model ini akan digunakan untuk mengubah query teks menjadi representasi vektor (angka)
# agar bisa dicocokkan dengan data produk dan data FAQ di MongoDB.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
