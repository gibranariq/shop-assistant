import sys
import os
import json
import pandas as pd
from pymongo import MongoClient

# Menambahkan folder root project ke path agar python bisa mengenali modul 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MONGO_CONNECTION_STRING
from src.embeddings import embeddings

def seed_database():
    print("=== Memulai Proses Seeding Database ===")
    
    # 1. Hubungkan ke MongoDB Atlas
    # tlsAllowInvalidCertificates digunakan agar aman dari kendala SSL lokal pada beberapa sistem operasi
    client = MongoClient(MONGO_CONNECTION_STRING, tlsAllowInvalidCertificates=True)
    db = client['depato_store']
    
    # Kumpulan nama collection yang akan digunakan
    products_col = db['products']
    common_info_col = db['common_info']
    materials_col = db['materials']
    categories_col = db['categories']
    
    # 2. Proses dan simpan Data Produk (jika koleksi kosong)
    if products_col.count_documents({}) == 0:
        print("\nLoading data/datasets.pkl...")
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/datasets.pkl'))
        
        if not os.path.exists(dataset_path):
            print(f"Error: File {dataset_path} tidak ditemukan!")
            return
            
        df = pd.read_pickle(dataset_path)
        
        # Hilangkan baris yang kosong atau tidak valid
        df = df.dropna(subset=['title'])
        
        print(f"Ditemukan {len(df)} data produk. Menghitung embedding...")
        
        # Persiapkan data dokumen
        documents = []
        for index, row in df.iterrows():
            descriptions = row["description"]
            if isinstance(descriptions, list):
                descriptions = " ".join(descriptions)
            elif isinstance(descriptions, str):
                descriptions = descriptions.strip("[]").strip("''")
            else:
                descriptions = ""
                
            content = f"{row['title']}\n {descriptions}"
            
            # Buat representasi dictionary yang akan disimpan ke MongoDB
            doc = {
                "content": content,
                "meta": {
                    "asin": row['asin'] if pd.notna(row['asin']) else "",
                    "title": row['title'] if pd.notna(row['title']) else "",
                    "brand": row['brand'] if pd.notna(row['brand']) else "Generic",
                    "price": float(row['price']) if pd.notna(row['price']) and str(row['price']).replace('.', '', 1).isdigit() else 0.0,
                    "gender": row['gender'] if pd.notna(row['gender']) else "unisex",
                    "material": row['material'] if pd.notna(row['material']) else "Unknown",
                    "category": row['category'] if pd.notna(row['category']) else "Other"
                }
            }
            documents.append(doc)
            
        # Untuk menghemat waktu dan meminimalkan pemanggilan, kita lakukan embedding per batch
        print("Menghitung embedding produk (bisa memakan waktu beberapa menit)...")
        texts_to_embed = [doc["content"] for doc in documents]
        
        # Menghitung embedding menggunakan HuggingFace model
        embeddings_list = embeddings.embed_documents(texts_to_embed)
        
        # Tambahkan vektor embedding ke masing-masing dokumen
        for doc, emb in zip(documents, embeddings_list):
            doc["embedding"] = emb
            
        # Simpan ke MongoDB
        print(f"Menyimpan {len(documents)} produk ke MongoDB collection 'products'...")
        products_col.insert_many(documents)
        print("Data produk berhasil disimpan!")
        
        # Simpan Materials & Categories Unik ke MongoDB
        print("Menyimpan daftar material dan kategori unik...")
        unique_materials = list(set([doc["meta"]["material"] for doc in documents if doc["meta"]["material"]]))
        unique_categories = list(set([doc["meta"]["category"] for doc in documents if doc["meta"]["category"]]))
        
        materials_col.delete_many({}) # Reset jika ada isi sebelumnya
        categories_col.delete_many({})
        
        materials_col.insert_many([{"name": m} for m in unique_materials])
        categories_col.insert_many([{"name": c} for c in unique_categories])
        print("Daftar material dan kategori berhasil disimpan!")
        
    else:
        print("\nCollection 'products' sudah berisi data. Melewati proses seeding produk.")
        
    # 3. Proses dan simpan Data Common Information (FAQ)
    print("\nLoading data/common_info.json...")
    faq_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/common_info.json'))
    
    if os.path.exists(faq_path):
        with open(faq_path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
            
        print(f"Ditemukan {len(faqs)} FAQ. Menghitung embedding...")
        
        faq_documents = []
        for faq in faqs:
            question = faq["question"]
            answer = faq["answer"]
            content = f"Question: {question}\nAnswer: {answer}"
            
            # Buat representasi dictionary FAQ
            doc = {
                "question": question,
                "answer": answer,
                "content": content
            }
            faq_documents.append(doc)
            
        # Hitung embedding FAQ
        texts_to_embed = [doc["content"] for doc in faq_documents]
        embeddings_list = embeddings.embed_documents(texts_to_embed)
        
        # Gabungkan vektor embedding
        for doc, emb in zip(faq_documents, embeddings_list):
            doc["embedding"] = emb
            
        # Bersihkan koleksi faq lama dan masukkan yang baru
        print("Menghapus data FAQ lama di collection 'common_info'...")
        common_info_col.delete_many({})
        
        print(f"Menyimpan {len(faq_documents)} FAQ ke collection 'common_info'...")
        common_info_col.insert_many(faq_documents)
        print("Data FAQ berhasil disimpan!")
    else:
        print(f"Error: File {faq_path} tidak ditemukan!")
        
    print("\n=== Proses Seeding Selesai! ===")
    print("\n[PENTING] Silakan buat Vector Search Index berikut di MongoDB Atlas Anda:")
    print("1. Collection: products | Nama Index: vector_index")
    print("   Definisi Index:")
    print('   {\n     "fields": [\n       {\n         "numDimensions": 768,\n         "path": "embedding",\n         "similarity": "cosine",\n         "type": "vector"\n       }\n     ]\n   }')
    print("\n2. Collection: common_info | Nama Index: common_info_vector_index")
    print("   Definisi Index:")
    print('   {\n     "fields": [\n       {\n         "numDimensions": 768,\n         "path": "embedding",\n         "similarity": "cosine",\n         "type": "vector"\n       }\n     ]\n   }')

if __name__ == "__main__":
    seed_database()
