import sys
import os

# Menambahkan root directory ke path agar Python mengenali modul 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import ShopAssistantAgent

def run_test():
    print("=== Memulai Pengujian AI Agent Shop Assistant ===")
    
    # Inisialisasi Agen
    print("Menginisialisasi Agen (Menghubungkan ke Groq LLM & MongoDB)...")
    agent = ShopAssistantAgent()
    print("Agen berhasil diinisialisasi.\n")
    
    # Kumpulan query tes untuk memverifikasi mekanisme routing & tool panggilan
    test_queries = [
        "Halo, asisten! Siapa namamu dan apa tugasmu?",
        "Berapa lama waktu pengiriman barang ke wilayah Jawa Timur?",
        "Saya ingin mencari kemeja berbahan Polyester dengan harga di bawah 40 dollar.",
        "Bagaimana kebijakan jika barang yang saya terima ternyata cacat?",
        "Terima kasih banyak atas informasinya!"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 60)
        print(f"Test #{i}: {query}")
        print("=" * 60)
        
        try:
            # Jalankan kueri ke Agen
            response = agent.ask(query)
            print(f"\n[Jawaban Asisten]:\n{response}")
        except Exception as e:
            print(f"\n[ERROR terjadi saat eksekusi]: {str(e)}")
            
    print("\n=== Pengujian Selesai ===")

if __name__ == "__main__":
    run_test()
