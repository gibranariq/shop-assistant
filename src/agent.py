from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.pipelines.query_filter import paraphrase_query, generate_metadata_filter
from src.pipelines.product_rag import generate_product_recommendations
from src.pipelines.common_rag import generate_common_info_answer

# Skema input terstandar untuk Tool agar LLM (Groq) mengetahui parameter yang valid
class ToolInputSchema(BaseModel):
    query: str = Field(description="Pertanyaan atau kriteria pencarian dari pelanggan dalam bentuk teks.")

class ShopAssistantAgent:
    """
    Asisten belanja pintar (Shop Assistant) Depato Store.
    Menggunakan pencarian produk dan FAQ kebijakan toko terintegrasi.
    """
    def __init__(self):
        # Menyimpan riwayat percakapan lokal: [{"role": "user"/"assistant", "content": "..."}]
        self.history = []
        
        # Inisialisasi model LLM Groq
        self.llm = ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0
        )
        
        # Daftar Tools yang dapat digunakan Agen untuk menyelesaikan tugasnya
        self.tools = [
            Tool(
                name="product_recommendation",
                description="Gunakan tool ini untuk mencari dan merekomendasikan produk fesyen (seperti baju, gaun, kemeja, celana, aksesoris) berdasarkan kategori, bahan/material, harga, gender, atau merk.",
                func=self.product_recommendation_tool,
                args_schema=ToolInputSchema
            ),
            Tool(
                name="common_information",
                description="Gunakan tool ini untuk menjawab pertanyaan umum seputar kebijakan toko, biaya pengiriman, estimasi waktu kirim, refund (pengembalian barang), metode pembayaran, dan kontak customer service.",
                func=self.common_info_tool,
                args_schema=ToolInputSchema
            )
        ]
        
        # Prompt sistem untuk memandu perilaku Agen
        system_prompt = """
Anda adalah asisten belanja (Shop Assistant) pintar untuk toko online Depato Store.
Tugas Anda adalah membantu pelanggan dengan ramah, sopan, dan solutif.

ATURAN UTAMA:
1. Untuk sapaan, ucapan terima kasih, atau obrolan santai biasa (contoh: "Halo", "Siapa namamu?", "Terima kasih banyak"): Jawab langsung dengan teks biasa. JANGAN memanggil tool apa pun.
2. Untuk pencarian produk fesyen atau rekomendasi pakaian: Panggil tool 'product_recommendation' secara langsung. Parameter 'query' harus diisi dengan query pencarian lengkap.
3. Untuk pertanyaan tentang kebijakan toko, waktu/biaya pengiriman, kebijakan refund, metode pembayaran, atau cara hubungi CS: Panggil tool 'common_information'. Parameter 'query' harus diisi dengan pertanyaan lengkap.
4. Ketika Anda mendapatkan hasil eksekusi dari tool ('product_recommendation' atau 'common_information'), sampaikan hasil tersebut sepenuhnya kepada pelanggan. JANGAN meringkas secara berlebihan, menyembunyikan, atau mengganti informasi penting dari hasil tool tersebut. Gunakan informasi dari tool untuk menjawab pertanyaan pelanggan secara lengkap dan informatif.
5. Jangan mencampur teks jawaban Anda sendiri dengan panggilan tool dalam satu giliran.
6. Jika pelanggan bertanya hal di luar konteks belanja (misal: sains, politik, matematika): Tolak dengan sopan dan ingatkan bahwa Anda hanya bisa membantu untuk informasi toko dan produk.
"""
        
        # Menggunakan custom `create_agent` factory yang secara internal membangun StateGraph
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

    def get_history_string(self) -> str:
        """
        Mengonversi list riwayat percakapan menjadi string gabungan untuk dibaca oleh paraphraser.
        """
        lines = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_langchain_messages(self):
        """
        Mengonversi list riwayat percakapan ke format list pesan LangChain (HumanMessage / AIMessage).
        """
        messages = []
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def product_recommendation_tool(self, query: str) -> str:
        """
        Fungsi di balik tool 'product_recommendation'.
        """
        print(f"\n[Tool Execution] Menjalankan product_recommendation untuk query: '{query}'")
        history_str = self.get_history_string()
        
        # 1. Paraphrase query berdasarkan riwayat percakapan
        paraphrased = paraphrase_query(query, history_str)
        print(f"[Tool Execution] Hasil Paraphrase: '{paraphrased}'")
        
        # 2. Deteksi kriteria filter produk
        filters = generate_metadata_filter(paraphrased)
        print(f"[Tool Execution] Filter Terdeteksi: {filters}")
        
        # 3. Jalankan RAG rekomendasi produk ke database MongoDB Atlas
        return generate_product_recommendations(paraphrased, filters)

    def common_info_tool(self, query: str) -> str:
        """
        Fungsi di balik tool 'common_information'.
        """
        print(f"\n[Tool Execution] Menjalankan common_information untuk query: '{query}'")
        # Jalankan RAG FAQ/kebijakan toko ke database MongoDB Atlas
        return generate_common_info_answer(query)

    def ask(self, query: str) -> str:
        """
        Mengirimkan query dari pengguna ke Agen untuk diproses dan mengembalikan jawaban.
        Mengelola penyimpanan riwayat secara otomatis.
        """
        # Siapkan seluruh rangkaian pesan (riwayat + pesan terbaru)
        messages = self.get_langchain_messages() + [HumanMessage(content=query)]
        
        # Jalankan eksekusi agen dengan input state 'messages'
        res = self.agent.invoke({
            "messages": messages
        })
        
        # Jawaban akhir agen terletak pada pesan terakhir dari output state
        response_text = res["messages"][-1].content
        
        # Simpan pesan terbaru ke riwayat lokal
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response_text})
        
        return response_text

    def clear_history(self):
        """
        Mengosongkan riwayat percakapan.
        """
        self.history = []
