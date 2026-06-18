from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.embeddings import embeddings
from src.database import vector_search_common_info

# Inisialisasi model LLM Groq.
# Temperature diset ke 0.0 agar jawaban asisten patuh pada dokumen kebijakan toko (tidak berhalusinasi).
llm = ChatGroq(
    model_name=GROQ_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.0
)

# Template prompt untuk RAG Kebijakan Umum Toko
COMMON_RAG_TEMPLATE = """
Anda adalah asisten customer service toko online (Depato Store) yang ramah, sopan, dan profesional.
Tugas Anda adalah menjawab pertanyaan pelanggan mengenai informasi umum toko, metode pembayaran, pengiriman, atau pengembalian (refund) HANYA berdasarkan informasi (konteks) yang disediakan di bawah ini.

Aturan Penting:
1. Jawablah secara singkat, jelas, dan ramah menggunakan Bahasa Indonesia.
2. Gunakan HANYA informasi dari konteks di bawah ini. Jangan menambahkan informasi dari luar.
3. Jika informasi yang ditanyakan tidak tercantum di dalam konteks, katakan dengan sopan bahwa Anda tidak tahu atau sarankan untuk menghubungi Customer Service kami via WhatsApp (+62-812-3456-7890) atau email (support@depatostore.com).

Konteks Informasi Toko:
{context}

Pertanyaan Pelanggan: {query}

Jawaban Asisten:
"""

def generate_common_info_answer(query: str, limit: int = 3) -> str:
    """
    Menjawab pertanyaan umum pelanggan (FAQ/kebijakan toko) menggunakan pencarian RAG di database MongoDB.
    
    Proses:
    1. Mengubah query menjadi representasi vektor.
    2. Mencari dokumen FAQ terdekat di collection 'common_info'.
    3. Menggabungkan FAQ tersebut sebagai konteks masukan LLM.
    4. Meminta LLM merumuskan jawaban yang bersahabat sesuai konteks kebijakan toko.
    """
    # Step 1: Hitung embedding dari query
    query_vector = embeddings.embed_query(query)
    
    # Step 2: Cari FAQ terdekat di database MongoDB Atlas
    matched_faqs = vector_search_common_info(query_vector, limit=limit)
    
    if not matched_faqs:
        return (
            "Maaf, informasi tersebut tidak tersedia saat ini. "
            "Silakan hubungi Customer Service kami via WhatsApp di +62-812-3456-7890 untuk bantuan lebih lanjut."
        )
        
    # Step 3: Gabungkan FAQ menjadi teks konteks
    context_list = []
    for faq in matched_faqs:
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        context_list.append(f"Pertanyaan: {question}\nJawaban: {answer}")
        
    context = "\n---\n".join(context_list)
    
    # Step 4: Kirim ke LLM untuk merumuskan jawaban akhir
    prompt = ChatPromptTemplate.from_template(COMMON_RAG_TEMPLATE)
    chain = prompt | llm
    
    res = chain.invoke({
        "query": query,
        "context": context
    })
    
    return res.content.strip()
