from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Menambahkan root directory project ke path agar Python mengenali modul 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import ShopAssistantAgent

# Inisialisasi FastAPI
app = FastAPI(
    title="SmartShopper Assistant API (LangChain)",
    description="REST API untuk asisten belanja pintar menggunakan LangChain, Groq, dan MongoDB.",
    version="1.0.0"
)

# Inisialisasi global instance ShopAssistantAgent untuk demo
agent = ShopAssistantAgent()

# Skema Request untuk endpoint /recommend
class RecommendRequest(BaseModel):
    query: str

# Skema Response untuk endpoint /recommend
class RecommendResponse(BaseModel):
    query: str
    response: str

@app.get("/health")
def health_check():
    """
    Endpoint untuk mengecek status kesehatan API.
    """
    return {
        "status": "healthy",
        "service": "SmartShopper API",
        "engine": "LangChain + Groq"
    }

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendation(request: RecommendRequest):
    """
    Endpoint utama untuk mengirimkan pesan user ke AI Agent.
    Agen akan secara otomatis mendeteksi apakah query merupakan sapaan, pencarian produk, atau FAQ kebijakan toko.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")
        
    try:
        # Panggil fungsi ask pada agen untuk mengeksekusi routing tool dan mengembalikan respon
        response_text = agent.ask(request.query)
        return RecommendResponse(
            query=request.query,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kegagalan pemrosesan: {str(e)}")

@app.delete("/history")
def clear_history():
    """
    Endpoint untuk menghapus riwayat percakapan yang disimpan di memory agen.
    """
    try:
        agent.clear_history()
        return {"status": "success", "message": "Riwayat percakapan berhasil dihapus!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus riwayat: {str(e)}")

# Untuk menjalankan API secara mandiri
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
