from pymongo import MongoClient
from src.config import MONGO_CONNECTION_STRING

# Menggunakan koneksi tunggal (Singleton) agar lebih efisien
_client = None

def get_db():
    """
    Fungsi untuk membuat dan mengembalikan koneksi database MongoDB Atlas.
    """
    global _client
    if _client is None:
        _client = MongoClient(MONGO_CONNECTION_STRING, tlsAllowInvalidCertificates=True)
    return _client['depato_store']

def get_materials():
    """
    Mengambil semua daftar nama material unik dari MongoDB Atlas.
    """
    db = get_db()
    return [doc['name'] for doc in db.materials.find()]

def get_categories():
    """
    Mengambil semua daftar nama kategori produk unik dari MongoDB Atlas.
    """
    db = get_db()
    return [doc['name'] for doc in db.categories.find()]

def get_products_vector_index_name() -> str:
    """
    Menemukan nama index vector search yang aktif untuk collection 'products' secara dinamis.
    Ini berguna jika pengguna memiliki penamaan index yang sedikit berbeda.
    """
    try:
        db = get_db()
        indexes = list(db.products.list_search_indexes())
        for idx in indexes:
            if idx.get("type") == "vectorSearch":
                return idx.get("name")
    except Exception:
        pass
    return "vector_index" # Fallback default

def get_common_info_vector_index_name() -> str:
    """
    Menemukan nama index vector search yang aktif untuk 'common_info' secara dinamis.
    Mengatasi kemungkinan adanya typo penamaan (seperti 'commong_info_vector_index').
    """
    try:
        db = get_db()
        indexes = list(db.common_info.list_search_indexes())
        for idx in indexes:
            if idx.get("type") == "vectorSearch":
                return idx.get("name")
    except Exception:
        pass
    return "common_info_vector_index" # Fallback default

def convert_filter(llm_filter):
    """
    Mengonversi struktur filter JSON hasil LLM menjadi query filter MongoDB yang valid.
    """
    if not llm_filter or not isinstance(llm_filter, dict):
        return {}
    
    operator = llm_filter.get("operator")
    conditions = llm_filter.get("conditions")
    
    if not operator or not conditions:
        return {}
        
    mongo_conditions = []
    for cond in conditions:
        if "operator" in cond and "conditions" in cond:
            nested = convert_filter(cond)
            if nested:
                mongo_conditions.append(nested)
        else:
            field = cond.get("field")
            op = cond.get("operator")
            val = cond.get("value")
            
            # Proteksi typo "filed" vs "field"
            if not field and "filed" in cond:
                field = cond.get("filed")
                
            if field and op is not None and val is not None:
                if op == "==":
                    mongo_conditions.append({field: val})
                elif op == "!=":
                    mongo_conditions.append({field: {"$ne": val}})
                elif op == "<=":
                    mongo_conditions.append({field: {"$lte": val}})
                elif op == ">=":
                    mongo_conditions.append({field: {"$gte": val}})
                elif op == "<":
                    mongo_conditions.append({field: {"$lt": val}})
                elif op == ">":
                    mongo_conditions.append({field: {"$gt": val}})
                    
    if not mongo_conditions:
        return {}
        
    if operator == "AND":
        return {"$and": mongo_conditions}
    elif operator == "OR":
        return {"$or": mongo_conditions}
        
    return {}

def vector_search_products(query_vector, llm_filter=None, limit=10):
    """
    Melakukan Vector Search di collection 'products' menggunakan MongoDB Atlas Vector Search.
    """
    db = get_db()
    collection = db['products']
    
    # Deteksi nama index secara dinamis
    index_name = get_products_vector_index_name()
    
    vector_search_config = {
        "index": index_name,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": limit * 10,
        "limit": limit
    }
    
    mongo_filter = convert_filter(llm_filter)
    if mongo_filter:
        vector_search_config["filter"] = mongo_filter
        
    pipeline = [
        {"$vectorSearch": vector_search_config}
    ]
    
    return list(collection.aggregate(pipeline))

def vector_search_common_info(query_vector, limit=3):
    """
    Melakukan Vector Search di collection 'common_info' untuk pencarian FAQ.
    """
    db = get_db()
    collection = db['common_info']
    
    # Deteksi nama index secara dinamis untuk menangani typo seperti 'commong_info_vector_index'
    index_name = get_common_info_vector_index_name()
    print(f"[Database Search] Menggunakan index vector search: '{index_name}'")
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": limit * 10,
                "limit": limit
            }
        }
    ]
    
    return list(collection.aggregate(pipeline))
