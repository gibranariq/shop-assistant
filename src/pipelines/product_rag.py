from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.embeddings import embeddings
from src.database import vector_search_products

# Inisialisasi model LLM Groq
llm = ChatGroq(
    model_name=GROQ_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.3  # Gunakan sedikit kreativitas (0.3) untuk membuat rekomendasi yang menarik
)

# Template prompt RAG rekomendasi produk yang disesuaikan dari website.py
PRODUCT_RAG_TEMPLATE = """
You are a helpful shop assistant that will give products recommendation based on user query and metadata filtering.

Your task is to generate a list of products that best match the query.

The output should be a list of products in the following format:

<summary_of_query>
<index>. <product_name> 
Price: <product_price>
Material: <product_material>
Category: <product_category>
Brand: <product_brand>
Recommendation: <product_recommendation>

From the format above, you should pay attention to the following:
1. <summary_of_query> should be a short summary of the query.
2. <index> should be a number starting from 1.
3. <product_name> should be the name of the product, this product name can be found from the product_name field.
4. <product_price> should be the price of the product, this product price can be found from the product_price field.
5. <product_material> should be the material of the product, this product material can be found from the product_material field.
6. <product_category> should be the category of the product, this product category can be found from the product_category field.
7. <product_brand> should be the brand of the product, this product brand can be found from the product_brand field.
8. <product_recommendation> should be the recommendation of the product, you should give a recommendation why this product is recommended, please pay attention to the product_content field.

You should only return the list of products that best match the query, do not return any other information.

if there is no matching product below, please say so.

The query is: {query}

Products list:
{products_context}

Answer:
"""

def generate_product_recommendations(paraphrased_query: str, metadata_filter: dict, limit: int = 5) -> str:
    """
    Menghasilkan teks rekomendasi produk berdasarkan query user dan filter yang ditentukan.
    
    Proses:
    1. Membuat vector embedding dari query.
    2. Mencari produk yang relevan di MongoDB Atlas (dengan pre-filter).
    3. Menggabungkan informasi produk menjadi konteks teks.
    4. Mengirimkan konteks ke LLM untuk memformat jawaban rekomendasi.
    """
    # Step 1: Hitung embedding dari query user
    query_vector = embeddings.embed_query(paraphrased_query)
    
    # Step 2: Lakukan Vector Search di MongoDB Atlas dengan filter metadata
    matched_products = vector_search_products(query_vector, llm_filter=metadata_filter, limit=limit)
    
    # Step 3: Format data produk menjadi teks konteks untuk LLM
    if not matched_products:
        return "Maaf, kami tidak dapat menemukan produk yang sesuai dengan kriteria Anda saat ini."
        
    products_context_list = []
    for i, p in enumerate(matched_products):
        meta = p.get("meta", {})
        title = meta.get("title", "Unknown Product")
        price = meta.get("price", "N/A")
        material = meta.get("material", "Unknown")
        category = meta.get("category", "Unknown")
        brand = meta.get("brand", "Unknown")
        content = p.get("content", "")
        
        prod_str = (
            f"===========================================================\n"
            f"{i+1}. product_name: {title}\n"
            f"product_price: {price}\n"
            f"product_material: {material}\n"
            f"product_category: {category}\n"
            f"product_brand: {brand}\n"
            f"product_content: {content}\n"
        )
        products_context_list.append(prod_str)
        
    products_context = "\n".join(products_context_list) + "\n===========================================================\n"
    
    # Step 4: Kirim ke LLM untuk membuat jawaban rekomendasi final
    prompt = ChatPromptTemplate.from_template(PRODUCT_RAG_TEMPLATE)
    chain = prompt | llm
    
    res = chain.invoke({
        "query": paraphrased_query,
        "products_context": products_context
    })
    
    return res.content.strip()
