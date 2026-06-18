import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.database import get_materials, get_categories

# Inisialisasi model LLM Groq menggunakan LangChain.
# Menggunakan model GROQ_MODEL dengan temperature 0.0 agar output formatnya konsisten dan terprediksi.
llm = ChatGroq(
    model_name=GROQ_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.0
)

def paraphrase_query(query: str, chat_history_str: str) -> str:
    """
    Menulis ulang query pembeli agar memasukkan konteks history chat sebelumnya.
    Contoh:
      History: User mencari baju katun.
      Query: 'Berapa harganya?'
      Hasil paraphrase: 'Berapa harga baju katun tersebut?'
    """
    if not chat_history_str.strip():
        return query
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise query paraphraser. Your job is to rewrite the latest user query to include necessary context from the chat history (like the product category, material, or price being discussed) so it can be searched as a standalone query. "
            "Follow these strict rules:\n"
            "1. Output ONLY the paraphrased query. Absolutely no introduction, explanation, conversational text, or preamble.\n"
            "2. If the query is already self-contained and doesn't need context from the history, output the original query exactly.\n"
            "3. Do not say things like 'Here is the paraphrased query' or 'Based on the history...'. Just output the search query text.\n\n"
            "Examples:\n"
            "History:\n"
            "User: Saya mencari celana jeans.\n"
            "Assistant: Berikut beberapa celana jeans...\n"
            "Query: Berapa harganya?\n"
            "Output: Berapa harga celana jeans tersebut?\n\n"
            "History:\n"
            "User: Halo selamat pagi\n"
            "Assistant: Halo! Ada yang bisa dibantu?\n"
            "Query: rekomendasi jaket di bawah 40 dolar\n"
            "Output: rekomendasi jaket di bawah 40 dolar"
        )),
        ("user", "History:\n{history}\n\nQuery: {query}\nOutput:")
    ])
    
    # LangChain Expression Language (LCEL) chain
    chain = prompt | llm
    res = chain.invoke({"query": query, "history": chat_history_str})
    return res.content.strip()

# Template prompt untuk mem-parsing criteria belanja menjadi format JSON terstandar.
# Menggunakan double kurung kurawal {{ }} agar tidak dibaca sebagai variabel python format().
METADATA_FILTER_TEMPLATE = """
You are a json generator that have a job to generate json based on the input.
The return json should be in the format:
```json
{{
    "operator": "AND",
    "conditions":[
        {{"field": "meta.category", "operator":"==", "value": <category>}},
        {{"field": "meta.material", "operator":"==", "value": <material>}},
        {{"field": "meta.gender", "operator":"==", "value" : <male|female|unisex>}},
        {{"field": "meta.price", "operator":<"<="|">="|"==">, "value": <price>}}
    ]
}}
```
The json key above can be omitted if the value is not provided in the input, so please make sure to only return the keys that are provided in the input.

For the material and category, you can only use the material and category that are provided below:
Materials: [ {materials} ]
Categories: [ {categories} ]

if the input does not contain any of the keys above, you should return an empty json object like this:
```json
{{}}
```
Sometimes the material and category can be negated, so you should also handle that by using the operator "!=" for material and category. 

Sometimes the material and category is not explicitly mentioned, you should analyze which material and category is the most suitable based on the input, and return the json with the material and category that you think is the most suitable.

Nested conditions are allowed, for nested conditions, you can use "OR" and "AND" as the operator, and the conditions should be in the "conditions" array.

if user said the price around some value, please find the price between those value -10 and value +10.

The example of the result are expected to be like this:

1. Input: "can you give me a dress with cotton material?"
output:
```json
{{
    "operator": "AND",
    "conditions": [
        {{"field": "meta.material", "operator": "==", "value": "Cotton"}},
        {{"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"}}
    ]
}}
```

2. Input: "Give me Shirt that is not made of cotton and has a price less than $100"
output:
```json
{{
    "operator": "AND",
    "conditions": [
        {{"field": "meta.category", "operator": "==", "value": "Tops"}},
        {{"field": "meta.material", "operator": "!=", "value": "Cotton"}},
        {{"field": "meta.price", "operator": "<=", "value": 100}}
    ]
}}
```

3. Input: "I want a dress that is not hot and has a price greater than $50"
output:
```json
{{
    "operator": "AND",
    "conditions": [
        {{"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"}},
        {{"field": "meta.price", "operator": ">=", "value": 50}},
        {{
            "operator": "OR",
            "conditions": [
                {{"field": "meta.material", "operator": "==", "value": "Cotton"}},
                {{"field": "meta.material", "operator": "==", "value": "Polyester"}}
            ]
        }}
    ]
}}
```

4. Input: i want tops that have price between $20 and $50
output:
```json
{{
    "operator": "AND",
    "conditions": [
        {{"field": "meta.category", "operator": "==", "value": "Tops"}},
        {{
            "operator": "AND",
            "conditions":[
                {{"field": "meta.price", "operator": ">=", "value": 20}},
                {{"field": "meta.price", "operator": "<=", "value": 50}}
            ]
        }}
    ]
}}
```

5. Input: I want the dress price around $50
output: 
```json
{{
    "operator": "AND",
    "conditions": [
        {{"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"}},
        {{
            "operator": "AND",
            "conditions":[
                {{"field": "meta.price", "operator": ">=", "value": 40}},
                {{"field": "meta.price", "operator": "<=", "value": 60}}
            ]
        }}
    ]
}}
```

6. Input: {query}
output:
"""

def generate_metadata_filter(query: str) -> dict:
    """
    Memproses query user menggunakan LLM untuk mengekstraksi kriteria filter pencarian.
    Hasilnya berupa kamus Python yang dapat dikonversi menjadi query filter MongoDB Atlas.
    """
    materials = get_materials()
    categories = get_categories()
    
    materials_str = ", ".join(materials)
    categories_str = ", ".join(categories)
    
    formatted_prompt = METADATA_FILTER_TEMPLATE.format(
        materials=materials_str,
        categories=categories_str,
        query=query
    )
    
    res = llm.invoke(formatted_prompt)
    output_text = res.content.strip()
    
    json_filter = {}
    try:
        # Mencari blok JSON di dalam teks output LLM secara robust
        json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            json_filter = json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON filter: {e}. Output text: {output_text}")
        json_filter = {}
        
    return json_filter
