import streamlit as st
import sys
import os

# Menambahkan root directory project ke path agar Python mengenali modul 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import ShopAssistantAgent

# Konfigurasi tampilan halaman Streamlit
st.set_page_config(
    page_title="SmartShopper Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ SmartShopper AI Assistant")
st.caption("Asisten belanja interaktif bertenaga AI dengan LangChain, Groq, dan MongoDB Atlas Vector Search.")

# Inisialisasi instance ShopAssistantAgent ke dalam Streamlit Session State agar tidak ter-reset saat halaman refresh
if "agent" not in st.session_state:
    st.session_state.agent = ShopAssistantAgent()

# Tambahkan sidebar dengan informasi tambahan dan tombol reset
st.sidebar.title("Pengaturan Asisten")
st.sidebar.write("Gunakan tombol di bawah ini untuk menghapus seluruh riwayat percakapan Anda.")
if st.sidebar.button("Clear Chat History", type="primary"):
    st.session_state.agent.clear_history()
    st.sidebar.success("Riwayat percakapan berhasil dihapus!")
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    """
    ### Contoh Pertanyaan:
    * *Sapaan*: "Halo, selamat pagi!"
    * *Rekomendasi Produk*: "Saya cari dress dari bahan katun di bawah 40 dollar"
    * *Kebijakan Toko (FAQ)*: "Berapa lama waktu pengiriman barang ke luar Jawa?"
    * *Kontak CS*: "Bisa minta nomor WhatsApp CS?"
    """
)

# Menampilkan seluruh riwayat pesan yang ada di memory Agen
for message in st.session_state.agent.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Menerima input chat baru dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang produk fesyen atau kebijakan toko kami..."):
    # Tampilkan pesan user ke halaman chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan animasi loading selagi Agen memproses data & memanggil tool
    with st.spinner("Asisten sedang mengetik..."):
        try:
            response = st.session_state.agent.ask(prompt)
            # Tampilkan respon asisten ke halaman chat
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses permintaan Anda: {str(e)}")
