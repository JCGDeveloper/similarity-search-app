"""
📱 Similarity Search App — Buscador de Productos Similares

Arranca con: streamlit run similarity_app.py
"""

import base64
import hashlib
import io
import json
import os
import pickle
import re
import tempfile
import urllib.parse
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "product_data.pkl"
st.set_page_config(
    page_title="Buscador Similar — ¡Encuentra lo que buscas!",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Cargar datos y modelo (con caché) ───────────────────────────────────────

@st.cache_resource
def load_model():
    """Carga el modelo de embeddings (all-MiniLM-L6-v2, ligero ~80MB)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    """Carga productos y embeddings precomputados."""
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['products'], data['embeddings']


# ─── Búsqueda por similitud ───────────────────────────────────────────────────

def search_similar(query_embedding, all_embeddings, products, top_k=10):
    """Encuentra los top_k productos más similares por coseno."""
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalizar
    q_norm = query_embedding / np.linalg.norm(query_embedding)
    e_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    # Similitud coseno
    scores = np.dot(e_norm, q_norm.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            **products[idx],
            'score': float(scores[idx]),
        })
    return results


def get_text_embedding(text, model):
    """Genera embedding para un texto de búsqueda."""
    return model.encode(text)


def get_image_embedding(uploaded_file, model):
    """
    Genera embedding de imagen usando sentence-transformers.
    Usamos un modelo CLIP para poder comparar imágenes y texto en el mismo espacio.
    """
    try:
        from sentence_transformers.clip import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        image = Image.open(uploaded_file).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        embedding = clip_model.get_image_features(**inputs)
        return embedding.detach().numpy().flatten()
    except Exception as e:
        st.error(f"Error al procesar imagen con CLIP: {e}")
        st.info("⚠️ Usando descripción textual de la imagen como fallback.")
        return get_text_embedding("imagen de producto genérico", model)


def extract_from_url(url):
    """Intenta extraer información de un enlace de producto."""
    info = {"title": "", "description": ""}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)

        # Intentar extraer título y meta description
        import re as regex
        # Título
        match = regex.search(r'<title>(.*?)</title>', resp.text, regex.IGNORECASE | regex.DOTALL)
        if match:
            info["title"] = match.group(1).strip()

        # Meta description
        match = regex.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
            resp.text, regex.IGNORECASE
        )
        if not match:
            match = regex.search(
                r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']description["\']',
                resp.text, regex.IGNORECASE
            )
        if match:
            info["description"] = match.group(1).strip()

        # Si no hay descripción, coger texto visible del body
        if not info["description"]:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            info["description"] = text[:500]

        return info
    except Exception as e:
        return info


# ─── UI Helpers ───────────────────────────────────────────────────────────────

def product_card(product, cols):
    """Renderiza una tarjeta de producto en una columna de Streamlit."""
    with cols:
        # Placeholder visual con inicial del producto
        initial = product['name'][0].upper() if product['name'] else '?'
        bg_color = hash_color(product['name'])

        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border: 1px solid #e8e8e8;
            height: 100%;
            display: flex;
            flex-direction: column;
        ">
            <div style="
                width: 100%;
                height: 120px;
                background: {bg_color};
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 48px;
                font-weight: bold;
                color: white;
                margin-bottom: 12px;
            ">{initial}</div>
            <div style="flex-grow: 1;">
                <div style="font-size: 12px; color: #666; margin-bottom: 4px;">
                    {product['brand']} · {product['category']}
                </div>
                <div style="font-weight: 600; font-size: 15px; margin-bottom: 6px; line-height: 1.3;">
                    {product['name']}
                </div>
                <div style="font-size: 13px; color: #444; line-height: 1.4; margin-bottom: 8px;">
                    {product['description'][:120]}{'...' if len(product['description']) > 120 else ''}
                </div>
            </div>
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 8px;
                border-top: 1px solid #eee;
            ">
                <span style="font-size: 18px; font-weight: 700; color: #1a73e8;">
                    {product['price']}
                </span>
                <span style="font-size: 12px; color: #888;">
                    {(product['score'] * 100):.0f}% coincidencia
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def hash_color(text):
    """Genera un color pastel consistente basado en el texto."""
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    r = (h & 0xFF) % 200 + 55
    g = ((h >> 8) & 0xFF) % 200 + 55
    b = ((h >> 16) & 0xFF) % 200 + 55
    return f"rgb({r},{g},{b})"


# ─── CSS personalizado ────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Fondo general */
    .stApp {
        background: #f5f7fa;
    }

    /* Header con gradiente */
    .app-header {
        background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 100%);
        color: white;
        padding: 32px 24px;
        border-radius: 20px;
        margin-bottom: 28px;
        text-align: center;
    }
    .app-header h1 {
        font-size: 2.2rem;
        margin: 0 0 6px 0;
        color: white;
    }
    .app-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        color: white;
    }

    /* Tabs bonitos */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Inputs */
    .stTextInput > div > div > input {
        border-radius: 30px;
        padding: 12px 20px;
        font-size: 16px;
        border: 2px solid #ddd;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 3px rgba(26,115,232,0.12);
    }

    /* Botones */
    .stButton > button {
        border-radius: 30px;
        padding: 8px 28px;
        font-weight: 600;
        background: linear-gradient(135deg, #1a73e8, #6c5ce7);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        opacity: 0.92;
        box-shadow: 0 4px 14px rgba(26,115,232,0.3);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1a73e8;
        border-radius: 16px;
        padding: 20px;
        background: rgba(26,115,232,0.03);
    }

    /* Métricas */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Resultados stats */
    .results-stats {
        background: white;
        border-radius: 14px;
        padding: 16px 20px;
        margin: 16px 0 24px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        font-size: 15px;
        color: #444;
    }

    /* Tooltip en input de URL */
    .url-hint {
        font-size: 13px;
        color: #888;
        margin-top: 4px;
    }

    /* Sin resultados */
    .no-results {
        text-align: center;
        padding: 60px 20px;
        color: #888;
    }
    .no-results h3 {
        color: #555;
        margin-bottom: 8px;
    }
</style>
"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Buscador de Productos Similares</h1>
        <p>Encuentra productos parecidos a lo que buscas — por texto, enlace o imagen</p>
    </div>
    """, unsafe_allow_html=True)

    # Cargar datos
    with st.spinner("📦 Cargando modelo y datos..."):
        model = load_model()
        products, all_embeddings = load_data()

    # ─── Pestañas de búsqueda ─────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📝 Buscar por texto",
        "🔗 Buscar por enlace",
        "📸 Buscar por imagen",
    ])

    results = None
    search_query = ""

    # ─── TAB 1: TEXTO ─────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([4, 1])
        with col1:
            text_query = st.text_input(
                "",
                placeholder="Ej: patatas fritas, novela de misterio, auriculares bluetooth...",
                label_visibility="collapsed",
                key="text_search"
            )
        with col2:
            search_text = st.button("🔍 Buscar", type="primary", use_container_width=True)

        # Sugerencias rápidas
        st.markdown(
            '<div style="margin: 8px 0 16px 0;">'
            '<span style="font-size: 13px; color: #888;">Sugerencias: </span>'
            '</div>',
            unsafe_allow_html=True
        )

        suggestions = [
            "🥔 Patatas fritas",
            "📖 Novela de aventuras",
            "🎧 Cascos música",
            "☕ Cafetera eléctrica",
            "👟 Zapatillas deportivas",
            "🎸 Instrumento musical",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True,
                             help=f"Buscar: {sug}"):
                    text_query = sug.split(" ", 1)[1] if " " in sug else sug
                    search_text = True

        if search_text and text_query:
            search_query = text_query
            with st.spinner("🔎 Buscando productos similares..."):
                query_emb = get_text_embedding(text_query, model)
                results = search_similar(query_emb, all_embeddings, products)

    # ─── TAB 2: ENLACE ────────────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([4, 1])
        with col1:
            url_query = st.text_input(
                "",
                placeholder="Ej: https://www.amazon.es/dp/B08...",
                label_visibility="collapsed",
                key="url_search"
            )
        with col2:
            search_url = st.button("🔍 Analizar", type="primary", use_container_width=True)

        st.markdown(
            '<div class="url-hint">🔗 Pega un enlace de producto (Amazon, El Corte Inglés, etc.) '
            'y buscaré productos similares.</div>',
            unsafe_allow_html=True
        )

        if search_url and url_query:
            search_query = url_query
            with st.spinner("🌐 Analizando el enlace..."):
                info = extract_from_url(url_query)
                combined = f"{info['title']}. {info['description']}"
                if combined.strip(".").strip():
                    query_emb = get_text_embedding(combined, model)
                    results = search_similar(query_emb, all_embeddings, products)
                else:
                    st.warning("No se pudo extraer información del enlace. Prueba con otro.")

    # ─── TAB 3: IMAGEN ────────────────────────────────────────────────────────────
    with tab3:
        uploaded_file = st.file_uploader(
            "Sube una foto del producto que te gusta",
            type=["jpg", "jpeg", "png", "webp"],
            help="Se analizará la imagen y se buscarán productos visualmente similares."
        )

        if uploaded_file is not None:
            # Mostrar la imagen subida
            col_img, col_info = st.columns([1, 2])
            with col_img:
                image = Image.open(uploaded_file)
                st.image(image, caption="Tu imagen", use_container_width=True)
            with col_info:
                st.markdown("### 📸 Imagen recibida")
                st.write(f"Archivo: `{uploaded_file.name}`")
                st.write(f"Tamaño: {image.size[0]} × {image.size[1]} px")

                search_img = st.button("🔍 Buscar similares", type="primary",
                                       use_container_width=True)

                if search_img:
                    search_query = uploaded_file.name
                    with st.spinner("🖼️ Analizando imagen y buscando similares..."):
                        query_emb = get_image_embedding(uploaded_file, model)
                        results = search_similar(query_emb, all_embeddings, products)

    # ─── MOSTRAR RESULTADOS ───────────────────────────────────────────────────────
    if results is not None:
        st.markdown("---")

        if len(results) == 0:
            st.markdown("""
            <div class="no-results">
                <h3>😕 No se encontraron resultados</h3>
                <p>Prueba con otra búsqueda o un término más general.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Stats
            cat_counts = {}
            for r in results:
                cat_counts[r['category']] = cat_counts.get(r['category'], 0) + 1

            cats_str = " · ".join([f"**{k}** ({v})" for k, v in
                                   sorted(cat_counts.items(), key=lambda x: -x[1])])

            st.markdown(f"""
            <div class="results-stats">
                🎯 <b>{len(results)} resultados</b> para "{search_query[:60]}"
                {('...' if len(search_query) > 60 else '')}
                <br>
                📂 Categorías: {cats_str}
            </div>
            """, unsafe_allow_html=True)

            # Mostrar en grid de 4 columnas
            cols_per_row = 4
            for i in range(0, len(results), cols_per_row):
                row_products = results[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for j, product in enumerate(row_products):
                    product_card(product, cols[j])

    # ─── Footer ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 32px 0 16px 0; color: #999; font-size: 13px;">
        🛠️ Similarity Search App · Hecho por José Luis
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
