"""
📱 Catálogo de Productos Similares — App estilo tienda

Arranca con: streamlit run catalog_app.py
"""

import hashlib
import os
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "product_data.pkl"

st.set_page_config(
    page_title="SimilariFinder — Encuentra productos relacionados",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CARGAR DATOS ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['products'], data['embeddings']

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def hash_color(text):
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    r = (h & 0xFF) % 180 + 75
    g = ((h >> 8) & 0xFF) % 180 + 75
    b = ((h >> 16) & 0xFF) % 180 + 75
    return f"rgb({r},{g},{b})"

def get_product_by_id(pid, products):
    for p in products:
        if p["id"] == pid:
            return p
    return None

def search_similar(query_emb, all_embeddings, products, top_k=20):
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    q_norm = query_emb / np.linalg.norm(query_emb)
    e_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    scores = np.dot(e_norm, q_norm.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({**products[idx], 'score': float(scores[idx])})
    return results

# ─── CSS ─────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .stApp { background: #f4f6f9; }

    /* Top Bar */
    .top-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 16px 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
    }
    .top-bar h1 { margin: 0; font-size: 1.4rem; color: white; }
    .top-bar span { font-size: 0.85rem; opacity: 0.8; }

    /* Category Cards */
    .cat-card {
        background: white;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    .cat-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
    .cat-card .icon { font-size: 36px; margin-bottom: 8px; }
    .cat-card .title { font-size: 14px; font-weight: 600; color: #333; }
    .cat-card .count { font-size: 12px; color: #999; }

    /* Product Card */
    .prod-card {
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: all 0.2s;
        height: 100%;
        cursor: pointer;
    }
    .prod-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
    .prod-card .color-bar { height: 80px; display: flex; align-items: center; justify-content: center; font-size: 40px; }
    .prod-card .info { padding: 14px; }
    .prod-card .brand { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    .prod-card .name { font-size: 14px; font-weight: 600; color: #222; margin: 4px 0; line-height: 1.3; min-height: 36px; }
    .prod-card .price { font-size: 17px; font-weight: 700; color: #0f3460; }
    .prod-card .rating { font-size: 12px; color: #888; }
    .prod-card .badge {
        display: inline-block; font-size: 10px; padding: 2px 8px;
        border-radius: 20px; background: #e8f0fe; color: #1a73e8; margin-top: 6px;
    }

    /* Product Detail */
    .detail-header {
        background: white;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .detail-header .icon-big { font-size: 64px; margin-bottom: 12px; }
    .detail-features { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    .detail-features span {
        background: #e8f0fe; color: #1a73e8; font-size: 12px;
        padding: 4px 12px; border-radius: 20px;
    }
    .detail-tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .detail-tags span {
        background: #f0f0f0; color: #666; font-size: 11px;
        padding: 3px 10px; border-radius: 12px;
    }

    /* Related Section */
    .related-section {
        margin-top: 24px;
    }
    .related-section h2 {
        font-size: 1.3rem; font-weight: 600; margin-bottom: 16px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 30px; font-weight: 500;
    }
    .back-btn > button {
        background: transparent; border: 1px solid #ddd; color: #555;
    }
    .back-btn > button:hover { border-color: #1a73e8; color: #1a73e8; }

    /* Search */
    .stTextInput > div > div > input {
        border-radius: 30px; padding: 12px 20px; font-size: 16px;
        border: 2px solid #e0e0e0;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0f3460; box-shadow: 0 0 0 3px rgba(15,52,96,0.1);
    }

    /* Breadcrumb */
    .breadcrumb {
        font-size: 13px; color: #888; margin-bottom: 16px;
    }
    .breadcrumb a { color: #1a73e8; cursor: pointer; text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }

    /* Badge stock */
    .stock-badge {
        display: inline-block; font-size: 12px; font-weight: 500;
        padding: 3px 10px; border-radius: 20px;
    }
    .stock-badge.in { background: #e6f4ea; color: #1e7e34; }
    .stock-badge.low { background: #fef7e0; color: #e37400; }
    .stock-badge.out { background: #fce8e6; color: #c5221f; }

    /* Related card */
    .rel-card {
        background: white;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        transition: all 0.2s;
        height: 100%;
        cursor: pointer;
    }
    .rel-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
    .rel-card .color-bar { height: 60px; display: flex; align-items: center; justify-content: center; font-size: 28px; }
    .rel-card .info { padding: 10px 12px; }
    .rel-card .brand { font-size: 10px; color: #888; text-transform: uppercase; }
    .rel-card .name { font-size: 12px; font-weight: 600; margin: 2px 0; line-height: 1.2; }
    .rel-card .price { font-size: 14px; font-weight: 700; color: #0f3460; }
</style>
"""

# ─── UI COMPONENTS ───────────────────────────────────────────────────────────

def product_card_html(p, size="small"):
    """Generate HTML for a product card."""
    icon = p.get("icon", "📦")
    color = hash_color(p["name"])
    rating_stars = "⭐" * int(p.get("rating", 0) // 1) + "☆" * (5 - int(p.get("rating", 0) // 1))
    stock_class = {"En stock": "in", "Pocas unidades": "low", "Agotado": "out"}.get(p["stock"], "in")

    badge = ""
    if "novedad" in p.get("tags", []):
        badge = '<span class="badge">✨ Novedad</span>'
    elif "oferta" in p.get("tags", []):
        badge = '<span class="badge">🔥 Oferta</span>'

    if size == "large":
        return f"""
        <div class="prod-card" onclick="window.location.href='?p={p['id']}'">
            <div class="color-bar" style="background:{color};">{icon}</div>
            <div class="info">
                <div class="brand">{p['brand']} · {p['category']}</div>
                <div class="name">{p['name'][:70]}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="price">{p['price']}</span>
                    <span class="rating">{rating_stars} ({p.get('reviews',0)})</span>
                </div>
                <span class="stock-badge {stock_class}">{p['stock']}</span>
                {badge}
            </div>
        </div>
        """
    else:
        return f"""
        <div class="rel-card" onclick="window.location.href='?p={p['id']}'">
            <div class="color-bar" style="background:{color};">{icon}</div>
            <div class="info">
                <div class="brand">{p['brand']}</div>
                <div class="name">{p['name'][:60]}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="price">{p['price']}</span>
                    <span class="rating">{rating_stars[:3]}</span>
                </div>
            </div>
        </div>
        """


# ─── PAGES ───────────────────────────────────────────────────────────────────

def show_category_landing(products, model, all_embeddings):
    """Show main page with category grid and search."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Top bar
    st.markdown("""
    <div class="top-bar">
        <div>
            <h1>🔍 SimilariFinder</h1>
            <span>Encuentra productos relacionados — busca, navega, descubre</span>
        </div>
        <div style="font-size: 13px; opacity: 0.7;">
            {} productos · 10 categorías
        </div>
    </div>
    """.format(len(products)), unsafe_allow_html=True)

    # ─── Search ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", placeholder="🔎 Busca productos, marcas, categorías...",
                              label_visibility="collapsed", key="main_search")
    with col2:
        search_click = st.button("Buscar", type="primary", use_container_width=True)

    if search_click and query:
        with st.spinner("Buscando..."):
            q_emb = model.encode(query)
            results = search_similar(q_emb, all_embeddings, products, top_k=30)
            st.session_state["search_results"] = results
            st.session_state["search_query"] = query
            st.session_state["page"] = "search"
            st.rerun()

    # ─── Quick suggestions ──────────────────────────────────────────────────
    suggestions = ["🥔 Patatas fritas", "📖 Novela policíaca", "🎧 Auriculares",
                   "☕ Cafetera", "👟 Zapatillas", "🎸 Guitarra"]
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with cols[i]:
            if st.button(sug, key=f"qs_{i}", use_container_width=True, type="secondary"):
                q_emb = model.encode(sug.split(" ", 1)[1])
                results = search_similar(q_emb, all_embeddings, products, top_k=30)
                st.session_state["search_results"] = results
                st.session_state["search_query"] = sug
                st.session_state["page"] = "search"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Category grid ──────────────────────────────────────────────────────
    st.markdown("### 🏪 Explora por categorías")

    # Load categories from products
    from collections import Counter
    cat_counter = Counter(p["category"] for p in products)
    icons = {}
    for p in products:
        if p["category"] not in icons:
            icons[p["category"]] = p.get("icon", "📦")

    categories = sorted(cat_counter.keys())
    cols_per_row = 5
    for i in range(0, len(categories), cols_per_row):
        row_cats = categories[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, cat in enumerate(row_cats):
            with cols[j]:
                icon = icons.get(cat, "📦")
                count = cat_counter[cat]
                bg = hash_color(cat)
                st.markdown(f"""
                <div class="cat-card" style="border-color: {bg};">
                    <div class="icon">{icon}</div>
                    <div class="title">{cat}</div>
                    <div class="count">{count} productos</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Ver", key=f"cat_{cat}", use_container_width=True):
                    st.session_state["selected_category"] = cat
                    st.session_state["page"] = "category"
                    st.rerun()

    # ─── Featured products (random selection) ───────────────────────────────
    st.markdown("<br><h3>✨ Productos destacados</h3>", unsafe_allow_html=True)
    featured = random.sample(products, min(12, len(products)))

    cols = st.columns(4)
    for i, p in enumerate(featured):
        with cols[i % 4]:
            st.markdown(product_card_html(p, size="large"), unsafe_allow_html=True)
            if st.button(f"Ver {p['name'][:25]}...", key=f"fp_{p['id']}", use_container_width=True):
                st.session_state["selected_product"] = p["id"]
                st.session_state["page"] = "product"
                st.rerun()


def show_category(products, model, all_embeddings):
    """Show products filtered by category."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    cat = st.session_state.get("selected_category", products[0]["category"])
    cat_products = [p for p in products if p["category"] == cat]
    icon = cat_products[0].get("icon", "📦") if cat_products else "📦"

    # Back + breadcrumb
    st.markdown(f"""
    <div class="breadcrumb">
        <a onclick="history.back()">← Inicio</a> / <b>{icon} {cat}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="top-bar">
        <h1>{icon} {cat}</h1>
        <span>{len(cat_products)} productos</span>
    </div>
    """, unsafe_allow_html=True)

    # Search within category
    query = st.text_input("", placeholder=f"🔎 Buscar en {cat}...",
                          label_visibility="collapsed", key="cat_search")
    search_cat = st.button("Buscar", key="search_cat_btn", use_container_width=True)

    if search_cat and query:
        q_emb = model.encode(query)
        results = search_similar(q_emb, all_embeddings, cat_products, top_k=30)
        st.session_state["search_results"] = results
        st.session_state["search_query"] = query
        st.session_state["page"] = "search"
        st.rerun()

    # Grid of products
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, p in enumerate(cat_products):
        with cols[i % 4]:
            st.markdown(product_card_html(p, size="large"), unsafe_allow_html=True)
            if st.button(f"Ver {p['name'][:25]}...", key=f"catp_{p['id']}", use_container_width=True):
                st.session_state["selected_product"] = p["id"]
                st.session_state["page"] = "product"
                st.rerun()


def show_product_detail(products, model, all_embeddings):
    """Show full product detail with related products."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    pid = st.session_state.get("selected_product", products[0]["id"])
    p = get_product_by_id(pid, products)
    if not p:
        st.error("Producto no encontrado")
        st.session_state["page"] = "home"
        st.rerun()
        return

    icon = p.get("icon", "📦")
    color = hash_color(p["name"])
    rating_stars = "⭐" * int(p.get("rating", 0) // 1) + "☆" * (5 - int(p.get("rating", 0) // 1))
    stock_class = {"En stock": "in", "Pocas unidades": "low", "Agotado": "out"}.get(p["stock"], "in")

    # Breadcrumb + back
    st.markdown(f"""
    <div class="breadcrumb">
        <a onclick="history.back()">← Inicio</a> / <a onclick="history.back()">{p['category']}</a> / <b>{p['name'][:50]}</b>
    </div>
    """, unsafe_allow_html=True)

    back_cols = st.columns([1, 10])
    with back_cols[0]:
        if st.button("← Volver", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    # ─── PRODUCT DETAIL ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="detail-header">
        <table style="width:100%; border-collapse: collapse;">
        <tr>
            <td style="width:120px; text-align:center; vertical-align:top;">
                <div style="width:100px;height:100px;background:{color};border-radius:20px;
                    display:flex;align-items:center;justify-content:center;font-size:48px;
                    margin: 0 auto;">
                    {icon}
                </div>
            </td>
            <td style="vertical-align:top; padding-left:24px;">
                <div style="font-size:13px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">
                    {p['brand']} · {p['subcategory']}
                </div>
                <h1 style="font-size:1.5rem;margin:4px 0;color:#222;">{p['name']}</h1>
                <div style="font-size:12px;color:#888;margin-bottom:8px;">
                    {rating_stars} {p['rating']} ({p['reviews']:,} valoraciones)
                </div>
                <div style="font-size:28px;font-weight:700;color:#0f3460;">{p['price']}</div>
                <div style="margin:8px 0;">
                    <span class="stock-badge {stock_class}">{p['stock']}</span>
                </div>
                <p style="color:#555;line-height:1.5;margin:12px 0;">{p['description']}</p>
    """, unsafe_allow_html=True)

    # Features
    if p.get("features"):
        st.markdown('<div class="detail-features">', unsafe_allow_html=True)
        for f in p["features"]:
            st.markdown(f'<span>✓ {f}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tags
    st.markdown('<div class="detail-tags">', unsafe_allow_html=True)
    for t in p.get("tags", []):
        st.markdown(f'<span>#{t}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Variants
    if p.get("variants"):
        st.markdown(f"""
        <div style="margin-top:12px;font-size:13px;color:#888;">
            Formatos disponibles: {', '.join(p['variants'])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</td></tr></table></div>", unsafe_allow_html=True)

    # ─── RELATED PRODUCTS SECTION ──────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="related-section">', unsafe_allow_html=True)
    st.markdown(f"## 🔗 Productos relacionados con {p['name'][:40]}")

    # Collect related products
    related_products = []
    relation_labels = []

    # 1. Explicit related_ids (predefined by relationships)
    for rid in p.get("related_ids", []):
        rp = get_product_by_id(rid, products)
        if rp and rp["id"] != p["id"]:
            related_products.append(rp)

    # 2. Semantic similarity (via embeddings)
    if len(related_products) < 12:
        p_idx = products.index(p)
        q_emb = all_embeddings[p_idx]
        semantic = search_similar(q_emb, all_embeddings, products, top_k=15)
        for r in semantic:
            if r["id"] != p["id"] and r not in related_products:
                related_products.append(r)
                if len(related_products) >= 16:
                    break

    # Remove self, limit to 16
    related_products = [r for r in related_products if r["id"] != p["id"]][:16]

    # Display related products
    if related_products:
        st.markdown(f"<p style='color:#888;font-size:14px;'>{len(related_products)} productos relacionados</p>",
                    unsafe_allow_html=True)

        # Group into sections
        cols = st.columns(4)
        for i, rp in enumerate(related_products):
            with cols[i % 4]:
                st.markdown(product_card_html(rp, size="small"), unsafe_allow_html=True)
                btn_key = f"rel_{i}_{rp['id']}"
                if st.button(f"Ver", key=btn_key, use_container_width=True):
                    st.session_state["selected_product"] = rp["id"]
                    st.session_state["page"] = "product"
                    st.rerun()
    else:
        st.markdown("<p style='color:#888;'>No se encontraron productos relacionados.</p>",
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def show_search_results(products, model, all_embeddings):
    """Show search results page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    results = st.session_state.get("search_results", [])
    query = st.session_state.get("search_query", "")

    # Back
    st.markdown(f"""
    <div class="breadcrumb">
        <a onclick="history.back()">← Inicio</a> / Resultados: "<b>{query[:50]}</b>"
    </div>
    """, unsafe_allow_html=True)

    back_cols = st.columns([1, 10])
    with back_cols[0]:
        if st.button("← Volver", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    # Stats
    cat_counts = Counter(r["category"] for r in results)
    cats_str = " · ".join([f"**{k}** ({v})" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])[:5]])

    st.markdown(f"""
    <div class="top-bar" style="padding: 16px 20px;">
        <div>
            <h1 style="font-size:1.1rem;">🔍 Resultados para "{query[:60]}"</h1>
            <span style="font-size:12px;">{len(results)} productos encontrados</span>
        </div>
        <div style="font-size:12px;opacity:0.8;">{cats_str}</div>
    </div>
    """, unsafe_allow_html=True)

    if not results:
        st.markdown("<br><h3 style='text-align:center;color:#888;'>😕 No se encontraron resultados. Prueba con otra búsqueda.</h3>", unsafe_allow_html=True)
        return

    # Show results
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.markdown(product_card_html(r, size="large"), unsafe_allow_html=True)
            if st.button(f"Ver {r['name'][:25]}...", key=f"sr_{r['id']}", use_container_width=True):
                st.session_state["selected_product"] = r["id"]
                st.session_state["page"] = "product"
                st.rerun()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    # Load data
    with st.spinner("📦 Cargando catálogo..."):
        model = load_model()
        products, all_embeddings = load_data()

    # Route
    page = st.session_state["page"]

    if page == "home":
        show_category_landing(products, model, all_embeddings)
    elif page == "category":
        show_category(products, model, all_embeddings)
    elif page == "product":
        show_product_detail(products, model, all_embeddings)
    elif page == "search":
        show_search_results(products, model, all_embeddings)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:32px 0 16px;color:#aaa;font-size:13px;">
        🛠️ SimilariFinder · {cantidad} productos · Hecho por José Luis para Joaquín
    </div>
    """.format(cantidad=len(products)), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
