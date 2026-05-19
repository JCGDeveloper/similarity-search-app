"""
📱 SimilariFinder — Versión Móvil

App tipo tienda/catálogo con productos relacionados.
Diseñada para verse bien en móvil.
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

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "product_data.pkl"

st.set_page_config(page_title="SimilariFinder", page_icon="🔍", layout="centered")

# ─── DATA ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['products'], data['embeddings']

def get_product_by_id(pid, products):
    for p in products:
        if p["id"] == pid:
            return p
    return None

def search_similar(query_emb, all_embeddings, products, top_k=24):
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    q_norm = query_emb / np.linalg.norm(query_emb)
    e_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    scores = np.dot(e_norm, q_norm.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**products[i], 'score': float(scores[i])} for i in top_indices]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def hash_color(text):
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    r = (h & 0xFF) % 180 + 75
    g = ((h >> 8) & 0xFF) % 180 + 75
    b = ((h >> 16) & 0xFF) % 180 + 75
    return f"rgb({r},{g},{b})"

STOCK_EMOJI = {"En stock": "✅", "Pocas unidades": "⚠️", "Agotado": "❌"}

# ─── CSS MOBILE-FIRST ────────────────────────────────────────────────────────

CSS = """
<style>
    /* Reset */
    * { box-sizing: border-box; }
    .stApp { background: #f5f5f5; padding: 0 !important; }
    .block-container { padding: 8px 12px !important; max-width: 480px !important; }

    /* Barra superior */
    .bar {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        color: white; padding: 14px 16px; border-radius: 14px;
        margin-bottom: 12px;
    }
    .bar h1 { font-size: 18px; margin: 0; color: white; }
    .bar p { font-size: 12px; margin: 2px 0 0; opacity: .7; color: white; }
    .bar-sm { padding: 10px 14px; }
    .bar-sm h1 { font-size: 15px; }

    /* Buscador */
    .stTextInput>div>div>input { border-radius: 30px !important; padding: 10px 16px !important; font-size: 14px !important; border: 1px solid #ddd !important; }
    .stButton>button { border-radius: 30px !important; font-size: 13px !important; padding: 6px 16px !important; }

    /* Breadcrumb */
    .bc { font-size: 12px; color: #888; margin-bottom: 8px; }
    .bc a { color: #1a73e8; text-decoration: none; }

    /* Grilla de categorías */
    .cg { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
    .c {
        background: white; border-radius: 14px; padding: 12px;
        text-align: center; flex: 1 0 calc(33% - 8px); min-width: 90px;
        box-shadow: 0 1px 4px rgba(0,0,0,.05); cursor: pointer;
    }
    .c .ci { font-size: 24px; }
    .c .cn { font-size: 12px; font-weight: 600; margin-top: 4px; color: #333; }
    .c .cc { font-size: 10px; color: #999; }

    /* Producto card (horizontal en home) */
    .pc {
        background: white; border-radius: 12px; padding: 10px;
        margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.04);
        display: flex; gap: 10px; align-items: center; cursor: pointer;
    }
    .pc .pb {
        width: 48px; height: 48px; border-radius: 12px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 22px;
    }
    .pc .pi { flex: 1; min-width: 0; }
    .pc .pbrand { font-size: 10px; color: #999; text-transform: uppercase; }
    .pc .pname { font-size: 13px; font-weight: 600; color: #222; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .pc .pprice { font-size: 15px; font-weight: 700; color: #0f3460; }

    /* Producto card (relacionados, aún más pequeño) */
    .prc {
        background: white; border-radius: 10px; padding: 8px;
        margin-bottom: 6px; display: flex; gap: 8px; align-items: center; cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,.03);
    }
    .prc .prb { width: 36px; height: 36px; border-radius: 8px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 16px; }
    .prc .prname { font-size: 12px; font-weight: 600; flex: 1; color: #222; }
    .prc .prprice { font-size: 13px; font-weight: 700; color: #0f3460; }

    /* Detalle producto */
    .dt {
        background: white; border-radius: 16px; padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,.04); margin-bottom: 16px;
    }
    .dt .dticon {
        width: 72px; height: 72px; border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 36px; margin-bottom: 12px;
    }
    .dt .dtbrand { font-size: 11px; color: #999; text-transform: uppercase; }
    .dt .dtname { font-size: 18px; font-weight: 700; color: #222; margin: 2px 0 4px; }
    .dt .dtprice { font-size: 24px; font-weight: 800; color: #0f3460; }
    .dt .dtrating { font-size: 12px; color: #888; margin-bottom: 6px; }
    .dt .dtdesc { font-size: 13px; color: #555; line-height: 1.5; margin: 8px 0; }
    .dt .dtfeat { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
    .dt .dtfeat span { background: #e8f0fe; color: #1a73e8; font-size: 11px; padding: 3px 10px; border-radius: 12px; }
    .dt .dttags { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
    .dt .dttags span { background: #f0f0f0; color: #666; font-size: 10px; padding: 2px 8px; border-radius: 10px; }
    .dt .dtstock { font-size: 13px; margin-top: 4px; }

    /* Sección relacionados */
    .rs h2 { font-size: 16px; font-weight: 700; margin: 16px 0 8px; color: #222; }
    .rs p { font-size: 12px; color: #999; margin: 0 0 8px; }

    /* Sugerencias rápidas */
    .sg { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px; }
    .sg .sgbtn {
        background: white; border: 1px solid #e0e0e0; border-radius: 20px;
        padding: 5px 12px; font-size: 12px; cursor: pointer; color: #555;
    }
</style>
"""

def show_product_card(p, key_prefix=""):
    """Render a horizontal product card."""
    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    k = f"{key_prefix}{p['id']}"

    st.markdown(f"""
    <div class="pc" onclick="document.getElementById('{k}').click()">
        <div class="pb" style="background:{color};">{icon}</div>
        <div class="pi">
            <div class="pbrand">{p['brand']} · {p['category']}</div>
            <div class="pname">{p['name'][:60]}</div>
        </div>
        <div class="pprice">{p['price']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Hidden button to handle click via session state
    if st.button(f"Ver {p['name'][:30]}", key=k, use_container_width=True):
        st.session_state["selected_product"] = p["id"]
        st.session_state["page"] = "product"
        st.rerun()

def show_related_card(p, key_prefix=""):
    """Smaller card for related products section."""
    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    k = f"{key_prefix}rel_{p['id']}"

    st.markdown(f"""
    <div class="prc" onclick="document.getElementById('{k}').click()">
        <div class="prb" style="background:{color};">{icon}</div>
        <div class="prname">{p['name'][:50]}</div>
        <div class="prprice">{p['price']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"Rel {p['name'][:25]}", key=k, use_container_width=True):
        st.session_state["selected_product"] = p["id"]
        st.session_state["page"] = "product"
        st.rerun()

# ─── PAGES ───────────────────────────────────────────────────────────────────

def page_home(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)

    # Barra
    st.markdown(f'<div class="bar"><h1>🔍 SimilariFinder</h1><p>{len(products)} productos · Encuentra relacionados</p></div>', unsafe_allow_html=True)

    # Buscador
    query = st.text_input("", placeholder="🔎 Busca productos, marcas...", label_visibility="collapsed")
    if query:
        q_emb = model.encode(query)
        results = search_similar(q_emb, embeddings, products)
        st.session_state["results"] = results
        st.session_state["q"] = query
        st.session_state["page"] = "search"
        st.rerun()

    # Sugerencias
    sugs = ["🥔 Patatas", "📖 Libros", "🎧 Auriculares", "☕ Café", "👟 Zapatillas", "🎸 Guitarra"]
    cols = st.columns(3)
    for i, s in enumerate(sugs):
        with cols[i % 3]:
            if st.button(s, key=f"s{i}", use_container_width=True):
                q_emb = model.encode(s.split(" ", 1)[1])
                results = search_similar(q_emb, embeddings, products)
                st.session_state["results"] = results
                st.session_state["q"] = s
                st.session_state["page"] = "search"
                st.rerun()

    st.markdown('<div class="cg">', unsafe_allow_html=True)

    # Categorías
    cats = list(dict.fromkeys(p["category"] for p in products))
    icons = {}
    for p in products:
        if p["category"] not in icons:
            icons[p["category"]] = p.get("icon", "📦")
    counts = Counter(p["category"] for p in products)

    for cat in cats:
        icon = icons.get(cat, "📦")
        count = counts[cat]
        st.markdown(f'<div class="c"><div class="ci">{icon}</div><div class="cn">{cat}</div><div class="cc">{count}</div></div>', unsafe_allow_html=True)
        if st.button(f"cat_{cat}", key=f"cat_{cat}"):
            st.session_state["cat"] = cat
            st.session_state["page"] = "category"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Destacados
    st.markdown("### ✨ Destacados")
    featured = random.sample(products, min(10, len(products)))
    for p in featured:
        show_product_card(p, "fp_")

def page_category(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    cat = st.session_state.get("cat", products[0]["category"])
    cat_prods = [p for p in products if p["category"] == cat]
    icon = cat_prods[0].get("icon", "📦") if cat_prods else "📦"

    if st.button("← Volver", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown(f'<div class="bar bar-sm"><h1>{icon} {cat}</h1><p>{len(cat_prods)} productos</p></div>', unsafe_allow_html=True)

    query = st.text_input("", placeholder=f"🔎 Buscar en {cat}...", label_visibility="collapsed")
    if query:
        q_emb = model.encode(query)
        results = search_similar(q_emb, embeddings, cat_prods)
        st.session_state["results"] = results
        st.session_state["q"] = query
        st.session_state["page"] = "search"
        st.rerun()

    for p in cat_prods:
        show_product_card(p, "cp_")

def page_product(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    pid = st.session_state.get("selected_product", products[0]["id"])
    p = get_product_by_id(pid, products)
    if not p:
        st.session_state["page"] = "home"
        st.rerun()
        return

    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    stars = "⭐" * int(p.get("rating", 0) // 1) + "☆" * (5 - int(p.get("rating", 0) // 1))
    stock_emoji = STOCK_EMOJI.get(p["stock"], "")

    if st.button("← Volver", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown(f'<div class="bc"><a onclick="history.back()">Inicio</a> / <b>{p["name"][:40]}</b></div>', unsafe_allow_html=True)

    # Detalle del producto
    feat_html = ""
    if p.get("features"):
        feat_html = '<div class="dtfeat">' + "".join(f"<span>✓ {f}</span>" for f in p["features"]) + "</div>"

    tags_html = ""
    if p.get("tags"):
        tags_html = '<div class="dttags">' + "".join(f"<span>#{t}</span>" for t in p["tags"][:8]) + "</div>"

    variants_html = ""
    if p.get("variants"):
        variants_html = f'<div style="font-size:12px;color:#888;margin-top:8px;">Formatos: {", ".join(p["variants"])}</div>'

    st.markdown(f"""
    <div class="dt">
        <div class="dticon" style="background:{color};">{icon}</div>
        <div class="dtbrand">{p['brand']} · {p['subcategory']}</div>
        <div class="dtname">{p['name']}</div>
        <div class="dtrating">{stars} {p['rating']} ({p['reviews']:,})</div>
        <div class="dtprice">{p['price']}</div>
        <div class="dtstock">{stock_emoji} {p['stock']}</div>
        <div class="dtdesc">{p['description']}</div>
        {feat_html}
        {tags_html}
        {variants_html}
    </div>
    """, unsafe_allow_html=True)

    # Productos relacionados
    st.markdown('<div class="rs">', unsafe_allow_html=True)
    st.markdown(f"## 🔗 Relacionados")

    related = []
    for rid in p.get("related_ids", []):
        rp = get_product_by_id(rid, products)
        if rp and rp["id"] != p["id"]:
            related.append(rp)

    if len(related) < 8:
        p_idx = products.index(p)
        q_emb = embeddings[p_idx]
        semantic = search_similar(q_emb, embeddings, products, top_k=12)
        for r in semantic:
            if r["id"] != p["id"] and r not in related:
                related.append(r)
                if len(related) >= 12:
                    break

    related = [r for r in related if r["id"] != p["id"]][:12]

    if related:
        st.markdown(f"<p>{len(related)} productos similares</p>", unsafe_allow_html=True)
        for r in related:
            show_related_card(r, f"rp_{p['id']}_")
    else:
        st.markdown("<p style='color:#888;'>No se encontraron productos relacionados.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def page_search(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    results = st.session_state.get("results", [])
    q = st.session_state.get("q", "")

    if st.button("← Volver", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown(f'<div class="bar bar-sm"><h1>🔍 "{q[:50]}"</h1><p>{len(results)} resultados</p></div>', unsafe_allow_html=True)

    if not results:
        st.markdown("<p style='text-align:center;color:#888;padding:40px 0;'>😕 No se encontraron resultados</p>", unsafe_allow_html=True)
        return

    for r in results:
        show_product_card(r, "sr_")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    with st.spinner("Cargando..."):
        model = load_model()
        products, embeddings = load_data()

    page = st.session_state["page"]
    if page == "home":
        page_home(products, model, embeddings)
    elif page == "category":
        page_category(products, model, embeddings)
    elif page == "product":
        page_product(products, model, embeddings)
    elif page == "search":
        page_search(products, model, embeddings)

if __name__ == "__main__":
    main()
