"""
📱 Catálogo — Buscador de Productos con Similares (Mobile-First)

App tipo tienda/catálogo con búsqueda semántica.
Arranca con: streamlit run catalog_app.py --server.port 8512
"""

import hashlib
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "product_data.pkl"

st.set_page_config(
    page_title="Catálogo",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# ─── DATA ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['products'], data['embeddings']


def get_product(pid, products):
    for p in products:
        if p["id"] == pid:
            return p
    return None


def search_similar(query_emb, all_emb, products, top_k=20):
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    q_norm = query_emb / np.linalg.norm(query_emb)
    e_norm = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)
    scores = np.dot(e_norm, q_norm.T).flatten()
    top = np.argsort(scores)[::-1][:top_k]
    return [{**products[i], 'score': float(scores[i])} for i in top]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def hash_color(text):
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    r = (h & 0xFF) % 180 + 75
    g = ((h >> 8) & 0xFF) % 180 + 75
    b = ((h >> 16) & 0xFF) % 180 + 75
    return f"rgb({r},{g},{b})"


STOCK_COLORS = {"En stock": "#27ae60", "Pocas unidades": "#f39c12", "Agotado": "#e74c3c"}
STOCK_LABELS = {"En stock": "✅ En stock", "Pocas unidades": "⚠️ Últimas uds.", "Agotado": "❌ Agotado"}
CAT_EMOJIS = {
    "Alimentación": "🍕", "Belleza": "💄", "Deportes": "⚽", "Electrónica": "🔌",
    "Hogar": "🏠", "Juguetes": "🎮", "Libros": "📚", "Mascotas": "🐾",
    "Moda": "👕", "Música": "🎵",
}
SEARCHES = ["🥔 Patatas", "📖 Novela", "🎧 Auriculares", "☕ Café", "👟 Zapatillas", "🎸 Guitarra"]


def make_stars(rating):
    r = float(rating)
    f = int(r)
    h = 1 if r - f >= 0.3 else 0
    return "★" * f + ("½" if h else "") + "☆" * max(0, 5 - f - h)


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = r"""
<style>
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    .stApp { background: #f5f5f5 !important; }
    .block-container { padding: 10px 12px !important; max-width: 480px !important; }
    .stAppHeader, #MainMenu, div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stVerticalBlock"] > div[style*="flex"] > div { margin-bottom: 4px !important; }
    div[data-testid="column"] { gap: 0 !important; }

    /* ── HEADER ─────────────────────────────────────── */
    .hdr {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #fff; padding: 14px 16px; border-radius: 16px; margin: 0 0 12px 0;
    }
    .hdr h1 { font-size: 18px; margin: 0; font-weight: 700; color: #fff; }
    .hdr p { font-size: 11px; margin: 1px 0 0; opacity: .65; color: #fff; }
    .hdr-sm { padding: 10px 14px; }
    .hdr-sm h1 { font-size: 15px; }

    /* ── CATEGORY GRID ──────────────────────────────── */
    .cg { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 14px 0; }
    .cg-itm {
        flex: 1 0 calc(33.33% - 6px); background: #fff; border-radius: 14px;
        padding: 10px 4px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.04);
        cursor: pointer; transition: transform .12s;
    }
    .cg-itm:active { transform: scale(.95); }
    .cg-itm .ci { font-size: 24px; }
    .cg-itm .cn { font-size: 11px; font-weight: 600; color: #222; margin-top: 3px; }
    .cg-itm .cc { font-size: 9px; color: #999; }

    /* ── PRODUCT CARD ───────────────────────────────── */
    .prod-row {
        background: #fff; border-radius: 12px; padding: 8px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.04); transition: transform .12s;
        display: flex; align-items: center; gap: 8px; margin: 0 !important;
    }
    .prod-row:active { transform: scale(.98); }
    .prod-icon {
        width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 20px;
    }
    .prod-body { flex: 1; min-width: 0; }
    .prod-brand { font-size: 9px; color: #999; text-transform: uppercase; letter-spacing: .3px; }
    .prod-name { font-size: 13px; font-weight: 600; color: #222; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .prod-meta { font-size: 11px; color: #888; }
    .prod-price { font-size: 14px; font-weight: 700; color: #0f3460; flex-shrink: 0; }

    /* ── RELATED CARD ───────────────────────────────── */
    .rel-row {
        background: #fff; border-radius: 10px; padding: 7px 9px;
        box-shadow: 0 1px 2px rgba(0,0,0,.03);
        display: flex; align-items: center; gap: 7px; margin: 0 !important;
    }
    .rel-icon { width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 14px; }
    .rel-body { flex: 1; min-width: 0; }
    .rel-name { font-size: 11px; font-weight: 600; color: #222; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .rel-price { font-size: 12px; font-weight: 700; color: #0f3460; flex-shrink: 0; }
    .rel-badge { display: inline-block; font-size: 9px; padding: 1px 6px; border-radius: 6px; margin-right: 3px; text-transform: uppercase; }
    .rel-badge.sub { background: #e3f2fd; color: #1565c0; }
    .rel-badge.brd { background: #fce4ec; color: #c62828; }
    .rel-badge.sem { background: #f3e5f5; color: #6a1b9a; }
    .rel-badge.crs { background: #e8f5e9; color: #2e7d32; }

    /* ── DETAIL CARD ────────────────────────────────── */
    .dt {
        background: #fff; border-radius: 16px; padding: 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,.04); margin-bottom: 14px;
    }
    .dt .dticon { width: 64px; height: 64px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 32px; margin-bottom: 12px; }
    .dt .dtbrand { font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: .4px; }
    .dt .dtname { font-size: 18px; font-weight: 700; color: #222; margin: 2px 0 3px; line-height: 1.25; }
    .dt .dtprice { font-size: 24px; font-weight: 800; color: #0f3460; margin: 3px 0; }
    .dt .dtrating { font-size: 11px; color: #888; margin-bottom: 4px; }
    .dt .dtstock { font-size: 12px; margin: 4px 0 6px; }
    .dt .dtdesc { font-size: 13px; color: #555; line-height: 1.55; margin: 8px 0; }
    .dt .dtcat { font-size: 12px; color: #888; margin-bottom: 6px; }
    .dt .dtfeats { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
    .dt .dtfeats span { background: #e8f0fe; color: #0f3460; font-size: 10px; font-weight: 500; padding: 3px 10px; border-radius: 10px; }
    .dt .dttags { display: flex; flex-wrap: wrap; gap: 3px; margin: 6px 0; }
    .dt .dttags span { background: #f0f0f0; color: #666; font-size: 9px; padding: 2px 8px; border-radius: 8px; }
    .dt .dtvars { font-size: 11px; color: #888; margin-top: 6px; }

    /* ── SECTION TITLES ─────────────────────────────── */
    .stitle { font-size: 14px; font-weight: 700; color: #222; margin: 0 0 8px 0; }
    .ssub { font-size: 11px; color: #999; margin: -6px 0 10px 0; }

    /* ── SEARCH INPUT ───────────────────────────────── */
    .stTextInput>div>div>input {
        border-radius: 30px !important; padding: 10px 16px !important;
        font-size: 14px !important; border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.06) !important;
    }
    .stTextInput>div>div>input:focus {
        box-shadow: 0 2px 14px rgba(15,52,96,.12) !important;
    }

    /* ── BUTTON OVERRIDES ───────────────────────────── */
    .stButton>button {
        border-radius: 30px !important; font-size: 12px !important;
        padding: 5px 12px !important; border: none !important;
        background: #e8f0fe !important; color: #0f3460 !important;
        font-weight: 600 !important; transition: all .15s !important;
    }
    .stButton>button:hover {
        background: #d2e3fc !important;
    }
    .stButton>button:active {
        transform: scale(.96) !important;
    }
    /* Full width buttons (back, category selects) */
    .stButton>button[kind="secondaryFormSubmit"] {
        background: #fff !important; box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
    }

    /* ── BREADCRUMB ─────────────────────────────────── */
    .bc { font-size: 11px; color: #999; margin-bottom: 8px; }
    .bc span { color: #999; }
    .bc strong { color: #444; }

    /* ── NO RESULTS ─────────────────────────────────── */
    .empty { text-align: center; padding: 40px 20px; color: #999; }
    .empty .eicon { font-size: 40px; margin-bottom: 8px; }
    .empty .etitle { font-size: 15px; font-weight: 600; color: #666; }
    .empty .esub { font-size: 12px; }

    /* ── CHIP ROW ───────────────────────────────────── */
    .chip-row { display: flex; flex-wrap: wrap; gap: 5px; margin: 0 0 12px 0; }
    .chip-row .stButton>button {
        background: #fff !important; border: 1px solid #e0e0e0 !important;
        border-radius: 20px !important; font-size: 11px !important;
        padding: 4px 12px !important; color: #555 !important;
    }
    .chip-row .stButton>button:hover { border-color: #0f3460 !important; color: #0f3460 !important; }
</style>
"""


# ─── PRODUCT CARD (using columns + button) ────────────────────────────────────

def prod_card(p, key_prefix=""):
    """Render a product card row with button click."""
    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    pid = p["id"]
    k = f"{key_prefix}{pid}"
    sc = STOCK_COLORS.get(p.get("stock", "En stock"), "#27ae60")

    cols = st.columns([0.9, 3.2, 1], gap="small")
    with cols[0]:
        st.markdown(f'<div class="prod-icon" style="background:{color};">{icon}</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="prod-brand">{p["brand"]}</div><div class="prod-name">{p["name"][:50]}</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="prod-price">{p["price"]}</div>', unsafe_allow_html=True)

    if st.button(f"Ver", key=k, use_container_width=True):
        st.session_state.selected_product = pid
        st.session_state.page = "product"
        st.rerun()


def rel_card(p, reason="", key_prefix=""):
    """Smaller related product card."""
    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    pid = p["id"]
    k = f"{key_prefix}rel_{pid}"

    badge = {"subcat": "alternativa", "brand": "misma marca", "sem": "similar", "cross": "complemento"}
    bcls = {"subcat": "sub", "brand": "brd", "sem": "sem", "cross": "crs"}
    badge_html = f'<span class="rel-badge {bcls.get(reason, "sem")}">{badge.get(reason, "similar")}</span>' if reason else ""

    cols = st.columns([0.7, 3.5, 1], gap="small")
    with cols[0]:
        st.markdown(f'<div class="rel-icon" style="background:{color};">{icon}</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="rel-name">{p["name"][:45]}</div>{badge_html}', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="rel-price">{p["price"]}</div>', unsafe_allow_html=True)

    if st.button(f"›", key=k, use_container_width=True):
        st.session_state.selected_product = pid
        st.session_state.page = "product"
        st.rerun()


# ─── PAGES ────────────────────────────────────────────────────────────────────

def page_home(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hdr"><h1>🛍️ Catálogo</h1><p>{len(products)} productos · Busca y descubre</p></div>', unsafe_allow_html=True)

    # Search
    query = st.text_input("", placeholder="🔎 Busca productos, marcas...", label_visibility="collapsed", key="hs")
    if query:
        q_emb = model.encode(query)
        results = search_similar(q_emb, embeddings, products)
        st.session_state.results = results
        st.session_state.q = query
        st.session_state.page = "search"
        st.rerun()

    # Quick chips
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    for s in SEARCHES:
        if st.button(s, key=f"ch_{s.split()[0]}", use_container_width=False):
            q_emb = model.encode(s.split(" ", 1)[1] if " " in s else s)
            results = search_similar(q_emb, embeddings, products)
            st.session_state.results = results
            st.session_state.q = s
            st.session_state.page = "search"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Categories
    st.markdown('<div class="stitle">📂 Categorías</div>', unsafe_allow_html=True)
    cats = list(dict.fromkeys(p["category"] for p in products))
    counts = Counter(p["category"] for p in products)

    st.markdown('<div class="cg">', unsafe_allow_html=True)
    for cat in cats:
        emoji = CAT_EMOJIS.get(cat, "📦")
        cnt = counts[cat]
        st.markdown(f'<div class="cg-itm"><div class="ci">{emoji}</div><div class="cn">{cat}</div><div class="cc">{cnt}</div></div>', unsafe_allow_html=True)
        if st.button(f"", key=f"c_{cat}", use_container_width=False):
            st.session_state.cat = cat
            st.session_state.page = "category"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Featured (top rated)
    st.markdown('<div class="stitle">⭐ Destacados</div>', unsafe_allow_html=True)
    featured = sorted(products, key=lambda p: (p["rating"] * min(p["reviews"], 1000)), reverse=True)[:10]
    for p in featured:
        prod_card(p, "fp_")


def page_category(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    cat = st.session_state.get("cat", products[0]["category"])
    cat_prods = [p for p in products if p["category"] == cat]
    emoji = CAT_EMOJIS.get(cat, "📦")

    if st.button("← Volver", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(f'<div class="hdr hdr-sm"><h1>{emoji} {cat}</h1><p>{len(cat_prods)} productos</p></div>', unsafe_allow_html=True)

    query = st.text_input("", placeholder=f"🔎 Buscar en {cat}...", label_visibility="collapsed", key="cs")
    if query:
        q_emb = model.encode(query)
        results = search_similar(q_emb, embeddings, cat_prods)
        st.session_state.results = results
        st.session_state.q = f"{query} en {cat}"
        st.session_state.page = "search"
        st.rerun()

    for p in cat_prods:
        prod_card(p, "cp_")


def page_search(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    results = st.session_state.get("results", [])
    q = st.session_state.get("q", "")

    if st.button("← Volver", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(f'<div class="hdr hdr-sm"><h1>🔍 Resultados</h1><p>{len(results)} para "{q[:40]}{"…" if len(q) > 40 else ""}"</p></div>', unsafe_allow_html=True)

    if not results:
        st.markdown('<div class="empty"><div class="eicon">🔍</div><div class="etitle">Sin resultados</div><div class="esub">Prueba con otros términos</div></div>', unsafe_allow_html=True)
        return

    for r in results:
        prod_card(r, "sr_")


def page_product(products, model, embeddings):
    st.markdown(CSS, unsafe_allow_html=True)
    pid = st.session_state.get("selected_product", products[0]["id"])
    p = get_product(pid, products)
    if not p:
        st.session_state.page = "home"
        st.rerun()
        return

    color = hash_color(p["name"])
    icon = p.get("icon", "📦")
    stock = p.get("stock", "En stock")
    sc = STOCK_COLORS.get(stock, "#27ae60")
    sl = STOCK_LABELS.get(stock, stock)

    if st.button("← Volver", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    # Breadcrumb
    st.markdown(f'<div class="bc">Inicio <span>›</span> {p["category"]} <span>›</span> <strong>{p["name"][:30]}</strong></div>', unsafe_allow_html=True)

    stars = make_stars(p.get("rating", 4.0))

    feats = ""
    if p.get("features"):
        feats = '<div class="dtfeats">' + "".join(f"<span>✓ {f}</span>" for f in p["features"]) + "</div>"
    tags = ""
    if p.get("tags"):
        tags = '<div class="dttags">' + "".join(f"<span>#{t}</span>" for t in p["tags"][:8]) + "</div>"
    vars_ = ""
    if p.get("variants"):
        vars_ = f'<div class="dtvars">📦 Formatos: {", ".join(p["variants"])}</div>'

    st.markdown(f"""
    <div class="dt">
        <div class="dticon" style="background:{color};">{icon}</div>
        <div class="dtbrand">{p['brand']} · {p['subcategory']}</div>
        <div class="dtname">{p['name']}</div>
        <div class="dtprice">{p['price']}</div>
        <div class="dtrating"><span style="color:#f1c40f;">{stars}</span> {p['rating']} ({p.get('reviews', 0):,})</div>
        <div class="dtstock"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{sc};"></span> {sl}</div>
        <div class="dtdesc">{p['description']}</div>
        {feats}
        {tags}
        {vars_}
    </div>
    """, unsafe_allow_html=True)

    # ── RELATED ──
    st.markdown('<div class="stitle">🔗 Relacionados</div>', unsafe_allow_html=True)

    related = []
    seen = set()

    def add_rel(rp, reason):
        if rp["id"] not in seen and rp["id"] != p["id"]:
            related.append((rp, reason))
            seen.add(rp["id"])

    # 1. Same subcategory (alternatives, by price proximity)
    same_sub = [x for x in products if x["subcategory"] == p["subcategory"] and x["id"] != p["id"]]
    same_sub.sort(key=lambda x: abs(x["price_num"] - p["price_num"]))
    for rp in same_sub[:4]:
        add_rel(rp, "subcat")

    # 2. Same brand, different category
    same_brand = [x for x in products if x["brand"] == p["brand"] and x["category"] != p["category"] and x["id"] != p["id"]]
    for rp in same_brand[:2]:
        add_rel(rp, "brand")

    # 3. Explicit related_ids
    for rid in p.get("related_ids", []):
        if len(related) >= 10:
            break
        rp = get_product(rid, products)
        if rp and rp["id"] not in seen and rp["id"] != p["id"]:
            if rp["category"] != p["category"] and rp["brand"] == p["brand"]:
                reason = "brand"
            elif rp["category"] != p["category"]:
                reason = "cross"
            elif rp["subcategory"] == p["subcategory"]:
                reason = "subcat"
            else:
                reason = "sem"
            add_rel(rp, reason)

    # 4. Semantic fill
    if len(related) < 6:
        p_idx = products.index(p)
        q_emb = embeddings[p_idx]
        semantic = search_similar(q_emb, embeddings, products, top_k=12)
        for r in semantic:
            if len(related) >= 10:
                break
            add_rel(r, "sem")

    if related:
        st.markdown(f'<div class="ssub">{len(related)} similares</div>', unsafe_allow_html=True)
        for rp, reason in related[:10]:
            rel_card(rp, reason, f"rp_{p['id']}_")
    else:
        st.markdown('<div style="text-align:center;padding:24px 0;color:#999;font-size:12px;">No hay productos relacionados.</div>', unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "results" not in st.session_state:
        st.session_state.results = []
    if "q" not in st.session_state:
        st.session_state.q = ""

    with st.spinner("Cargando..."):
        model = load_model()
        products, embeddings = load_data()

    page = st.session_state.page
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
