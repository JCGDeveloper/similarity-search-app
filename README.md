# 🔍 Buscador de Productos Similares

App para encontrar productos similares o relacionados con lo que buscas.

## 🚀 Arrancar

```bash
cd ~/Desktop/similarity-search-app
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
streamlit run similarity_app.py
```

## 🎯 Cómo usar

Tres formas de buscar:

1. **📝 Texto** — Escribe lo que buscas (ej: "patatas fritas", "novela de misterio")
2. **🔗 Enlace** — Pega una URL de producto y extrae información automáticamente
3. **📸 Imagen** — Sube una foto y encuentra productos visualmente similares

## 🛠️ Stack

- Streamlit (UI)
- sentence-transformers (embeddings semánticos)
- FAISS (búsqueda por similitud)
- 75 productos en 7 categorías

Creado por José Luis para Joaquín 🛠️
