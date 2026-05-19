"""
📦 Generador de Base de Datos de Productos con Relaciones
Genera 500+ productos reales con categorías, marcas, etiquetas y relaciones.
"""

import json
import os
import pickle
import random
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent
random.seed(42)

# ─── CATEGORÍAS PRINCIPALES ──────────────────────────────────────────────────

SUBCATEGORIES = {
    "Alimentación": {
        "icon": "🍕",
        "subs": [
            "Snacks salados", "Snacks dulces", "Bebidas", "Galletas y pastas",
            "Chocolates", "Frutos secos", "Conservas", "Desayunos",
            "Salsas y condimentos", "Pastas y arroces", "Aceites y vinagres",
            "Cafés e infusiones", "Productos ecológicos"
        ]
    },
    "Libros": {
        "icon": "📚",
        "subs": [
            "Novela", "Ensayo", "Ciencia ficción", "Historia",
            "Autoayuda", "Infantil", "Poesía", "Cómic y novela gráfica",
            "Cocina", "Viajes", "Tecnología", "Filosofía"
        ]
    },
    "Electrónica": {
        "icon": "🔌",
        "subs": [
            "Auriculares", "Móviles", "Portátiles", "Tablets",
            "Smartwatches", "Altavoces", "Cámaras", "Monitores",
            "Teclados", "Ratones", "Cargadores", "Discos duros"
        ]
    },
    "Hogar": {
        "icon": "🏠",
        "subs": [
            "Cocina", "Limpieza", "Decoración", "Menaje",
            "Iluminación", "Jardín", "Baño", "Dormitorio",
            "Organización", "Bricolaje", "Muebles", "Electrodomésticos"
        ]
    },
    "Moda": {
        "icon": "👕",
        "subs": [
            "Zapatillas", "Ropa deportiva", "Accesorios", "Bolsos y mochilas",
            "Relojes", "Gafas", "Joyas", "Bufandas y complementos"
        ]
    },
    "Juguetes": {
        "icon": "🎮",
        "subs": [
            "Construcción", "Puzzles", "Juegos de mesa", "Muñecas",
            "Vehículos", "Juguetes educativos", "Películas", "Instrumentos musicales infantiles"
        ]
    },
    "Deportes": {
        "icon": "⚽",
        "subs": [
            "Fútbol", "Running", "Ciclismo", "Natación",
            "Yoga", "Gimnasio", "Escalada", "Tenis",
            "Pádel", "Camping", "Deportes de invierno"
        ]
    },
    "Belleza": {
        "icon": "💄",
        "subs": [
            "Perfumes", "Maquillaje", "Cuidado facial", "Cuidado corporal",
            "Cuidado capilar", "Hombre", "Protección solar", "Bienestar"
        ]
    },
    "Música": {
        "icon": "🎵",
        "subs": [
            "Instrumentos", "Vinilos", "Equipo de sonido", "Accesorios musicales"
        ]
    },
    "Mascotas": {
        "icon": "🐾",
        "subs": [
            "Perros", "Gatos", "Peces", "Aves",
            "Roedores", "Accesorios"
        ]
    }
}

# ─── MARCAS POR CATEGORÍA ────────────────────────────────────────────────────

BRANDS = {
    "Alimentación": ["Lays", "Ruffles", "Doritos", "Pringles", "Milka", "Lindt", "Nestlé", "Coca-Cola",
                     "Pepsi", "Danone", "Act II", "Oreo", "Borges", "Nature Valley", "MC Vities",
                     "Don Simón", "Grefusa", "Matutano", "El Caserío", "Heinz", "Gallo", "La Piara",
                     "Nocilla", "ColaCao", "Font Vella", "Mahou", "Estrella Galicia"],
    "Libros": ["Penguin Random House", "Planeta", "Anagrama", "Alfaguara", "Siruela", "Tusquets",
               "Seix Barral", "Debolsillo", "Cátedra", "Alianza", "Salamandra", "Destino",
               "HarperCollins", "Ediciones B", "Lumen"],
    "Electrónica": ["Sony", "Samsung", "Apple", "Bose", "JBL", "Logitech", "Razer", "ASUS",
                    "Dell", "HP", "LG", "Xiaomi", "Nothing", "Beats", "Marshall", "Sennheiser",
                    "Shure", "Corsair", "HyperX", "SteelSeries"],
    "Hogar": ["Tefal", "Philips", "Bosch", "Dyson", "Roomba", "Roborock", "De'Longhi",
              "Nespresso", "Zwilling", "IKEA", "Leroy Merlin", "Emma", "Pikolin",
              "Rowenta", "Cecotec", "Orbegozo", "Balay", "Samsung"],
    "Moda": ["Nike", "Adidas", "New Balance", "Puma", "The North Face", "Eastpak",
             "Herschel", "Casio G-Shock", "Ray-Ban", "Tommy Hilfiger", "Levi's",
             "Bershka", "Zara", "Pull&Bear", "Decathlon", "Reebok", "Converse"],
    "Juguetes": ["LEGO", "Ravensburger", "Hasbro", "Barbie", "Hot Wheels", "Mattel",
                 "Playmobil", "Educa", "Fisher-Price", "Nerf", "Science4you", "Clementoni"],
    "Deportes": ["Nike", "Adidas", "Puma", "Decathlon", "Head", "Wilson", "Babolat",
                 "Salomon", "The North Face", "Patagonia", "Columbia", "Camelbak",
                 "Garmin", "Suunto", "Polar"],
    "Belleza": ["Chanel", "Dior", "Lancôme", "Yves Saint Laurent", "Garnier", "L'Oréal",
                "Nivea", "Dove", "Bioderma", "La Roche-Posay", "Vichy", "Isdin",
                "Carolina Herrera", "Paco Rabanne"],
    "Música": ["Yamaha", "Casio", "Fender", "Gibson", "Marshall", "Bose", "Sony",
               "Audio-Technica", "Roland", "Shure"],
    "Mascotas": ["Royal Canin", "Purina", "Affinity", "Hill's", "Acana", "Orijen",
                 "Compesan", "Tiendanimal", "Ferplast", "Trixie"]
}

# ─── GENERADOR DE PRODUCTOS ──────────────────────────────────────────────────

def generate_products():
    """Generate 500+ products with rich metadata."""
    products = []
    product_id = 0

    # ─── ALIMENTACIÓN ──────────────────────────────────────────────────────
    food_templates = [
        ("Patatas Fritas {} {}", lambda b, sub: f"{b} - {sub}"),
        ("Galletas {} {}", lambda b, sub: f"{b} - {sub}"),
        ("Chocolate {} {}", lambda b, sub: f"{b} - Tableta {sub}"),
        ("Barrita de Cereal {}", lambda b, sub: f"{b} - {sub}"),
        ("Zumo de Frutas {}", lambda b, sub: f"{b} - {sub}"),
        ("Café Molido {}", lambda b, sub: f"{b} - {sub}"),
        ("Arroz {} kg", lambda b, sub: f"{b} - {sub}kg"),
        ("Aceite de Oliva {}", lambda b, sub: f"{b} - {sub}L"),
        ("Refresco {} {}", lambda b, sub: f"{b} - {sub}L"),
        ("Cerveza {}", lambda b, sub: f"{b} - Pack {sub}"),
    ]
    food_descs = [
        "Crujientes y sabrosas, ideales para picar entre horas.",
        "El clásico que nunca falla, perfecto para cualquier ocasión.",
        "Suave y cremoso, elaborado con los mejores ingredientes.",
        "Energía natural para tu día a día, sin azúcares añadidos.",
        "Refrescante y natural, exprimido en su punto óptimo de maduración.",
        "Aroma intenso y sabor equilibrado, molido para cafetera.",
        "Grano seleccionado de la mejor cosecha, cocción perfecta.",
        "Virgen extra de primera presión en frío, sabor afrutado.",
        "Burbujas finas y sabor inconfundible, edición limitada.",
        "Cerveza artesana de alta fermentación, lúpulo seleccionado.",
    ]

    for cat_name, cat_data in SUBCATEGORIES.items():
        cat_icon = cat_data["icon"]
        subcats = cat_data["subs"]
        brand_list = BRANDS.get(cat_name, ["Marca Genérica"])

        num_products_per_cat = random.randint(40, 60)

        for _ in range(num_products_per_cat):
            product_id += 1
            subcat = random.choice(subcats)
            brand = random.choice(brand_list)

            # Generate a name
            if cat_name == "Alimentación":
                idx = random.randint(0, len(food_templates)-1)
                base_name, gen_func = food_templates[idx]
                suffix = random.choice(["Clásico", "Premium", "Familiar", "Edición Limitada", "Ecológico", "Tradicional", "Light", "XXL"])
                name = f"{base_name} {suffix}"
            else:
                adjectives = ["Premium", "Clásico/a", "Profesional", "Ultra", "Compacto/a",
                              "Ecológico/a", "De lujo", "Inteligente", "Portátil", "Ergonómico/a",
                              "Elegante", "Moderno/a", "XXL", "Ligero/a", "Resistente"]
                types = ["Edición Limitada", "Modelo Avanzado", "Serie Oro", "Versión Pro",
                         "Básico", "Plus", "Max", "Air", "Next Gen", "V2"]
                name = f"{brand} {random.choice(adjectives)} {random.choice(types)}"

            # Generate a price
            if cat_name in ["Electrónica", "Hogar", "Deportes"]:
                price = random.randint(10, 2000)
            elif cat_name in ["Alimentación", "Belleza"]:
                price = random.randint(1, 50)
            elif cat_name == "Libros":
                price = random.randint(7, 40)
            elif cat_name == "Moda":
                price = random.randint(15, 500)
            elif cat_name == "Juguetes":
                price = random.randint(5, 200)
            elif cat_name == "Música":
                price = random.randint(15, 1500)
            elif cat_name == "Mascotas":
                price = random.randint(5, 80)
            else:
                price = random.randint(5, 500)

            # Generate description
            if cat_name == "Alimentación":
                desc = random.choice(food_descs)
            else:
                features = random.sample([
                    "alta calidad", "materiales premium", "diseño innovador",
                    "fabricación sostenible", "larga duración", "fácil de usar",
                    "garantía incluida", "envío rápido", "embalaje ecológico",
                    "resistente y duradero", "compacto y ligero",
                    "fácil mantenimiento", "versátil y multiuso",
                    "edición especial", "color exclusivo"
                ], k=3)
                desc = f"{cat_name.replace(chr(237), 'i')} de {random.choice(features)} con acabados {features[1]}. {features[2].capitalize()}."

            # Tags
            all_tags = [subcat.lower(), brand.lower().split()[0].lower(), cat_name.lower()]
            extra_tags = random.sample([
                "oferta", "novedad", "top ventas", "recomendado", "ecológico",
                "premium", "básico", "coleccionista", "regalo", "viaje",
                "hogar", "profesional", "iniciación", "edición limitada",
                "sostenible", "garantía", "español", "importado"
            ], k=3)
            all_tags.extend(extra_tags)

            # Weight / size info
            if cat_name == "Alimentación":
                variants = ["200g", "500g", "1kg", "pack 3", "pack 6"]
            elif cat_name == "Electrónica":
                variants = ["1 unidad", "kit básico", "pack profesional"]
            elif cat_name == "Moda":
                variants = ["Talla única", "S/M", "L/XL"]
            else:
                variants = ["1 unidad", "lote 2 uds", "pack ahorro"]

            product = {
                "id": f"PROD_{product_id:04d}",
                "name": name,
                "brand": brand,
                "category": cat_name,
                "subcategory": subcat,
                "icon": cat_icon,
                "description": desc,
                "price": f"{price:.2f}€" if price < 10 else f"{price}€",
                "price_num": price,
                "tags": all_tags,
                "variants": random.sample(variants, k=random.randint(1, 3)),
                "rating": round(3.0 + random.random() * 2.0, 1),
                "reviews": random.randint(0, 5000),
                "stock": random.choice(["En stock", "Pocas unidades", "Agotado"]),
                "features": random.sample([
                    "Garantía 2 años", "Envío gratuito", "Devolución 30 días",
                    "Fabricado en España", "Certificado ecológico", "Hipoalergénico",
                    "Apto veganos", "Sin gluten", "Resistente al agua",
                    "Batería recargable", "App incluida", "Bluetooth 5.3"
                ], k=random.randint(2, 4)) if price > 20 else [],
                "related_ids": [],  # Will fill later
            }
            products.append(product)

    # ─── ESTABLECER RELACIONES ──────────────────────────────────────────────
    # For each product, find related products by:
    # 1. Same subcategory (direct substitutes)
    # 2. Same category but different subcategory (complementary)
    # 3. Same brand (other products from same brand)
    # 4. Cross-category (e.g., coffee + coffee mugs)

    for p in products:
        # Build candidate pool
        same_sub = [x for x in products if x["subcategory"] == p["subcategory"] and x["id"] != p["id"]]
        same_cat = [x for x in products if x["category"] == p["category"] and x["subcategory"] != p["subcategory"] and x["id"] != p["id"]]
        same_brand = [x for x in products if x["brand"] == p["brand"] and x["category"] != p["category"] and x["id"] != p["id"]]

        # Complementary cross-category relationships
        cross_relations = {
            "Cafés e infusiones": ["Tazas", "Cafeteras", "Leches vegetales"],
            "Snacks salados": ["Salsas y condimentos", "Bebidas", "Cerveza"],
            "Snacks dulces": ["Chocolates", "Bebidas", "Desayunos"],
            "Móviles": ["Auriculares", "Cargadores", "Fundas"],
            "Portátiles": ["Monitores", "Teclados", "Ratones", "Auriculares", "Mochilas"],
            "Running": ["Zapatillas", "Ropa deportiva", "Smartwatches", "Camelbak"],
            "Cocina": ["Sartenes", "Cuchillos", "Menaje", "Electrodomésticos"],
            "Perros": ["Gatos", "Accesorios mascotas"],
            "Gatos": ["Perros", "Accesorios mascotas"],
        }

        cross_prods = []
        sub = p["subcategory"]
        if sub in cross_relations:
            targets = cross_relations[sub]
            cross_prods = [x for x in products if x["subcategory"] in targets and x["id"] != p["id"]]

        # Select related products from each pool (prioritize semantic)
        related = []

        # Same subcategory: strongest relation, take up to 4
        if same_sub:
            same_sub_sorted = sorted(same_sub, key=lambda x: abs(x["price_num"] - p["price_num"]))
            related.extend(random.sample(same_sub_sorted, min(4, len(same_sub_sorted))))

        # Same brand cross-category: take up to 2
        if same_brand:
            related.extend(random.sample(same_brand, min(2, len(same_brand))))

        # Cross-category complementary: take up to 2
        if cross_prods:
            related.extend(random.sample(cross_prods, min(2, len(cross_prods))))

        # Same category other subcats: fill up to 6 total
        if len(related) < 6 and same_cat:
            remaining = [x for x in same_cat if x not in related]
            needed = min(6 - len(related), len(remaining))
            related.extend(random.sample(remaining, needed))

        # Remove duplicates while preserving order
        seen = set()
        unique_related = []
        for r in related:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique_related.append(r)
                if len(unique_related) >= 8:
                    break

        p["related_ids"] = [r["id"] for r in unique_related]

    return products


def generate_embeddings(products):
    """Generate semantic embeddings for all products."""
    print(f"📦 Generando embeddings para {len(products)} productos...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = []
    for p in products:
        combined = f"{p['name']}. {p['brand']}. {p['category']} - {p['subcategory']}. {p['description']}. Tags: {', '.join(p['tags'])}"
        texts.append(combined)

    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def main():
    print("🔨 Generando base de datos de productos...")
    products = generate_products()
    print(f"   ✅ {len(products)} productos generados")

    # Count by category
    from collections import Counter
    cat_counts = Counter(p["category"] for p in products)
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"      {SUBCATEGORIES[cat]['icon']} {cat}: {count}")

    # Count relationships
    total_relations = sum(len(p["related_ids"]) for p in products)
    avg_relations = total_relations / len(products)
    print(f"\n   🔗 Relaciones totales: {total_relations} (avg {avg_relations:.1f} por producto)")

    # Generate embeddings
    embeddings = generate_embeddings(products)

    # Save
    data_path = os.path.join(DATA_DIR, "product_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({
            "products": products,
            "embeddings": embeddings,
        }, f)

    # Also save a JSON preview
    preview = []
    for p in products[:5]:
        preview.append({
            "id": p["id"],
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "price": p["price"],
            "related_count": len(p["related_ids"]),
        })
    preview_path = os.path.join(DATA_DIR, "product_preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Datos guardados en {data_path}")
    print(f"   Preview guardado en {preview_path}")


if __name__ == "__main__":
    main()
