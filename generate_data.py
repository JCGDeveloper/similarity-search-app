"""
📦 Generador de Base de Datos de Productos con Relaciones
Genera 500+ productos reales con categorías, marcas, etiquetas y relaciones.
Versión mejorada: descripciones coherentes, nombres correctos, más variedad.
"""

import json
import os
import pickle
import random
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

# ─── DESCRIPCIONES POR SUBCATEGORÍA ─────────────────────────────────────────

def make_food_desc(subcat):
    """Return sensible descriptions based on the actual food subcategory."""
    descs = {
        "Snacks salados": [
            "Crujientes y sabrosas, el snack perfecto para cualquier momento del día.",
            "El clásico que nunca falla, ideal para compartir con amigos.",
            "Punto exacto de sal, textura crujiente que engancha.",
            "Sabor intenso y textura inigualable. Bolsa grande para compartir.",
            "Receta tradicional con un toque moderno. Perfectas para el aperitivo."
        ],
        "Snacks dulces": [
            "Dulce tentación en cada bocado. El capricho que te mereces.",
            "Suave y cremoso, elaborado con ingredientes seleccionados.",
            "El dulce perfecto para alegrar cualquier momento del día.",
            "Combinación irresistible de sabores. Para los más golosos.",
            "Textura suave y sabor intenso. El placer de un buen dulce."
        ],
        "Bebidas": [
            "Refrescante y natural, elaborada con ingredientes de primera calidad.",
            "Burbujas finas y sabor inconfundible. Ideal para cualquier ocasión.",
            "El sabor original que tanto te gusta, ahora en formato práctico.",
            "Receta equilibrada, perfecta para acompañar tus comidas.",
            "Frescura garantizada en cada sorbo. Nutritivo y delicioso."
        ],
        "Galletas y pastas": [
            "Crujientes por fuera, tiernas por dentro. Horneadas artesanalmente.",
            "La merienda perfecta. Acompaña tu café o té favorito.",
            "Receta tradicional con mantequilla. Como las de antes.",
            "Galletas con trocitos de chocolate. Irresistibles para toda la familia.",
            "Textura crujiente y sabor auténtico. El clásico de siempre."
        ],
        "Chocolates": [
            "Cacao puro de comercio justo. Sabor intenso y sedoso.",
            "Tableta de chocolate belga. Suavidad y sabor en cada onza.",
            "Cobertura de chocolate negro 70%. Para auténticos amantes del cacao.",
            "Praliné artesanal con avellanas enteras. Una delicia única.",
            "Chocolate con leche cremoso. El capricho dulce perfecto."
        ],
        "Frutos secos": [
            "Selección de frutos secos tostados y salados. Energía natural.",
            "Cacahuetes crujientes con un punto justo de sal. Aperitivo saludable.",
            "Mezcla de frutos secos variados. Energía para tu día a día.",
            "Almendras seleccionadas, tostadas y ligeramente saladas.",
            "Nueces de California, ricas en omega-3. Saludables y deliciosas."
        ],
        "Conservas": [
            "Conserva tradicional elaborada con productos de la huerta.",
            "Aceite de oliva virgen extra y productos seleccionados. Calidad garantizada.",
            "Receta de la abuela en conserva. Sabor casero auténtico.",
            "Productos del mar en conserva. Listos para disfrutar.",
            "Verduras en conserva de temporada. Frescura todo el año."
        ],
        "Desayunos": [
            "Empieza el día con energía. Desayuno completo y equilibrado.",
            "Cereales tostados con miel. El desayuno que te da energía.",
            "Tostadas crujientes, perfectas para untar. El clásico del desayuno.",
            "Mezcla de cereales y frutas para un desayuno nutritivo.",
            "Porridge cremoso de avena. Desayuno caliente y reconfortante."
        ],
        "Salsas y condimentos": [
            "Salsa artesanal con ingredientes naturales. El toque que lo cambia todo.",
            "Receta tradicional, el acompañamiento perfecto para tus platos.",
            "Aliño mediterráneo con hierbas aromáticas. Frescura y sabor.",
            "Salsa picante con el punto exacto. Para los que se atreven.",
            "Mayonesa artesanal con huevos de corral. Cremosa y deliciosa."
        ],
        "Pastas y arroces": [
            "Pasta de sémola de trigo duro. Cocción perfecta al dente.",
            "Arroz de grano redondo, ideal para paellas y risottos.",
            "Pasta artesanal italiana. La auténtica calidad importada.",
            "Arroz basmati de grano largo. Aroma y sabor exóticos.",
            "Pasta integral rica en fibra. Saludable sin sacrificar el sabor."
        ],
        "Aceites y vinagres": [
            "Aceite de oliva virgen extra. Primera presión en frío, sabor afrutado.",
            "AOVE de cosecha temprana. Verde, frutado y ligeramente picante.",
            "Vinagre balsámico de Módena envejecido 12 meses.",
            "Aceite de oliva suave, ideal para cocinar y freír.",
            "Aceite de girasol alto oleico. Perfecto para frituras saludables."
        ],
        "Cafés e infusiones": [
            "Café molido de tueste natural. Aroma intenso y sabor equilibrado.",
            "Cápsulas de café compatibles. La dosis perfecta cada mañana.",
            "Infusión de hierbas relajante. Ideal para después de cenar.",
            "Té verde orgánico. Antioxidante y revitalizante.",
            "Café de especialidad de origen único. Para paladares exigentes."
        ],
        "Productos ecológicos": [
            "Producto ecológico certificado. Cultivado sin pesticidas ni químicos.",
            "De cultivo ecológico y comercio justo. Sabor auténtico y responsable.",
            "Alimento bio con certificado ecológico. Cuida de ti y del planeta.",
            "Pack de productos ecológicos variados. La cesta saludable.",
            "Superfood orgánico. Energía pura de la naturaleza."
        ]
    }
    return random.choice(descs.get(subcat, ["Producto de alta calidad. Disfruta de su sabor auténtico y natural."]))


def make_general_desc(category, subcat, brand):
    """Generate a natural description for non-food categories."""
    templates = [
        f"{category} de alto rendimiento con diseño vanguardista. {random.choice(['Máxima calidad', 'Materiales premium', 'Tecnología puntera'])}.",
        f"{subcat} profesional de la marca {brand}. {random.choice(['Durabilidad garantizada', 'Diseño ergonómico', 'Acabados impecables'])}.",
        f"Descubre el {subcat.lower()} perfecto de {brand}. {random.choice(['Funcional y elegante', 'Innovador y práctico', 'Potente y fiable'])}.",
    ]
    extra = [
        "Fabricado con materiales sostenibles. Apuesta por un consumo responsable.",
        "Diseñado para durar. Garantía de calidad incluida.",
        "Ligero, compacto y fácil de transportar. Llévalo contigo a todas partes.",
        "Edición limitada con acabados exclusivos. Un producto único.",
        "Resultados profesionales a tu alcance. Calidad premium a buen precio.",
    ]
    desc = random.choice(templates) + " " + random.choice(extra)
    return desc


# ─── FEATURES POR CATEGORÍA ─────────────────────────────────────────────────

FEATURES_BY_CAT = {
    "Alimentación": ["Sin gluten", "Sin azúcares añadidos", "Apto veganos", "Fuente de fibra",
                     "Alto contenido en proteínas", "Sin aceite de palma", "Sin conservantes",
                     "Ingredientes naturales", "Certificado ecológico"],
    "Electrónica": ["Bluetooth 5.3", "Batería recargable", "App incluida", "Garantía 2 años",
                    "Resistente al agua", "Carga rápida", "Cancelación de ruido", "Emparejamiento NFC",
                    "Actualización firmware", "Compatibilidad universal"],
    "Hogar": ["Fácil montaje", "Materiales reciclados", "Ahorro energético", "Lavable a máquina",
              "Antideslizante", "Fácil limpieza", "Apilable", "Plegable", "Resistente altas temperaturas"],
    "Moda": ["Lavable a máquina", "Secado rápido", "Tejido transpirable", "Costuras reforzadas",
             "Material reciclado", "Ajuste ergonómico", "Ligero", "Resistente al agua"],
    "Deportes": ["Transpirable", "Ligero", "Agarre óptimo", "Amortiguación", "Antideslizante",
                 "Compresión", "Aislamiento térmico", "Resistente al agua"],
    "Belleza": ["Dermatológicamente testado", "Sin parabenos", "Vegano", "Hipoalergénico",
                "Protección solar SPF50", "Sin alcohol", "Apto pieles sensibles", "Eco-friendly"],
    "Libros": ["Edición de bolsillo", "Tapa dura", "Ilustrado", "Edición especial", "Incluye marcapáginas",
               "Letra grande", "Edición anotada", "Idioma original"],
    "Juguetes": ["Edad recomendada 3+", "Certificado CE", "Materiales no tóxicos", "Piezas lavables",
                 "Pilas incluidas", "Fomenta la creatividad", "Aprendizaje STEM", "Juego colaborativo"],
    "Música": ["Afinación precisa", "Estuche incluido", "Cuerdas de repuesto", "Pastillas activas",
               "Salida USB", "Conectividad MIDI", "Pedal de sustain incluido"],
    "Mascotas": ["Apto todas las razas", "Sin cereales", "Alto contenido en proteína animal",
                 "Enriquecimiento ambiental", "Fácil limpieza", "Material no tóxico", "Ajustable"]
}


def pick_features(category, price):
    """Pick appropriate features based on category and price."""
    base = FEATURES_BY_CAT.get(category, ["Garantía de calidad"])
    if price > 20 and len(base) >= 2:
        return random.sample(base, k=min(random.randint(2, 4), len(base)))
    return []


# ─── GENERADOR DE PRODUCTOS ──────────────────────────────────────────────────

def generate_products():
    """Generate 500+ products with rich metadata and coherent names/descriptions."""
    products = []
    product_id = 0

    # Food-specific product templates (name + proper description)
    # Each entry: (name_template, subcat_hint, price_range)
    # name_template uses {brand} and {suffix}
    food_templates_by_subcat = {
        "Snacks salados": [
            ("Patatas Fritas {brand} {suffix}", (2, 6)),
            ("Nachos {brand} {suffix}", (2, 5)),
            ("Gusanitos {brand} {suffix}", (1, 4)),
            ("Mix de Snacks {brand} {suffix}", (3, 8)),
            ("Palomitas {brand} {suffix}", (2, 5)),
        ],
        "Snacks dulces": [
            ("Barrita Energética {brand} {suffix}", (2, 5)),
            ("Mix Dulce {brand} {suffix}", (3, 7)),
            ("Alfajor {brand} {suffix}", (2, 5)),
            ("Piruleta {brand} {suffix}", (1, 3)),
        ],
        "Bebidas": [
            ("Zumo de Frutas {brand} {suffix}", (2, 5)),
            ("Refresco {brand} {suffix}", (1, 4)),
            ("Agua Mineral {brand} {suffix}", (1, 3)),
            ("Batido {brand} {suffix}", (2, 4)),
        ],
        "Galletas y pastas": [
            ("Galletas {brand} {suffix}", (2, 5)),
            ("Magdalenas {brand} {suffix}", (3, 6)),
            ("Bizcocho {brand} {suffix}", (3, 8)),
            ("Barquillos {brand} {suffix}", (2, 4)),
        ],
        "Chocolates": [
            ("Chocolate con Leche {brand} {suffix}", (3, 8)),
            ("Chocolate Negro {brand} {suffix}", (3, 10)),
            ("Bombones {brand} {suffix}", (5, 15)),
            ("Crema de Cacao {brand} {suffix}", (3, 6)),
        ],
        "Frutos secos": [
            ("Cacahuetes Tostados {brand} {suffix}", (2, 5)),
            ("Almendras {brand} {suffix}", (3, 8)),
            ("Mix de Frutos Secos {brand} {suffix}", (4, 10)),
            ("Pistachos {brand} {suffix}", (4, 9)),
        ],
        "Conservas": [
            ("Atún en Aceite {brand} {suffix}", (2, 5)),
            ("Espárragos Blancos {brand} {suffix}", (3, 6)),
            ("Pimiento Asado {brand} {suffix}", (2, 5)),
            ("Mejillones en Escabeche {brand} {suffix}", (3, 6)),
        ],
        "Desayunos": [
            ("Cereales Integrales {brand} {suffix}", (3, 6)),
            ("Muesli {brand} {suffix}", (4, 8)),
            ("Porridge de Avena {brand} {suffix}", (3, 7)),
            ("Tostadas Integrales {brand} {suffix}", (2, 4)),
        ],
        "Salsas y condimentos": [
            ("Salsa Kétchup {brand} {suffix}", (2, 4)),
            ("Mayonesa {brand} {suffix}", (2, 5)),
            ("Aliño Mediterráneo {brand} {suffix}", (3, 6)),
            ("Salsa Picante {brand} {suffix}", (3, 6)),
        ],
        "Pastas y arroces": [
            ("Pasta Espagueti {brand} {suffix}", (2, 5)),
            ("Arroz Redondo {brand} {suffix}", (2, 5)),
            ("Pasta Fusilli {brand} {suffix}", (2, 5)),
            ("Arroz Basmati {brand} {suffix}", (3, 7)),
        ],
        "Aceites y vinagres": [
            ("Aceite de Oliva Virgen Extra {brand} {suffix}", (5, 15)),
            ("Vinagre Balsámico {brand} {suffix}", (4, 10)),
            ("Aceite de Girasol {brand} {suffix}", (3, 6)),
        ],
        "Cafés e infusiones": [
            ("Café Molido {brand} {suffix}", (3, 8)),
            ("Cápsulas de Café {brand} {suffix}", (4, 10)),
            ("Té Verde {brand} {suffix}", (3, 7)),
            ("Infusión Relajante {brand} {suffix}", (3, 6)),
        ],
        "Productos ecológicos": [
            ("Pack Ecológico {brand} {suffix}", (5, 15)),
            ("Quinoa Ecológica {brand} {suffix}", (4, 9)),
            ("Superfood Mix {brand} {suffix}", (6, 14)),
        ]
    }

    food_suffixes = ["Clásico", "Premium", "Familiar", "Edición Limitada", "Tradicional", "Light", "XXL", "Ecológico"]

    # Non-food templates
    nonfood_adjectives = ["Premium", "Clásico", "Profesional", "Ultra", "Compacto",
                          "Ecológico", "De lujo", "Inteligente", "Portátil", "Ergonómico",
                          "Elegante", "Moderno", "XXL", "Ligero", "Resistente"]
    nonfood_types = ["Edición Limitada", "Modelo Avanzado", "Serie Oro", "Versión Pro",
                     "Básico", "Plus", "Max", "Air", "Next Gen", "V2"]

    for cat_name, cat_data in SUBCATEGORIES.items():
        cat_icon = cat_data["icon"]
        subcats = cat_data["subs"]
        brand_list = BRANDS.get(cat_name, ["Marca Genérica"])

        num_products = random.randint(40, 55)

        for _ in range(num_products):
            product_id += 1
            subcat = random.choice(subcats)
            brand = random.choice(brand_list)

            # --- Generate a proper name ---
            if cat_name == "Alimentación":
                templates_for_sub = food_templates_by_subcat.get(subcat, food_templates_by_subcat["Snacks salados"])
                tmpl = random.choice(templates_for_sub)
                name_tmpl, price_range = tmpl
                suffix = random.choice(food_suffixes)
                name = name_tmpl.format(brand=brand.split()[0], suffix=suffix)
                
                # Price range: (min, max) or fixed
                if isinstance(price_range, tuple):
                    price = random.randint(price_range[0], price_range[1])
                else:
                    price = price_range
                
                desc = make_food_desc(subcat)
            else:
                adj = random.choice(nonfood_adjectives)
                tp = random.choice(nonfood_types)
                name = f"{brand} {adj} {tp}"
                
                # Price ranges by category
                price_ranges = {
                    "Electrónica": (10, 2000),
                    "Hogar": (10, 800),
                    "Moda": (15, 500),
                    "Deportes": (10, 600),
                    "Libros": (7, 45),
                    "Juguetes": (5, 200),
                    "Belleza": (5, 80),
                    "Música": (20, 2000),
                    "Mascotas": (5, 90),
                }
                lo, hi = price_ranges.get(cat_name, (5, 500))
                price = random.randint(lo, hi)
                desc = make_general_desc(cat_name, subcat, brand)

            # --- Price formatting ---
            if price >= 100:
                price_str = f"{price}€"
            elif price >= 10:
                price_str = f"{price}€"
            else:
                price_str = f"{price:.2f}€"

            # --- Rating +/- reviews ---
            rating = round(3.0 + random.random() * 2.0, 1)
            reviews = random.randint(0, 5000)

            # --- Stock ---
            if rating >= 4.5:
                stock_weights = ["En stock", "En stock", "Pocas unidades", "Pocas unidades", "Agotado"]
            else:
                stock_weights = ["En stock", "En stock", "En stock", "Pocas unidades", "Agotado"]
            stock = random.choice(stock_weights)

            # --- Tags ---
            all_tags = list(set([subcat.lower(), brand.lower().split()[0].lower(), cat_name.lower()]))
            extra_tags = random.sample([
                "oferta", "novedad", "top ventas", "recomendado", "ecológico",
                "premium", "básico", "coleccionista", "regalo", "viaje",
                "hogar", "profesional", "iniciación", "edición limitada",
                "sostenible", "garantía", "español", "importado"
            ], k=3)
            all_tags.extend(extra_tags)

            # --- Variants ---
            variant_map = {
                "Alimentación": ["200g", "500g", "1kg", "pack 3", "pack 6"],
                "Electrónica": ["1 unidad", "kit básico", "pack profesional"],
                "Moda": ["Talla única", "S/M", "L/XL"],
                "Libros": ["Tapa blanda", "Tapa dura", "Edición digital"],
                "Belleza": ["30ml", "50ml", "100ml"],
            }
            variants = random.sample(variant_map.get(cat_name, ["1 unidad", "lote 2 uds", "pack ahorro"]), k=random.randint(1, 3))

            product = {
                "id": f"PROD_{product_id:04d}",
                "name": name,
                "brand": brand,
                "category": cat_name,
                "subcategory": subcat,
                "icon": cat_icon,
                "description": desc,
                "price": price_str,
                "price_num": price,
                "tags": all_tags[:10],
                "variants": variants,
                "rating": rating,
                "reviews": reviews,
                "stock": stock,
                "features": pick_features(cat_name, price),
                "related_ids": [],
            }
            products.append(product)

    # ─── ESTABLECER RELACIONES ──────────────────────────────────────────────
    for p in products:
        same_sub = [x for x in products if x["subcategory"] == p["subcategory"] and x["id"] != p["id"]]
        same_cat = [x for x in products if x["category"] == p["category"] and x["subcategory"] != p["subcategory"] and x["id"] != p["id"]]
        same_brand = [x for x in products if x["brand"] == p["brand"] and x["category"] != p["category"] and x["id"] != p["id"]]

        # Cross-category complementary relationships
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
            "Vehículos": ["Juguetes educativos", "Construcción"],
            "Ropa deportiva": ["Zapatillas", "Accesorios", "Running"],
            "Zapatillas": ["Ropa deportiva", "Accesorios"],
        }

        cross_prods = []
        sub = p["subcategory"]
        if sub in cross_relations:
            targets = cross_relations[sub]
            cross_prods = [x for x in products if x["subcategory"] in targets and x["id"] != p["id"]]

        related = []

        # Same subcategory: strongest relation, take up to 4 (by price proximity)
        if same_sub:
            same_sub_sorted = sorted(same_sub, key=lambda x: abs(x["price_num"] - p["price_num"]))
            related.extend(random.sample(same_sub_sorted, min(4, len(same_sub_sorted))))

        # Same brand cross-category: take up to 2
        if same_brand:
            related.extend(random.sample(same_brand, min(2, len(same_brand))))

        # Cross-category complementary: take up to 2
        if cross_prods:
            related.extend(random.sample(cross_prods, min(2, len(cross_prods))))

        # Same category other subcats: fill up to 8 total
        if len(related) < 8 and same_cat:
            remaining = [x for x in same_cat if x not in related]
            needed = min(8 - len(related), len(remaining))
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
    print(f"   Generando embeddings para {len(products)} productos...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = []
    for p in products:
        combined = f"{p['name']}. {p['brand']}. {p['category']} - {p['subcategory']}. {p['description']}. Tags: {', '.join(p['tags'])}"
        texts.append(combined)

    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def main():
    print("Generando base de datos de productos...")
    products = generate_products()
    print(f"   OK: {len(products)} productos generados")

    from collections import Counter
    cat_counts = Counter(p["category"] for p in products)
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"      {SUBCATEGORIES[cat]['icon']} {cat}: {count}")

    total_relations = sum(len(p["related_ids"]) for p in products)
    avg_relations = total_relations / len(products)
    print(f"\n   Relaciones totales: {total_relations} (promedio {avg_relations:.1f} por producto)")

    embeddings = generate_embeddings(products)

    data_path = os.path.join(DATA_DIR, "product_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({
            "products": products,
            "embeddings": embeddings,
        }, f)

    # Save a JSON preview
    preview = []
    for p in products[:5]:
        preview.append({
            "id": p["id"],
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "price": p["price"],
            "description": p["description"][:80],
            "related_count": len(p["related_ids"]),
        })
    preview_path = os.path.join(DATA_DIR, "product_preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)

    print(f"\nDatos guardados en {data_path}")
    print(f"Preview guardado en {preview_path}")
    print("Hecho!")


if __name__ == "__main__":
    main()
