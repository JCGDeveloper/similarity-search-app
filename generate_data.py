"""
Generate sample product data and precompute embeddings for the Similarity Search App.
"""

import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Product Database ────────────────────────────────────────────────────────

PRODUCTS = [
    # ── Alimentación / Snacks ──
    {"id": "food_001", "name": "Patatas Fritas Clásicas", "brand": "Lays", "category": "Alimentación",
     "description": "Patatas fritas de bolsa, sabor clásico, ligeramente saladas. Crujientes y finas.",
     "price": "2.50€", "image": None},
    {"id": "food_002", "name": "Patatas Fritas Onduladas", "brand": "Ruffles", "category": "Alimentación",
     "description": "Patatas fritas onduladas de textura gruesa, extra crujientes con sal marina.",
     "price": "2.80€", "image": None},
    {"id": "food_003", "name": "Patatas Fritas al Horno", "brand": "Pringles", "category": "Alimentación",
     "description": "Patatas fritas horneadas en formato tubular, textura uniforme y crujiente.",
     "price": "3.20€", "image": None},
    {"id": "food_004", "name": "Patatas Fritas BBQ", "brand": "Mackays", "category": "Alimentación",
     "description": "Patatas fritas con sabor a barbacoa ahumada, intensas y sabrosas.",
     "price": "2.90€", "image": None},
    {"id": "food_005", "name": "Patatas Fritas Agrias", "brand": "Herr's", "category": "Alimentación",
     "description": "Patatas fritas con vinagreta y sal, sabor ácido y refrescante.",
     "price": "2.70€", "image": None},
    {"id": "food_006", "name": "Gusanitos de Maíz", "brand": "Pantera", "category": "Alimentación",
     "description": "Snacks de maíz inflado con sabor a queso, ligeros y crujientes.",
     "price": "1.80€", "image": None},
    {"id": "food_007", "name": "Nachos con Queso", "brand": "Doritos", "category": "Alimentación",
     "description": "Totopos de maíz sabor queso cheddar, ideales para picar o con salsa.",
     "price": "3.00€", "image": None},
    {"id": "food_008", "name": "Palomitas de Maíz Mantequilla", "brand": "Act II", "category": "Alimentación",
     "description": "Palomitas para microondas sabor mantequilla, grano grande y esponjoso.",
     "price": "2.20€", "image": None},
    {"id": "food_009", "name": "Frutos Secos Variados", "brand": "Borges", "category": "Alimentación",
     "description": "Mezcla de almendras, nueces, anacardos y avellanas tostadas sin sal.",
     "price": "5.50€", "image": None},
    {"id": "food_010", "name": "Barrita de Cereal Chocolate", "brand": "Nature Valley", "category": "Alimentación",
     "description": "Barrita de cereales integrales con trozos de chocolate y miel.",
     "price": "1.50€", "image": None},
    {"id": "food_011", "name": "Galletas Digestive", "brand": "MC Vities", "category": "Alimentación",
     "description": "Galletas de trigo integral ligeramente dulces, clásicas para desayuno.",
     "price": "2.10€", "image": None},
    {"id": "food_012", "name": "Galletas Oreo", "brand": "Oreo", "category": "Alimentación",
     "description": "Galletas sándwich de chocolate con crema de vainilla, crujientes.",
     "price": "2.30€", "image": None},
    {"id": "food_013", "name": "Chocolate con Leche", "brand": "Milka", "category": "Alimentación",
     "description": "Tableta de chocolate con leche, suave y cremoso, 100g.",
     "price": "1.90€", "image": None},
    {"id": "food_014", "name": "Chocolate Negro 70%", "brand": "Lindt", "category": "Alimentación",
     "description": "Chocolate negro intenso con 70% cacao, sabor profundo y elegante.",
     "price": "3.50€", "image": None},
    {"id": "food_015", "name": "Coca-Cola Zero", "brand": "Coca-Cola", "category": "Alimentación",
     "description": "Refresco de cola sin azúcar, edulcorada, gasificada y refrescante.",
     "price": "1.20€", "image": None},
    {"id": "food_016", "name": "Zumo de Naranja Natural", "brand": "Don Simón", "category": "Alimentación",
     "description": "Zumo de naranja exprimida sin azúcares añadidos, envase brick 1L.",
     "price": "1.80€", "image": None},

    # ── Libros ──
    {"id": "book_001", "name": "Cien Años de Soledad", "brand": "Gabriel García Márquez", "category": "Libros",
     "description": "Novela icónica del realismo mágico sobre la familia Buendía en Macondo.",
     "price": "12.95€", "image": None},
    {"id": "book_002", "name": "El Amor en los Tiempos del Cólera", "brand": "Gabriel García Márquez", "category": "Libros",
     "description": "Historia de amor eterno entre Florentino Ariza y Fermina Daza a través de décadas.",
     "price": "11.50€", "image": None},
    {"id": "book_003", "name": "1984", "brand": "George Orwell", "category": "Libros",
     "description": "Distopía clásica sobre un régimen totalitario donde el Gran Hermano todo lo vigila.",
     "price": "10.95€", "image": None},
    {"id": "book_004", "name": "Rebelión en la Granja", "brand": "George Orwell", "category": "Libros",
     "description": "Sátira política donde los animales se rebelan contra sus dueños humanos.",
     "price": "8.95€", "image": None},
    {"id": "book_005", "name": "El Principito", "brand": "Antoine de Saint-Exupéry", "category": "Libros",
     "description": "Fábula poética sobre un niño de otro planeta que enseña lecciones de vida.",
     "price": "7.50€", "image": None},
    {"id": "book_006", "name": "Don Quijote de la Mancha", "brand": "Miguel de Cervantes", "category": "Libros",
     "description": "Clásico de la literatura universal sobre un hidalgo que enloquece leyendo novelas de caballería.",
     "price": "14.00€", "image": None},
    {"id": "book_007", "name": "La Sombra del Viento", "brand": "Carlos Ruiz Zafón", "category": "Libros",
     "description": "Misterio literario ambientado en la Barcelona de posguerra sobre un libro olvidado.",
     "price": "11.95€", "image": None},
    {"id": "book_008", "name": "Harry Potter y la Piedra Filosofal", "brand": "J.K. Rowling", "category": "Libros",
     "description": "Primera entrega de la saga del joven mago que descubre Hogwarts y su destino.",
     "price": "12.50€", "image": None},
    {"id": "book_009", "name": "El Señor de los Anillos: La Comunidad", "brand": "J.R.R. Tolkien", "category": "Libros",
     "description": "Primera parte de la épica fantasía sobre un anillo único y la lucha del bien contra el mal.",
     "price": "15.00€", "image": None},
    {"id": "book_010", "name": "El Hobbit", "brand": "J.R.R. Tolkien", "category": "Libros",
     "description": "Aventura de un hobbit llamado Bilbo que emprende un viaje con enanos y un dragón.",
     "price": "11.00€", "image": None},
    {"id": "book_011", "name": "La Casa de los Espíritus", "brand": "Isabel Allende", "category": "Libros",
     "description": "Saga familiar chilena que mezcla realismo mágico, política y pasión generacional.",
     "price": "13.50€", "image": None},
    {"id": "book_012", "name": "Los Juegos del Hambre", "brand": "Suzanne Collins", "category": "Libros",
     "description": "Distopía juvenil donde adolescentes luchan a muerte en un reality show televisado.",
     "price": "10.50€", "image": None},
    {"id": "book_013", "name": "El Código Da Vinci", "brand": "Dan Brown", "category": "Libros",
     "description": "Thriller de misterio y conspiración sobre simbología oculta en obras de arte.",
     "price": "11.95€", "image": None},
    {"id": "book_014", "name": "Sapiens: De Animales a Dioses", "brand": "Yuval Noah Harari", "category": "Libros",
     "description": "Ensayo histórico sobre la evolución de la humanidad desde la prehistoria hasta hoy.",
     "price": "13.90€", "image": None},
    {"id": "book_015", "name": "El Alquimista", "brand": "Paulo Coelho", "category": "Libros",
     "description": "Novela inspiracional sobre un pastor que sigue su sueño hasta las pirámides de Egipto.",
     "price": "9.95€", "image": None},

    # ── Electrónica ──
    {"id": "elec_001", "name": "Auriculares Bluetooth", "brand": "Sony WH-1000XM5", "category": "Electrónica",
     "description": "Auriculares inalámbricos con cancelación de ruido activa, 30h batería, plegables.",
     "price": "349€", "image": None},
    {"id": "elec_002", "name": "Auriculares Inalámbricos", "brand": "Bose QuietComfort", "category": "Electrónica",
     "description": "Cascos bluetooth con cancelación de ruido premium, sonido envolvente y micrófono.",
     "price": "329€", "image": None},
    {"id": "elec_003", "name": "Auriculares Deportivos", "brand": "Jabra Elite 8", "category": "Electrónica",
     "description": "Earbuds inalámbricos resistentes al agua, ideales para deporte y running.",
     "price": "199€", "image": None},
    {"id": "elec_004", "name": "Smartphone Android", "brand": "Samsung Galaxy S25", "category": "Electrónica",
     "description": "Móvil Android de gama alta con pantalla AMOLED, 256GB, triple cámara 200MP.",
     "price": "1,299€", "image": None},
    {"id": "elec_005", "name": "iPhone 16 Pro", "brand": "Apple", "category": "Electrónica",
     "description": "Smartphone Apple con chip A18 Pro, pantalla OLED 6.3\", cámara de 48MP.",
     "price": "1,219€", "image": None},
    {"id": "elec_006", "name": "Google Pixel 9", "brand": "Google", "category": "Electrónica",
     "description": "Móvil con cámara computacional avanzada, Android puro y asistente AI integrado.",
     "price": "899€", "image": None},
    {"id": "elec_007", "name": "Portátil Ultrabook", "brand": "MacBook Air M4", "category": "Electrónica",
     "description": "Portátil ligero con chip Apple M4, 16GB RAM, SSD 512GB, pantalla Liquid Retina.",
     "price": "1,449€", "image": None},
    {"id": "elec_008", "name": "Portátil Gaming", "brand": "ASUS ROG Zephyrus", "category": "Electrónica",
     "description": "Portátil gaming con RTX 4070, 32GB RAM, pantalla 165Hz, teclado RGB.",
     "price": "2,199€", "image": None},
    {"id": "elec_009", "name": "Tablet 10.9\"", "brand": "iPad Air M3", "category": "Electrónica",
     "description": "Tablet con chip M3, pantalla Liquid Retina 10.9\", compatible con Apple Pencil.",
     "price": "749€", "image": None},
    {"id": "elec_010", "name": "Tablet Android", "brand": "Samsung Galaxy Tab S10", "category": "Electrónica",
     "description": "Tablet con pantalla Dynamic AMOLED 11\", S Pen incluido, ideal para creatividad.",
     "price": "899€", "image": None},
    {"id": "elec_011", "name": "Smartwatch Deportivo", "brand": "Apple Watch Ultra 3", "category": "Electrónica",
     "description": "Reloj inteligente resistente al agua, GPS, sensor de salud, batería 36h.",
     "price": "899€", "image": None},
    {"id": "elec_012", "name": "Altavoz Bluetooth Portátil", "brand": "JBL Flip 7", "category": "Electrónica",
     "description": "Altavoz inalámbrico resistente al agua, sonido potente, 16h batería.",
     "price": "149€", "image": None},
    {"id": "elec_013", "name": "Cámara Mirrorless", "brand": "Sony A7 V", "category": "Electrónica",
     "description": "Cámara sin espejo full-frame, 33MP, grabación 4K, estabilización 5 ejes.",
     "price": "2,499€", "image": None},
    {"id": "elec_014", "name": "Monitor 27\" 4K", "brand": "Dell UltraSharp", "category": "Electrónica",
     "description": "Monitor IPS 27 pulgadas 4K UHD, precisión de color para diseño y edición.",
     "price": "699€", "image": None},

    # ── Hogar ──
    {"id": "home_001", "name": "Cafetera Superautomática", "brand": "De'Longhi Dinamica", "category": "Hogar",
     "description": "Cafetera con molinillo integrado, prepara espresso, cappuccino y latte con un botón.",
     "price": "599€", "image": None},
    {"id": "home_002", "name": "Cafetera de Cápsulas", "brand": "Nespresso Vertuo", "category": "Hogar",
     "description": "Cafetera de cápsulas con sistema centrífugo que extrae crema espesa.",
     "price": "129€", "image": None},
    {"id": "home_003", "name": "Robot Aspirador", "brand": "Roomba i7+", "category": "Hogar",
     "description": "Aspirador robótico con autovaciado, mapeo inteligente y control por app.",
     "price": "799€", "image": None},
    {"id": "home_004", "name": "Robot Aspirador y Fregasuelos", "brand": "Roborock Q5", "category": "Hogar",
     "description": "Robot 2 en 1 que aspira y friega suelos con navegación LiDAR y app.",
     "price": "449€", "image": None},
    {"id": "home_005", "name": "Silla Ergonómica Oficina", "brand": "Herman Miller Aeron", "category": "Hogar",
     "description": "Silla de oficina ergonómica con soporte lumbar ajustable y malla transpirable.",
     "price": "1,495€", "image": None},
    {"id": "home_006", "name": "Silla Gaming", "brand": "Secretlab Titan Evo", "category": "Hogar",
     "description": "Silla gaming ergonómica con reposabrazos 4D, cuero sintético premium.",
     "price": "549€", "image": None},
    {"id": "home_007", "name": "Lámpara Inteligente", "brand": "Philips Hue", "category": "Hogar",
     "description": "Bombilla LED inteligente con millones de colores, control por voz y app.",
     "price": "59€", "image": None},
    {"id": "home_008", "name": "Termostato Inteligente", "brand": "Netatmo", "category": "Hogar",
     "description": "Termostato wifi que aprende tus horarios y ahorra energía automáticamente.",
     "price": "129€", "image": None},
    {"id": "home_009", "name": "Set de Sartenes Antiadherentes", "brand": "Tefal Ingenio", "category": "Hogar",
     "description": "Juego de 3 sartenes con recubrimiento antiadherente y mangos desmontables.",
     "price": "89€", "image": None},
    {"id": "home_010", "name": "Cuchillos de Cocina Profesional", "brand": "Zwilling J.A. Henckels", "category": "Hogar",
     "description": "Set de 5 cuchillos de acero inoxidable forjado con bloque de madera.",
     "price": "199€", "image": None},
    {"id": "home_011", "name": "Purificador de Aire", "brand": "Dyson Purifier Hot+Cool", "category": "Hogar",
     "description": "Purificador con ventilador y calefactor, filtra alérgenos y partículas finas.",
     "price": "549€", "image": None},
    {"id": "home_012", "name": "Colchón Viscolástico", "brand": "Emma Original", "category": "Hogar",
     "description": "Colchón de espuma viscoelástica con 3 capas, transpirable y adaptable.",
     "price": "449€", "image": None},

    # ── Moda ──
    {"id": "fash_001", "name": "Zapatillas Running", "brand": "Nike Air Zoom", "category": "Moda",
     "description": "Zapatillas de running con amortiguación reactiva y parte superior transpirable.",
     "price": "149€", "image": None},
    {"id": "fash_002", "name": "Zapatillas Deportivas", "brand": "Adidas Ultraboost", "category": "Moda",
     "description": "Zapatillas con suela Boost ultraligera y tejido Primeknit adaptable.",
     "price": "179€", "image": None},
    {"id": "fash_003", "name": "Zapatillas Casual", "brand": "New Balance 574", "category": "Moda",
     "description": "Zapatillas clásicas de estilo retro con suela de goma y empeine de ante.",
     "price": "99€", "image": None},
    {"id": "fash_004", "name": "Chaqueta Impermeable", "brand": "The North Face", "category": "Moda",
     "description": "Chaqueta técnica impermeable y transpirable con capucha desmontable.",
     "price": "229€", "image": None},
    {"id": "fash_005", "name": "Mochila Urbana 25L", "brand": "Eastpak", "category": "Moda",
     "description": "Mochila resistente con compartimento para portátil, cremalleras reforzadas.",
     "price": "69€", "image": None},
    {"id": "fash_006", "name": "Reloj Deportivo", "brand": "Casio G-Shock", "category": "Moda",
     "description": "Reloj resistente a golpes, sumergible 200m, cronómetro y alarma.",
     "price": "129€", "image": None},
    {"id": "fash_007", "name": "Gafas de Sol Polarizadas", "brand": "Ray-Ban Aviator", "category": "Moda",
     "description": "Gafas de sol clásicas con lentes polarizadas y montura metálica dorada.",
     "price": "189€", "image": None},
    {"id": "fash_008", "name": "Bolso Bandolera", "brand": "Herschel", "category": "Moda",
     "description": "Bolso cruzado de lona con cierre de cordón, varios compartimentos interiores.",
     "price": "49€", "image": None},

    # ── Juguetes / Infantil ──
    {"id": "toy_001", "name": "Lego Star Wars Nave", "brand": "LEGO", "category": "Juguetes",
     "description": "Set de construcción del Halcón Milenario con 1300 piezas y 4 minifiguras.",
     "price": "149€", "image": None},
    {"id": "toy_002", "name": "Lego City Camión Bomberos", "brand": "LEGO", "category": "Juguetes",
     "description": "Set de bomberos con camión articulado, escaleras y figuras, 400 piezas.",
     "price": "59€", "image": None},
    {"id": "toy_003", "name": "Puzzle 1000 Piezas", "brand": "Ravensburger", "category": "Juguetes",
     "description": "Puzzle panorámico de 1000 piezas con imagen de paisaje natural.",
     "price": "24€", "image": None},
    {"id": "toy_004", "name": "Nerf Blaster", "brand": "Hasbro", "category": "Juguetes",
     "description": "Lanzador de dardos de espuma con mira telescópica y cargador de 12 dardos.",
     "price": "39€", "image": None},
    {"id": "toy_005", "name": "Muñeca Interactiva", "brand": "Barbie Dreamhouse", "category": "Juguetes",
     "description": "Casa de muñecas de 3 pisos con piscina, ascensor y muebles realistas.",
     "price": "179€", "image": None},
    {"id": "toy_006", "name": "Coche Teledirigido", "brand": "Hot Wheels RC", "category": "Juguetes",
     "description": "Coche radiocontrol con batería recargable, velocidad 20km/h y luces LED.",
     "price": "49€", "image": None},

    # ── Música / Entretenimiento ──
    {"id": "music_001", "name": "Guitarra Acústica", "brand": "Yamaha F310", "category": "Música",
     "description": "Guitarra acústica de iniciación con cuerdas de acero, cuerpo clásico y sonido cálido.",
     "price": "159€", "image": None},
    {"id": "music_002", "name": "Teclado Eléctrico 61 Teclas", "brand": "Casio CT-S1", "category": "Música",
     "description": "Teclado portátil con sonidos de piano, órgano y sintetizador, altavoces integrados.",
     "price": "199€", "image": None},
    {"id": "music_003", "name": "Vinilo Abbey Road", "brand": "The Beatles", "category": "Música",
     "description": "Disco de vinilo 180g del álbum clásico de The Beatles, edición remasterizada.",
     "price": "29€", "image": None},
    {"id": "music_004", "name": "Vinilo Thriller", "brand": "Michael Jackson", "category": "Música",
     "description": "Álbum vinilo 180g con el hit Thriller y otros clásicos del Rey del Pop.",
     "price": "27€", "image": None},
]


def generate_embeddings():
    """Generate text embeddings for all products and save to disk."""
    print(f"📦 Generating embeddings for {len(PRODUCTS)} products...")

    # Cargamos modelo ligero (all-MiniLM-L6-v2, ~80MB)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combinamos nombre, marca, categoría y descripción para una búsqueda más rica
    texts_for_embedding = []
    for p in PRODUCTS:
        combined = f"{p['name']}. {p['brand']}. {p['category']}. {p['description']}"
        texts_for_embedding.append(combined)

    # Generar embeddings de una sola vez (batch)
    embeddings = model.encode(texts_for_embedding, show_progress_bar=True)

    # Guardar
    data_path = os.path.join(DATA_DIR, 'product_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump({
            'products': PRODUCTS,
            'embeddings': embeddings,
        }, f)

    print(f"✅ Datos guardados en {data_path}")
    print(f"   Shape embeddings: {embeddings.shape}")
    print(f"   Categorías: {set(p['category'] for p in PRODUCTS)}")


if __name__ == "__main__":
    generate_embeddings()
