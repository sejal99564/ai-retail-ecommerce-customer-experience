"""Generate clearly-labelled synthetic demo data for the AI Retail and E-commerce system.
Seeds both relational SQLite database (retail_ecommerce.db) and CSV exports.
Includes:
- Products catalog across multiple retail categories
- Customer profiles with shopping preferences and loyalty tiers
- Customer orders with status tracking for Conversational AI lookup
- Transaction history for collaborative filtering and analytics
- Behavioral interaction logs
- Structured ML dataset for purchase intent classification
"""
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "retail_ecommerce.db"


def generate_all_data(random_seed: int = 42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    print("Generating synthetic retail e-commerce dataset...")

    # 1. Product Catalog (40 distinct realistic products across 6 major retail categories)
    raw_products = [
        # Electronics
        ("P001", "Aura Wireless Noise-Cancelling Headphones", "Electronics", "SoundAura", 6999.0, 4.7, 412, 38,
         "Over-ear premium headphones with 40-hour battery life and active noise cancellation.", "audio, bluetooth, headphones, wireless, travel", 0),
        ("P002", "BassPulse Waterproof Bluetooth Speaker", "Electronics", "PulseWave", 2499.0, 4.4, 289, 45,
         "Rugged IPX7 waterproof portable speaker with deep bass and 12-hour playtime.", "audio, speaker, portable, bluetooth, outdoor", 0),
        ("P003", "Apex Chrono Pro Smartwatch", "Electronics", "ApexTech", 4999.0, 4.6, 620, 24,
         "AMOLED display smartwatch with heart rate monitoring, SpO2, GPS, and sleep tracking.", "smartwatch, fitness, wearable, health, tech", 0),
        ("P004", "NovaTech Mechanical Gaming Keyboard", "Electronics", "NovaTech", 3499.0, 4.8, 310, 18,
         "RGB mechanical keyboard with custom blue switches and anti-ghosting keys.", "gaming, keyboard, rgb, accessories, pc", 0),
        ("P005", "SilentClick Ergonomic Wireless Mouse", "Electronics", "NovaTech", 1299.0, 4.3, 195, 60,
         "Dual-mode 2.4GHz and Bluetooth wireless ergonomic optical mouse with quiet click.", "mouse, wireless, office, accessories", 0),
        ("P006", "UltraView 27-inch 4K UHD Monitor", "Electronics", "ViewMax", 18999.0, 4.7, 142, 12,
         "Frameless 4K IPS display with 99% sRGB color accuracy, HDR10, and HDMI/Type-C ports.", "monitor, display, 4k, office, pc", 0),
        ("P007", "HyperDrive 7-in-1 USB-C Multiport Hub", "Electronics", "DriveGear", 1899.0, 4.5, 380, 55,
         "Aluminum hub with 4K HDMI, 100W Power Delivery, SD card reader, and USB 3.0 ports.", "hub, usb-c, adapter, laptop, accessories", 0),
        ("P008", "PowerVolt 20000mAh Fast Charging Power Bank", "Electronics", "PowerVolt", 1699.0, 4.6, 850, 4,
         "Compact power bank supporting 22.5W PD fast charging with dual USB and Type-C outputs.", "power bank, battery, charging, portable, travel", 0),
        ("P009", "ClearCast 1080p Full HD Pro Webcam", "Electronics", "ViewMax", 2299.0, 4.2, 160, 28,
         "High-definition video webcam with stereo noise-cancelling microphones and privacy shutter.", "webcam, streaming, video call, remote work", 0),
        ("P010", "SwiftData 1TB Portable External SSD", "Electronics", "DriveGear", 7499.0, 4.9, 520, 15,
         "Rugged NVMe external solid-state drive with ultra-fast transfer speeds up to 1050MB/s.", "storage, ssd, external, backup, high-speed", 0),

        # Fashion & Apparel
        ("P011", "AeroFly Men's Breathable Running Shoes", "Fashion", "AeroAthletics", 2999.0, 4.5, 340, 40,
         "Lightweight mesh cushioning running shoes engineered for shock absorption and comfort.", "shoes, running, sports, footwear, men", 0),
        ("P012", "UrbanGlide Classic Canvas Sneakers", "Fashion", "UrbanStyle", 1799.0, 4.3, 215, 32,
         "Timeless low-top canvas casual shoes with vulcanized rubber sole and soft insole.", "sneakers, canvas, casual, unisex, footwear", 0),
        ("P013", "PolarShield Thermal Winter Jacket", "Fashion", "NordicWear", 4499.0, 4.7, 180, 8,
         "Windproof and water-resistant insulated parka jacket with detachable faux fur hood.", "jacket, winter, coat, warm, outdoor", 0),
        ("P014", "EcoPure Organic Cotton Crewneck T-Shirt (Pack of 3)", "Fashion", "EcoPure", 1199.0, 4.4, 450, 75,
         "100% combed organic cotton pre-shrunk soft breathable basic everyday t-shirts.", "tshirt, cotton, casual, apparel, basics", 0),
        ("P015", "ComfortFlex Slim Fit Stretch Denim Jeans", "Fashion", "UrbanStyle", 1999.0, 4.2, 310, 26,
         "Modern classic slim-tapered durable stretch denim jeans for all-day flexible comfort.", "jeans, denim, pants, casual, fashion", 0),
        ("P016", "Horizon UV400 Polarized Aviator Sunglasses", "Fashion", "SunOptics", 1499.0, 4.6, 275, 50,
         "Ultra-light stainless steel aviators with scratch-resistant polarized glare reduction.", "sunglasses, eyewear, polarized, accessories, summer", 0),
        ("P017", "All-Weather Commuter Laptop Backpack (30L)", "Fashion", "UrbanStyle", 2499.0, 4.7, 530, 22,
         "Water-repellent anti-theft backpack with padded 15.6-inch laptop compartment and USB port.", "backpack, bag, travel, commuter, accessories", 0),
        ("P018", "Heritage Genuine Leather Bifold Wallet", "Fashion", "NordicWear", 1299.0, 4.5, 390, 65,
         "Handcrafted top-grain leather minimalist wallet with RFID blocking technology.", "wallet, leather, accessories, rfid, gifts", 0),

        # Home & Kitchen
        ("P019", "AromaBrew Programmable Drip Coffee Maker", "Home & Kitchen", "BrewCraft", 3999.0, 4.6, 410, 16,
         "12-cup digital coffee maker with reusable filter, 24-hour timer, and auto-shutoff.", "coffee maker, kitchen, appliances, brew, breakfast", 0),
        ("P020", "CrispAir 4.5L Digital Air Fryer", "Home & Kitchen", "KitchenPro", 5499.0, 4.8, 670, 11,
         "Rapid air circulation oil-free fryer with 8 one-touch cooking presets and LED display.", "air fryer, healthy cooking, appliances, kitchen", 0),
        ("P021", "ThermoShield 1000ml Insulated Stainless Steel Bottle", "Home & Kitchen", "HydroLife", 899.0, 4.7, 920, 95,
         "Double-wall vacuum insulated flask keeping drinks cold for 24h or hot for 12h.", "water bottle, insulated, flask, eco, hydration", 1),
        ("P022", "LumiDesk Smart LED Architect Desk Lamp", "Home & Kitchen", "LumiLife", 1899.0, 4.4, 210, 34,
         "Eye-caring dimmable desk lamp with 5 color temperatures, wireless phone charging base.", "lamp, led, lighting, desk, study, home office", 0),
        ("P023", "ErgoRest High-Back Mesh Ergonomic Office Chair", "Home & Kitchen", "ErgoComfort", 8999.0, 4.5, 330, 7,
         "Adjustable lumbar support chair with 3D armrests, breathable mesh, and reclining tilt.", "chair, ergonomic, office, furniture, posture", 0),
        ("P024", "PureBreeze HEPA Room Air Purifier", "Home & Kitchen", "PureBreeze", 6499.0, 4.7, 280, 14,
         "H13 True HEPA filter capturing 99.97% of airborne allergens, smoke, dust, and odors.", "air purifier, hepa, health, clean air, home", 1),
        ("P025", "GlowSmart 16 Million Color Smart WiFi LED Bulb", "Home & Kitchen", "LumiLife", 699.0, 4.3, 510, 80,
         "Voice-controlled smart bulb compatible with Alexa and Google Assistant without a hub.", "smart bulb, lighting, smart home, wifi, rgb", 0),
        ("P026", "ChefMaster 3-Piece Non-Stick Granite Cookware Set", "Home & Kitchen", "KitchenPro", 2799.0, 4.5, 390, 25,
         "PFOA-free granite induction bottom frying pan, saucepan, and kadai with glass lids.", "cookware, kitchen, pots, pans, non-stick", 0),

        # Fitness & Sports
        ("P027", "ZenFlow Extra-Thick 8mm Non-Slip Yoga Mat", "Fitness & Sports", "ZenFlow", 1299.0, 4.6, 440, 42,
         "Eco-friendly high-density TPE yoga exercise mat with alignment marks and carry strap.", "yoga mat, fitness, exercise, pilates, workout", 0),
        ("P028", "PowerFlex 5-Level Heavy Duty Resistance Bands Set", "Fitness & Sports", "ZenFlow", 799.0, 4.7, 610, 90,
         "Natural latex loop bands for strength training, physical therapy, and home workouts.", "resistance bands, gym, workout, fitness, muscle", 0),
        ("P029", "IronGrip Adjustable Dumbbells Pair (2.5kg - 24kg)", "Fitness & Sports", "TitanGym", 12999.0, 4.8, 175, 5,
         "Space-saving dial mechanism adjustable weights for progressive strength conditioning.", "dumbbells, weights, gym, strength, bodybuilding", 0),
        ("P030", "HydroSpeed Anti-Fog UV Protection Swimming Goggles", "Fitness & Sports", "AquaSpeed", 699.0, 4.2, 180, 58,
         "Wide-view leakproof silicone swim goggles with interchangeable nose bridges.", "swimming, goggles, watersports, pool, fitness", 0),
        ("P031", "SpeedJump High-Speed Aluminum Bearing Skipping Rope", "Fitness & Sports", "TitanGym", 499.0, 4.5, 340, 85,
         "Tangle-free steel wire jump rope for cardio, CrossFit, and endurance conditioning.", "jump rope, cardio, skipping, crossfit, speed", 0),

        # Beauty & Personal Care
        ("P032", "GlowRevive Vitamin C & Hyaluronic Acid Face Serum (30ml)", "Beauty & Personal Care", "DermaCare", 749.0, 4.6, 730, 68,
         "Dermatologically tested brightening facial serum reducing pigmentation and fine lines.", "serum, skincare, vitamin c, beauty, facial", 0),
        ("P033", "SonicClean Ultrasonic Rechargeable Electric Toothbrush", "Beauty & Personal Care", "DentSmile", 1899.0, 4.7, 490, 30,
         "40,000 VPM motor sonic toothbrush with 5 cleaning modes and 4 replacement brush heads.", "toothbrush, dental care, electric, health, hygiene", 1),
        ("P034", "ProGlide 2000W Ionic Ceramic Hair Dryer", "Beauty & Personal Care", "SalonPro", 2199.0, 4.4, 260, 22,
         "Fast-drying salon hair dryer with ionic conditioning for frizz-free shiny hair.", "hair dryer, styling, beauty, salon, grooming", 0),
        ("P035", "Titanium Precision Beard & Body Hair Trimmer", "Beauty & Personal Care", "GroomMaster", 1499.0, 4.5, 580, 37,
         "Cordless trimmer with self-sharpening stainless steel blades and 90-minute runtime.", "trimmer, grooming, beard, shaver, men", 0),

        # Books & Stationery
        ("P036", "Artisan Hardcover Dotted Journal Notebook (160 GSM)", "Books & Stationery", "PaperCraft", 599.0, 4.8, 420, 70,
         "Lay-flat bullet journal with bleed-resistant thick bamboo paper and dual bookmarks.", "notebook, journal, stationery, planner, writing", 0),
        ("P037", "Fineline Archival Waterproof Pigment Micro Pens (Set of 8)", "Books & Stationery", "PaperCraft", 699.0, 4.7, 310, 50,
         "Assorted needle point black drawing illustration pens for sketching and calligraphy.", "pens, sketching, drawing, stationery, art", 0),
        ("P038", "Aluminum Foldable Portable Laptop & Book Stand", "Books & Stationery", "ErgoComfort", 1099.0, 4.6, 680, 48,
         "Ventilated 6-level adjustable ergonomic riser for laptops, tablets, and textbooks.", "laptop stand, reading, desk, ergonomic, portable", 0),
        ("P039", "Designing Data-Intensive Applications by Martin Kleppmann", "Books & Stationery", "TechPress", 1899.0, 4.9, 890, 20,
         "Definitive engineering guide to distributed systems, scalability, and data architectures.", "book, software, engineering, architecture, learning", 0),
        ("P040", "Atomic Habits: Proven Way to Build Good Habits by James Clear", "Books & Stationery", "MindPress", 499.0, 4.8, 1450, 110,
         "Global bestselling self-help guide on transforming daily systems and behavior.", "book, self-help, habits, psychology, productivity", 0),
    ]

    products_df = pd.DataFrame(raw_products, columns=[
        "product_id", "name", "category", "brand", "base_price", "rating",
        "reviews_count", "stock", "description", "tags", "is_essential"
    ])
    products_df["current_price"] = products_df["base_price"]

    # 2. Customers Dataset (200 realistic customer profiles)
    first_names = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Muhammad", "Sai", "Ayaan", "Krishna",
        "Ishaan", "Shaurya", "Atharva", "Kabir", "Dhruv", "Ananya", "Diya", "Saanvi", "Aadhya", "Pari",
        "Kiara", "Myra", "Riya", "Anushka", "Fatima", "Zara", "Navya", "Meera", "Rohan", "Pooja",
        "Vikram", "Sneha", "Karan", "Kavita", "Siddharth", "Neha", "Rahul", "Priyanka", "Amit", "Tanvi"
    ]
    last_names = [
        "Sharma", "Verma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Mehta", "Joshi", "Chopra",
        "Nair", "Iyer", "Rao", "Das", "Bose", "Chatterjee", "Kapoor", "Malhotra", "Khan", "Deshmukh"
    ]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    tier_weights = [0.45, 0.30, 0.18, 0.07]
    categories = sorted(products_df["category"].unique())

    customers_list = []
    base_date = datetime(2023, 1, 1)

    for i in range(1, 251):
        cid = f"C{i:04d}"
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        name = f"{fname} {lname}"
        email = f"{fname.lower()}.{lname.lower()}{i}@example.com"
        pref_cat = random.choice(categories)
        loyalty = np.random.choice(tiers, p=tier_weights)

        # Budget range conditioned on loyalty tier
        if loyalty == "Platinum":
            budget = float(rng.integers(12000, 45000))
        elif loyalty == "Gold":
            budget = float(rng.integers(7000, 25000))
        elif loyalty == "Silver":
            budget = float(rng.integers(3000, 15000))
        else:
            budget = float(rng.integers(800, 8000))

        join_days_ago = rng.integers(30, 800)
        join_date = (datetime.now() - timedelta(days=int(join_days_ago))).strftime("%Y-%m-%d")

        customers_list.append({
            "customer_id": cid,
            "name": name,
            "email": email,
            "preferred_category": pref_cat,
            "budget": budget,
            "loyalty_tier": loyalty,
            "join_date": join_date
        })

    customers_df = pd.DataFrame(customers_list)

    # 3. Order History Dataset (Realistic trackable orders for Conversational AI)
    order_statuses = ["Delivered", "Shipped", "Out for Delivery", "Processing"]
    status_weights = [0.55, 0.22, 0.13, 0.10]
    cities = ["Mumbai", "Bengaluru", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Chandigarh"]

    orders_list = []
    for oid_num in range(1001, 1151):
        order_id = f"ORD-{oid_num}"
        cust = customers_df.sample(1).iloc[0]
        prod = products_df.sample(1).iloc[0]
        days_ago = rng.integers(1, 45)
        odate = (datetime.now() - timedelta(days=int(days_ago))).strftime("%Y-%m-%d")
        status = np.random.choice(order_statuses, p=status_weights)
        tracking_num = f"TRK-{rng.integers(10000000, 99999999)}"
        est_days = 2 if status == "Out for Delivery" else (4 if status == "Shipped" else 0)
        est_delivery = (datetime.now() + timedelta(days=est_days)).strftime("%Y-%m-%d") if status != "Delivered" else odate
        city = random.choice(cities)
        addr = f"Flat {rng.integers(101, 909)}, Sector {rng.integers(1, 45)}, {city}"

        orders_list.append({
            "order_id": order_id,
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "product_name": prod["name"],
            "order_date": odate,
            "status": status,
            "total_amount": float(prod["base_price"]),
            "tracking_number": tracking_num,
            "estimated_delivery": est_delivery,
            "shipping_address": addr
        })

    orders_df = pd.DataFrame(orders_list)

    # 4. Historical Transactions (for Collaborative Filtering & Analytics)
    payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]
    trans_list = []
    for t_idx in range(1, 801):
        tid = f"TXN-{t_idx:05d}"
        cust = customers_df.sample(1).iloc[0]
        # High probability to purchase in preferred category
        if rng.random() < 0.65:
            cat_prods = products_df[products_df["category"] == cust["preferred_category"]]
            prod = cat_prods.sample(1).iloc[0] if not cat_prods.empty else products_df.sample(1).iloc[0]
        else:
            prod = products_df.sample(1).iloc[0]

        qty = int(rng.choice([1, 1, 1, 2, 3]))
        unit_price = float(prod["base_price"])
        total_amt = round(unit_price * qty, 2)
        days_ago = rng.integers(1, 300)
        pdate = (datetime.now() - timedelta(days=int(days_ago))).strftime("%Y-%m-%d")
        pay_method = random.choice(payment_methods)
        sat_rating = round(float(rng.uniform(3.5, 5.0)), 1)

        trans_list.append({
            "transaction_id": tid,
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "purchase_date": pdate,
            "quantity": qty,
            "unit_price": unit_price,
            "total_amount": total_amt,
            "payment_method": pay_method,
            "satisfaction_rating": sat_rating
        })

    transactions_df = pd.DataFrame(trans_list)

    # 5. Customer Interactions (Browsing views, clicks, cart adds)
    interaction_types = ["view", "cart_add", "wishlist", "search_click"]
    type_weights = [0.55, 0.25, 0.10, 0.10]
    interactions_list = []
    for i_idx in range(1, 1501):
        iid = f"INT-{i_idx:05d}"
        cust = customers_df.sample(1).iloc[0]
        prod = products_df.sample(1).iloc[0]
        itype = np.random.choice(interaction_types, p=type_weights)
        sess_id = f"SESS-{rng.integers(1000, 9999)}"
        dur = int(rng.integers(15, 480))
        days_ago = rng.integers(0, 60)
        tstamp = (datetime.now() - timedelta(days=int(days_ago), minutes=int(rng.integers(1, 1440)))).strftime("%Y-%m-%d %H:%M:%S")

        interactions_list.append({
            "interaction_id": iid,
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "interaction_type": itype,
            "session_id": sess_id,
            "duration_seconds": dur,
            "timestamp": tstamp
        })

    interactions_df = pd.DataFrame(interactions_list)

    # 6. Synthetic Retail Machine Learning Dataset (for Purchase Intent Classification)
    # Reflects Figure 2 of the Research Paper: Customer Behavior + Product Features + Interactions
    n_samples = 3200
    ml_records = []

    for _ in range(n_samples):
        # Behavioral & session features
        session_duration = float(rng.gamma(shape=2.5, scale=75))  # in seconds, ~180s avg
        pages_viewed = int(max(1, rng.poisson(lam=4.2)))
        cart_additions = int(rng.choice([0, 0, 1, 1, 2, 3], p=[0.40, 0.25, 0.20, 0.08, 0.05, 0.02]))
        past_purchases = int(rng.poisson(lam=2.8))
        avg_historical_spend = float(rng.uniform(500, 18000))
        product_price = float(rng.uniform(499, 15000))
        discount_offered = float(rng.choice([0, 5, 10, 15, 20, 25, 30], p=[0.3, 0.2, 0.2, 0.15, 0.08, 0.05, 0.02]))
        product_rating = float(np.round(rng.uniform(3.0, 5.0), 1))
        stock_ratio = float(np.round(rng.uniform(0.05, 1.0), 2))
        category_affinity = float(np.round(rng.beta(2, 2), 2))  # 0.0 to 1.0
        price_sensitivity = float(np.round(rng.uniform(0.1, 1.0), 2))
        is_loyalty = int(rng.choice([0, 1], p=[0.6, 0.4]))

        # Calculate ground truth probability of purchase conversion based on economic logic
        # Positive drivers: high cart additions, high category affinity, loyalty status, high rating, high pages viewed
        # Negative drivers: high price sensitivity on expensive items, zero cart additions
        z = (
            -2.4
            + 1.65 * cart_additions
            + 1.10 * category_affinity
            + 0.55 * is_loyalty
            + 0.003 * min(session_duration, 600)
            + 0.15 * min(pages_viewed, 10)
            + 0.40 * (product_rating - 3.5)
            + 0.025 * discount_offered
            + 0.00003 * avg_historical_spend
            - 0.00008 * product_price
            - 0.90 * price_sensitivity * (product_price / 10000.0)
            + 0.20 * (1.0 if stock_ratio < 0.2 else 0.0)  # urgency effect
            + rng.normal(0, 0.45)  # realistic stochastic human noise
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        purchase_intent = 1 if rng.random() < prob else 0

        ml_records.append({
            "session_duration_secs": round(session_duration, 1),
            "pages_viewed": pages_viewed,
            "cart_additions": cart_additions,
            "past_purchases_count": past_purchases,
            "avg_historical_spend": round(avg_historical_spend, 2),
            "product_price": round(product_price, 2),
            "discount_offered_pct": round(discount_offered, 1),
            "product_rating": product_rating,
            "stock_level_ratio": stock_ratio,
            "category_affinity_score": category_affinity,
            "price_sensitivity_score": price_sensitivity,
            "is_loyalty_member": is_loyalty,
            "purchase_intent": purchase_intent
        })

    ml_df = pd.DataFrame(ml_records)

    # Save CSVs
    products_df.to_csv(DATA_DIR / "products.csv", index=False)
    customers_df.to_csv(DATA_DIR / "customers.csv", index=False)
    orders_df.to_csv(DATA_DIR / "orders.csv", index=False)
    transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)
    ml_df.to_csv(DATA_DIR / "synthetic_model_data.csv", index=False)

    # Seed SQLite Database
    from src.database import init_db, get_db_connection
    init_db(DB_PATH)
    conn = get_db_connection(DB_PATH)

    # Insert into SQLite
    products_df.to_sql("products", conn, if_exists="replace", index=False)
    customers_df.to_sql("customers", conn, if_exists="replace", index=False)
    orders_df.to_sql("orders", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
    interactions_df.to_sql("interactions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"Successfully generated:")
    print(f"  - Products: {len(products_df)} records across {len(categories)} categories")
    print(f"  - Customers: {len(customers_df)} records")
    print(f"  - Orders: {len(orders_df)} records")
    print(f"  - Transactions: {len(transactions_df)} records")
    print(f"  - Interactions: {len(interactions_df)} records")
    print(f"  - ML Dataset: {len(ml_df)} records (Target: purchase_intent, Pos Class: {ml_df['purchase_intent'].mean()*100:.1f}%)")
    print(f"  - Seeded SQLite database at: {DB_PATH}")
    print("NOTE: All data is clearly labelled as SYNTHETIC DEMONSTRATION DATA.")


if __name__ == "__main__":
    generate_all_data()
