"""
====================================================
NusantaraAI - Dataset Generator
Studi Kasus: Sistem Rekomendasi Wisata Indonesia
====================================================
Jalankan: python generate_dataset.py
Output  : 4 file CSV di folder ./dataset/
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ─── Seed untuk reprodusibilitas ─────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

os.makedirs("dataset", exist_ok=True)

# ─── Data Referensi ───────────────────────────────────────────────────────────
PROVINSI = [
    "Bali", "Yogyakarta", "Jakarta", "Lombok", "Raja Ampat",
    "Labuan Bajo", "Bromo", "Toba", "Belitung", "Wakatobi"
]

KATEGORI_WISATA = [
    "Pantai", "Gunung", "Budaya", "Kuliner", "Alam",
    "Sejarah", "Petualangan", "Religi", "Bahari", "Edukasi"
]

DESTINASI = [
    # Bali
    {"nama": "Tanah Lot", "provinsi": "Bali", "kategori": "Budaya",
     "rating": 4.7, "harga_tiket": 60000, "durasi_jam": 2, "fasilitas": "wifi,parkir,toilet,restoran"},
    {"nama": "Kuta Beach", "provinsi": "Bali", "kategori": "Pantai",
     "rating": 4.5, "harga_tiket": 0, "durasi_jam": 4, "fasilitas": "wifi,parkir,toilet"},
    {"nama": "Ubud Monkey Forest", "provinsi": "Bali", "kategori": "Alam",
     "rating": 4.6, "harga_tiket": 80000, "durasi_jam": 2, "fasilitas": "parkir,toilet,guide"},
    {"nama": "Tegalalang Rice Terrace", "provinsi": "Bali", "kategori": "Alam",
     "rating": 4.4, "harga_tiket": 20000, "durasi_jam": 2, "fasilitas": "parkir,toilet,warung"},
    {"nama": "Seminyak Beach", "provinsi": "Bali", "kategori": "Pantai",
     "rating": 4.5, "harga_tiket": 0, "durasi_jam": 3, "fasilitas": "wifi,restoran,toilet"},

    # Yogyakarta
    {"nama": "Candi Borobudur", "provinsi": "Yogyakarta", "kategori": "Sejarah",
     "rating": 4.9, "harga_tiket": 350000, "durasi_jam": 3, "fasilitas": "wifi,parkir,toilet,restoran,guide"},
    {"nama": "Candi Prambanan", "provinsi": "Yogyakarta", "kategori": "Sejarah",
     "rating": 4.8, "harga_tiket": 350000, "durasi_jam": 3, "fasilitas": "wifi,parkir,toilet,restoran"},
    {"nama": "Malioboro", "provinsi": "Yogyakarta", "kategori": "Kuliner",
     "rating": 4.6, "harga_tiket": 0, "durasi_jam": 4, "fasilitas": "toilet,atm,warung"},
    {"nama": "Keraton Yogyakarta", "provinsi": "Yogyakarta", "kategori": "Budaya",
     "rating": 4.7, "harga_tiket": 15000, "durasi_jam": 2, "fasilitas": "guide,toilet,parkir"},
    {"nama": "Goa Pindul", "provinsi": "Yogyakarta", "kategori": "Petualangan",
     "rating": 4.5, "harga_tiket": 70000, "durasi_jam": 2, "fasilitas": "guide,toilet,warung"},

    # Raja Ampat
    {"nama": "Wayag Island", "provinsi": "Raja Ampat", "kategori": "Bahari",
     "rating": 4.9, "harga_tiket": 500000, "durasi_jam": 8, "fasilitas": "guide,perahu"},
    {"nama": "Pianemo", "provinsi": "Raja Ampat", "kategori": "Bahari",
     "rating": 4.8, "harga_tiket": 250000, "durasi_jam": 6, "fasilitas": "guide,perahu,toilet"},
    {"nama": "Pasir Timbul", "provinsi": "Raja Ampat", "kategori": "Pantai",
     "rating": 4.7, "harga_tiket": 150000, "durasi_jam": 4, "fasilitas": "guide,perahu"},

    # Labuan Bajo
    {"nama": "Pulau Komodo", "provinsi": "Labuan Bajo", "kategori": "Alam",
     "rating": 4.9, "harga_tiket": 150000, "durasi_jam": 6, "fasilitas": "guide,perahu,toilet"},
    {"nama": "Pink Beach", "provinsi": "Labuan Bajo", "kategori": "Pantai",
     "rating": 4.8, "harga_tiket": 100000, "durasi_jam": 4, "fasilitas": "guide,perahu"},
    {"nama": "Padar Island", "provinsi": "Labuan Bajo", "kategori": "Petualangan",
     "rating": 4.8, "harga_tiket": 100000, "durasi_jam": 5, "fasilitas": "guide,perahu,toilet"},

    # Bromo
    {"nama": "Gunung Bromo", "provinsi": "Bromo", "kategori": "Gunung",
     "rating": 4.9, "harga_tiket": 320000, "durasi_jam": 8, "fasilitas": "guide,toilet,warung,jeep"},
    {"nama": "Bukit Teletubbies", "provinsi": "Bromo", "kategori": "Alam",
     "rating": 4.6, "harga_tiket": 100000, "durasi_jam": 3, "fasilitas": "guide,toilet,jeep"},

    # Toba
    {"nama": "Danau Toba", "provinsi": "Toba", "kategori": "Alam",
     "rating": 4.8, "harga_tiket": 25000, "durasi_jam": 6, "fasilitas": "wifi,parkir,toilet,restoran"},
    {"nama": "Pulau Samosir", "provinsi": "Toba", "kategori": "Budaya",
     "rating": 4.7, "harga_tiket": 50000, "durasi_jam": 5, "fasilitas": "guide,toilet,restoran"},

    # Lombok
    {"nama": "Pantai Kuta Lombok", "provinsi": "Lombok", "kategori": "Pantai",
     "rating": 4.7, "harga_tiket": 0, "durasi_jam": 4, "fasilitas": "parkir,toilet,warung"},
    {"nama": "Gunung Rinjani", "provinsi": "Lombok", "kategori": "Gunung",
     "rating": 4.9, "harga_tiket": 150000, "durasi_jam": 48, "fasilitas": "guide,camping"},
    {"nama": "Gili Trawangan", "provinsi": "Lombok", "kategori": "Bahari",
     "rating": 4.7, "harga_tiket": 20000, "durasi_jam": 6, "fasilitas": "wifi,toilet,restoran"},

    # Belitung
    {"nama": "Pantai Tanjung Tinggi", "provinsi": "Belitung", "kategori": "Pantai",
     "rating": 4.8, "harga_tiket": 15000, "durasi_jam": 3, "fasilitas": "parkir,toilet,warung"},
    {"nama": "Pulau Lengkuas", "provinsi": "Belitung", "kategori": "Bahari",
     "rating": 4.7, "harga_tiket": 200000, "durasi_jam": 6, "fasilitas": "perahu,toilet"},

    # Jakarta
    {"nama": "Kota Tua Jakarta", "provinsi": "Jakarta", "kategori": "Sejarah",
     "rating": 4.4, "harga_tiket": 5000, "durasi_jam": 3, "fasilitas": "wifi,parkir,toilet,guide"},
    {"nama": "Kepulauan Seribu", "provinsi": "Jakarta", "kategori": "Bahari",
     "rating": 4.5, "harga_tiket": 100000, "durasi_jam": 8, "fasilitas": "perahu,toilet,restoran"},

    # Wakatobi
    {"nama": "Pantai Cemara Wakatobi", "provinsi": "Wakatobi", "kategori": "Bahari",
     "rating": 4.9, "harga_tiket": 100000, "durasi_jam": 5, "fasilitas": "guide,perahu,toilet"},
    {"nama": "Pulau Hoga", "provinsi": "Wakatobi", "kategori": "Bahari",
     "rating": 4.8, "harga_tiket": 150000, "durasi_jam": 6, "fasilitas": "guide,perahu,diving"},
]

JENIS_KELAMIN = ["Laki-laki", "Perempuan"]
SEGMEN_WISATAWAN = ["Backpacker", "Family", "Honeymoon", "Solo Traveler", "Group Tour"]
TIPE_AKOMODASI = ["Budget", "Midrange", "Luxury", "Hostel", "Villa"]
METODE_BAYAR = ["Transfer Bank", "QRIS", "Kartu Kredit", "OVO", "GoPay", "Dana"]
STATUS_BOOKING = ["Selesai", "Dibatalkan", "Dalam Proses"]
INTERAKSI = ["view", "like", "bookmark", "share", "review", "book"]

# ─── 1. User Profiles ─────────────────────────────────────────────────────────
def generate_user_profiles(n=2000):
    print(f"Membuat {n} profil wisatawan...")
    users = []
    for i in range(1, n + 1):
        usia = random.randint(18, 65)
        segmen = random.choice(SEGMEN_WISATAWAN)
        # Budget disesuaikan dengan segmen
        if segmen == "Backpacker":
            budget = random.randint(500_000, 2_000_000)
        elif segmen == "Family":
            budget = random.randint(3_000_000, 15_000_000)
        elif segmen == "Luxury":
            budget = random.randint(10_000_000, 50_000_000)
        else:
            budget = random.randint(1_500_000, 8_000_000)

        users.append({
            "user_id": f"USR{i:05d}",
            "nama": f"Wisatawan_{i:05d}",
            "usia": usia,
            "jenis_kelamin": random.choice(JENIS_KELAMIN),
            "kota_asal": random.choice([
                "Jakarta", "Surabaya", "Bandung", "Medan", "Makassar",
                "Semarang", "Palembang", "Tangerang", "Depok", "Bekasi",
                "Yogyakarta", "Denpasar", "Padang", "Pekanbaru", "Balikpapan"
            ]),
            "segmen": segmen,
            "preferensi_kategori": "|".join(random.sample(KATEGORI_WISATA, k=random.randint(2, 4))),
            "budget_per_trip": budget,
            "tipe_akomodasi": random.choice(TIPE_AKOMODASI),
            "jumlah_trip": random.randint(1, 30),
            "member_sejak": (
                datetime.now() - timedelta(days=random.randint(30, 1460))
            ).strftime("%Y-%m-%d"),
            "rating_rata_rata": round(random.uniform(3.5, 5.0), 2),
            "is_active": random.choice([True, True, True, False]),
        })
    return pd.DataFrame(users)


# ─── 2. Destination Catalog ───────────────────────────────────────────────────
def generate_destination_catalog():
    print(f"Membuat katalog {len(DESTINASI)} destinasi wisata...")
    catalog = []
    for idx, d in enumerate(DESTINASI, 1):
        catalog.append({
            "destination_id": f"DST{idx:04d}",
            "nama_destinasi": d["nama"],
            "provinsi": d["provinsi"],
            "kategori": d["kategori"],
            "rating_rata_rata": d["rating"],
            "harga_tiket_dewasa": d["harga_tiket"],
            "durasi_kunjungan_jam": d["durasi_jam"],
            "fasilitas": d["fasilitas"],
            "deskripsi": f"Destinasi wisata {d['kategori'].lower()} unggulan di {d['provinsi']}",
            "koordinat_lat": round(random.uniform(-8.5, 2.0), 6),
            "koordinat_lng": round(random.uniform(95.0, 141.0), 6),
            "tersedia": True,
            "musim_terbaik": random.choice(["April-Oktober", "Mei-September", "Sepanjang Tahun"]),
            "min_umur": random.choice([0, 5, 12, 17]),
            "total_pengunjung_tahunan": random.randint(10_000, 2_000_000),
        })
    return pd.DataFrame(catalog)


# ─── 3. User Interactions ─────────────────────────────────────────────────────
def generate_user_interactions(users_df, destinations_df, n=15000):
    print(f"Membuat {n} data interaksi...")
    user_ids = users_df["user_id"].tolist()
    dst_ids = destinations_df["destination_id"].tolist()

    interactions = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(1, n + 1):
        user_id = random.choice(user_ids)
        dst_id = random.choice(dst_ids)
        tipe = random.choice(INTERAKSI)

        # Skor disesuaikan dengan tipe interaksi
        skor_map = {
            "view": random.uniform(0.1, 0.3),
            "like": random.uniform(0.5, 0.7),
            "bookmark": random.uniform(0.6, 0.8),
            "share": random.uniform(0.7, 0.9),
            "review": random.uniform(0.8, 1.0),
            "book": 1.0,
        }

        tanggal = base_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(6, 22),
            minutes=random.randint(0, 59)
        )

        interactions.append({
            "interaction_id": f"INT{i:07d}",
            "user_id": user_id,
            "destination_id": dst_id,
            "tipe_interaksi": tipe,
            "skor_interaksi": round(skor_map[tipe], 4),
            "timestamp": tanggal.strftime("%Y-%m-%d %H:%M:%S"),
            "durasi_detik": random.randint(5, 900) if tipe == "view" else None,
            "platform": random.choice(["Android", "iOS", "Web", "Desktop"]),
            "rating_diberikan": round(random.uniform(3.0, 5.0), 1) if tipe == "review" else None,
            "session_id": f"SES{random.randint(100000, 999999)}",
        })
    return pd.DataFrame(interactions)


# ─── 4. Transaction History ───────────────────────────────────────────────────
def generate_transactions(users_df, destinations_df, n=5000):
    print(f"Membuat {n} data transaksi...")
    user_ids = users_df["user_id"].tolist()
    dst_lookup = destinations_df.set_index("destination_id").to_dict("index")
    dst_ids = list(dst_lookup.keys())

    transactions = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(1, n + 1):
        user_id = random.choice(user_ids)
        dst_id = random.choice(dst_ids)
        dst_info = dst_lookup[dst_id]
        jml_orang = random.randint(1, 8)
        harga_per_orang = dst_info["harga_tiket_dewasa"]
        subtotal = harga_per_orang * jml_orang
        diskon_pct = random.choice([0, 0, 0, 5, 10, 15, 20])
        diskon_nominal = int(subtotal * diskon_pct / 100)
        total = subtotal - diskon_nominal

        check_in = base_date + timedelta(days=random.randint(7, 365))
        check_out = check_in + timedelta(days=random.randint(1, 7))
        status = random.choices(
            STATUS_BOOKING,
            weights=[0.75, 0.10, 0.15]
        )[0]

        transactions.append({
            "transaction_id": f"TRX{i:07d}",
            "user_id": user_id,
            "destination_id": dst_id,
            "nama_destinasi": dst_info["nama_destinasi"],
            "jumlah_orang": jml_orang,
            "harga_per_orang": harga_per_orang,
            "subtotal": subtotal,
            "diskon_persen": diskon_pct,
            "diskon_nominal": diskon_nominal,
            "total_bayar": total,
            "metode_pembayaran": random.choice(METODE_BAYAR),
            "tanggal_transaksi": (check_in - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "tanggal_check_in": check_in.strftime("%Y-%m-%d"),
            "tanggal_check_out": check_out.strftime("%Y-%m-%d"),
            "status_booking": status,
            "tipe_akomodasi": random.choice(TIPE_AKOMODASI),
            "rating_pasca_kunjungan": (
                round(random.uniform(3.5, 5.0), 1)
                if status == "Selesai" else None
            ),
            "ulasan": (
                f"Pengalaman luar biasa di {dst_info['nama_destinasi']}!"
                if status == "Selesai" and random.random() > 0.5
                else None
            ),
        })
    return pd.DataFrame(transactions)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NusantaraAI - Dataset Generator")
    print("=" * 60)

    users_df = generate_user_profiles(2000)
    users_df.to_csv("dataset/user_profiles.csv", index=False)
    print(f"  ✓ user_profiles.csv ({len(users_df)} baris)")

    destinations_df = generate_destination_catalog()
    destinations_df.to_csv("dataset/destination_catalog.csv", index=False)
    print(f"  ✓ destination_catalog.csv ({len(destinations_df)} baris)")

    interactions_df = generate_user_interactions(users_df, destinations_df, 15000)
    interactions_df.to_csv("dataset/user_interactions.csv", index=False)
    print(f"  ✓ user_interactions.csv ({len(interactions_df)} baris)")

    transactions_df = generate_transactions(users_df, destinations_df, 5000)
    transactions_df.to_csv("dataset/transaction_history.csv", index=False)
    print(f"  ✓ transaction_history.csv ({len(transactions_df)} baris)")

    print()
    print("Semua dataset berhasil dibuat di folder ./dataset/")
    print()
    print("Upload ke S3:")
    print("  aws s3 sync dataset/ s3://nusantaraai-ml-PROVINSI-NAMA/raw-data/")
