"""
====================================================
NusantaraAI - AWS Glue ETL Job (PySpark)
====================================================
Script ini dijalankan oleh AWS Glue Job: nusantaraai-etl-job
Input  : s3://nusantaraai-ml-PROVINSI-NAMA/raw-data/
Output : s3://nusantaraai-ml-PROVINSI-NAMA/processed-data/

Proses ETL:
1. Baca semua dataset dari S3 (raw-data)
2. Bersihkan & validasi data (hapus duplikat, null, outlier)
3. Feature engineering (normalisasi, encoding kategori)
4. Buat user-item interaction matrix
5. Hitung content-based features untuk destinasi
6. Simpan ke processed-data/ dalam format Parquet
====================================================
"""

import sys
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer, OneHotEncoder

# ─── Inisialisasi Glue & Spark ─────────────────────────────────────────────
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'S3_INPUT_PATH',
    'S3_OUTPUT_PATH',
    'DATABASE_NAME'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

INPUT_PATH  = args['S3_INPUT_PATH']   # s3://nusantaraai-ml-xxx/raw-data/
OUTPUT_PATH = args['S3_OUTPUT_PATH']  # s3://nusantaraai-ml-xxx/processed-data/
DB_NAME     = args['DATABASE_NAME']   # nusantaraai_database

print(f"[ETL] Start: {datetime.now()}")
print(f"[ETL] Input : {INPUT_PATH}")
print(f"[ETL] Output: {OUTPUT_PATH}")


# ════════════════════════════════════════════════════════════════
# BAGIAN 1: BACA DATA
# ════════════════════════════════════════════════════════════════

print("[ETL] === BAGIAN 1: Membaca Data dari S3 ===")

# Baca dari Glue Data Catalog (sudah di-crawl)
users_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=DB_NAME,
    table_name="user_profiles"
)

destinations_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=DB_NAME,
    table_name="destination_catalog"
)

interactions_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=DB_NAME,
    table_name="user_interactions"
)

transactions_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=DB_NAME,
    table_name="transaction_history"
)

# Konversi ke Spark DataFrame
users_df        = users_dyf.toDF()
destinations_df = destinations_dyf.toDF()
interactions_df = interactions_dyf.toDF()
transactions_df = transactions_dyf.toDF()

print(f"  Users       : {users_df.count()} baris")
print(f"  Destinations: {destinations_df.count()} baris")
print(f"  Interactions: {interactions_df.count()} baris")
print(f"  Transactions: {transactions_df.count()} baris")


# ════════════════════════════════════════════════════════════════
# BAGIAN 2: CLEANING & VALIDASI
# ════════════════════════════════════════════════════════════════

print("[ETL] === BAGIAN 2: Data Cleaning ===")

# ── 2a. User Profiles ─────────────────────────────────────────
users_clean = users_df \
    .dropDuplicates(["user_id"]) \
    .filter(F.col("user_id").isNotNull()) \
    .filter(F.col("usia").between(12, 80)) \
    .filter(F.col("budget_per_trip") > 0) \
    .withColumn("budget_per_trip", F.col("budget_per_trip").cast(LongType())) \
    .withColumn("jumlah_trip", F.col("jumlah_trip").cast(IntegerType())) \
    .withColumn("member_sejak", F.to_date("member_sejak", "yyyy-MM-dd")) \
    .withColumn("hari_sejak_join",
        F.datediff(F.current_date(), F.col("member_sejak"))) \
    .fillna({"is_active": True})

print(f"  Users clean : {users_clean.count()} baris")

# ── 2b. Destinations ──────────────────────────────────────────
destinations_clean = destinations_df \
    .dropDuplicates(["destination_id"]) \
    .filter(F.col("destination_id").isNotNull()) \
    .filter(F.col("rating_rata_rata").between(1.0, 5.0)) \
    .filter(F.col("harga_tiket_dewasa") >= 0) \
    .fillna({
        "fasilitas": "toilet",
        "tersedia": True,
        "total_pengunjung_tahunan": 0
    })

print(f"  Destinations: {destinations_clean.count()} baris")

# ── 2c. Interactions ──────────────────────────────────────────
interactions_clean = interactions_df \
    .dropDuplicates(["interaction_id"]) \
    .filter(F.col("user_id").isNotNull()) \
    .filter(F.col("destination_id").isNotNull()) \
    .filter(F.col("skor_interaksi").between(0.0, 1.0)) \
    .withColumn("timestamp",
        F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("jam", F.hour("timestamp")) \
    .withColumn("hari_minggu", F.dayofweek("timestamp")) \
    .withColumn("bulan", F.month("timestamp"))

print(f"  Interactions: {interactions_clean.count()} baris")

# ── 2d. Transactions ──────────────────────────────────────────
transactions_clean = transactions_df \
    .dropDuplicates(["transaction_id"]) \
    .filter(F.col("user_id").isNotNull()) \
    .filter(F.col("total_bayar") > 0) \
    .filter(F.col("status_booking").isin(["Selesai", "Dibatalkan", "Dalam Proses"])) \
    .withColumn("tanggal_transaksi",
        F.to_date("tanggal_transaksi", "yyyy-MM-dd")) \
    .withColumn("tanggal_check_in",
        F.to_date("tanggal_check_in", "yyyy-MM-dd")) \
    .withColumn("durasi_menginap",
        F.datediff(F.col("tanggal_check_out"), F.col("tanggal_check_in"))) \
    .withColumn("is_completed",
        (F.col("status_booking") == "Selesai").cast(IntegerType()))

print(f"  Transactions: {transactions_clean.count()} baris")


# ════════════════════════════════════════════════════════════════
# BAGIAN 3: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

print("[ETL] === BAGIAN 3: Feature Engineering ===")

# ── 3a. User Features ─────────────────────────────────────────

# Agregasi interaksi per user
user_interaction_agg = interactions_clean.groupBy("user_id").agg(
    F.count("interaction_id").alias("total_interaksi"),
    F.avg("skor_interaksi").alias("avg_skor"),
    F.sum(F.when(F.col("tipe_interaksi") == "book", 1).otherwise(0)).alias("total_book"),
    F.sum(F.when(F.col("tipe_interaksi") == "review", 1).otherwise(0)).alias("total_review"),
    F.sum(F.when(F.col("tipe_interaksi") == "like", 1).otherwise(0)).alias("total_like"),
    F.countDistinct("destination_id").alias("unique_destinasi"),
    F.max("timestamp").alias("last_interaction")
)

# Agregasi transaksi per user
user_transaction_agg = transactions_clean.groupBy("user_id").agg(
    F.count("transaction_id").alias("total_transaksi"),
    F.avg("total_bayar").alias("avg_spend"),
    F.max("total_bayar").alias("max_spend"),
    F.avg("rating_pasca_kunjungan").alias("avg_rating_diberikan"),
    F.avg("durasi_menginap").alias("avg_durasi_trip"),
    F.sum("is_completed").alias("transaksi_selesai")
)

# Gabungkan ke users
users_featured = users_clean \
    .join(user_interaction_agg, "user_id", "left") \
    .join(user_transaction_agg, "user_id", "left") \
    .fillna({
        "total_interaksi": 0, "avg_skor": 0.0,
        "total_book": 0, "total_review": 0, "total_like": 0,
        "unique_destinasi": 0, "total_transaksi": 0, "avg_spend": 0.0,
        "transaksi_selesai": 0
    })

# Normalisasi budget (log transform untuk distribusi skewed)
users_featured = users_featured.withColumn(
    "budget_log", F.log1p(F.col("budget_per_trip"))
)

# Engagement score (composite metric)
users_featured = users_featured.withColumn(
    "engagement_score",
    (F.col("total_book") * 5 +
     F.col("total_review") * 3 +
     F.col("total_like") * 1 +
     F.col("transaksi_selesai") * 4) / 13.0
)

print(f"  Users featured: {users_featured.count()} baris, {len(users_featured.columns)} kolom")

# ── 3b. Destination Features ──────────────────────────────────

# Hitung popularity dari interaksi
dst_popularity = interactions_clean.groupBy("destination_id").agg(
    F.count("interaction_id").alias("total_views"),
    F.avg("skor_interaksi").alias("avg_interaction_score"),
    F.countDistinct("user_id").alias("unique_visitors"),
    F.sum(F.when(F.col("tipe_interaksi") == "book", 1).otherwise(0)).alias("total_bookings"),
    F.avg(F.when(F.col("tipe_interaksi") == "review",
        F.col("rating_diberikan"))).alias("avg_user_rating")
)

# Hitung revenue dari transaksi
dst_revenue = transactions_clean \
    .filter(F.col("status_booking") == "Selesai") \
    .groupBy("destination_id").agg(
    F.sum("total_bayar").alias("total_revenue"),
    F.count("transaction_id").alias("total_completed_bookings"),
    F.avg("total_bayar").alias("avg_transaction_value")
)

# Fasilitas sebagai fitur biner
destinations_featured = destinations_clean \
    .join(dst_popularity, "destination_id", "left") \
    .join(dst_revenue, "destination_id", "left") \
    .fillna({
        "total_views": 0, "unique_visitors": 0, "total_bookings": 0,
        "total_revenue": 0, "avg_transaction_value": 0
    })

# Buat fitur fasilitas biner
facilities_to_check = ["wifi", "parkir", "toilet", "restoran", "guide", "perahu", "diving"]
for facility in facilities_to_check:
    destinations_featured = destinations_featured.withColumn(
        f"has_{facility}",
        F.when(F.col("fasilitas").contains(facility), 1).otherwise(0)
    )

# Popularity score
destinations_featured = destinations_featured.withColumn(
    "popularity_score",
    (F.coalesce(F.col("total_views"), F.lit(0)) * 0.3 +
     F.coalesce(F.col("unique_visitors"), F.lit(0)) * 0.4 +
     F.coalesce(F.col("total_bookings"), F.lit(0)) * 0.3)
)

# Normalisasi harga per kategori
window_kategori = Window.partitionBy("kategori")
destinations_featured = destinations_featured.withColumn(
    "harga_normalized",
    (F.col("harga_tiket_dewasa") - F.min("harga_tiket_dewasa").over(window_kategori)) /
    (F.max("harga_tiket_dewasa").over(window_kategori) -
     F.min("harga_tiket_dewasa").over(window_kategori) + F.lit(1))
)

print(f"  Destinations featured: {destinations_featured.count()} baris")

# ── 3c. User-Item Interaction Matrix ──────────────────────────

# Gabungkan semua sinyal interaksi & transaksi
interaction_matrix = interactions_clean.select(
    "user_id", "destination_id", "skor_interaksi"
).union(
    transactions_clean
    .filter(F.col("status_booking") == "Selesai")
    .select(
        F.col("user_id"),
        F.col("destination_id"),
        F.lit(1.0).alias("skor_interaksi")
    )
).groupBy("user_id", "destination_id").agg(
    F.sum("skor_interaksi").alias("total_score"),
    F.count("*").alias("interaction_count")
).withColumn(
    "final_score",
    # Clip ke 0-5 range
    F.least(F.col("total_score"), F.lit(5.0))
)

print(f"  Interaction matrix: {interaction_matrix.count()} pasang user-destinasi")

# ── 3d. Kategori Encoding ─────────────────────────────────────

# Encode kategori destinasi untuk ML
kategori_encoder = StringIndexer(
    inputCol="kategori",
    outputCol="kategori_idx",
    handleInvalid="keep"
)

destinations_featured = kategori_encoder.fit(destinations_featured) \
    .transform(destinations_featured)

provinsi_encoder = StringIndexer(
    inputCol="provinsi",
    outputCol="provinsi_idx",
    handleInvalid="keep"
)

destinations_featured = provinsi_encoder.fit(destinations_featured) \
    .transform(destinations_featured)


# ════════════════════════════════════════════════════════════════
# BAGIAN 4: SIMPAN KE S3 (PARQUET FORMAT)
# ════════════════════════════════════════════════════════════════

print("[ETL] === BAGIAN 4: Menyimpan ke S3 ===")

# User features
users_featured.write \
    .mode("overwrite") \
    .parquet(f"{OUTPUT_PATH}/user_features/")
print(f"  ✓ user_features/ disimpan")

# Destination features
destinations_featured.write \
    .mode("overwrite") \
    .parquet(f"{OUTPUT_PATH}/destination_features/")
print(f"  ✓ destination_features/ disimpan")

# Interaction matrix (untuk collaborative filtering)
interaction_matrix.write \
    .mode("overwrite") \
    .parquet(f"{OUTPUT_PATH}/interaction_matrix/")
print(f"  ✓ interaction_matrix/ disimpan")

# Clean transactions (untuk analisis)
transactions_clean.write \
    .mode("overwrite") \
    .parquet(f"{OUTPUT_PATH}/transactions_clean/")
print(f"  ✓ transactions_clean/ disimpan")

# ── Simpan metadata ringkasan ──────────────────────────────────
summary_data = [
    ("total_users", str(users_featured.count())),
    ("total_destinations", str(destinations_featured.count())),
    ("total_interactions", str(interaction_matrix.count())),
    ("total_transactions", str(transactions_clean.count())),
    ("etl_timestamp", str(datetime.now())),
    ("etl_job", args['JOB_NAME']),
]
summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.write \
    .mode("overwrite") \
    .csv(f"{OUTPUT_PATH}/metadata/etl_summary/", header=True)
print(f"  ✓ metadata/etl_summary/ disimpan")

print(f"[ETL] Selesai: {datetime.now()}")
job.commit()
