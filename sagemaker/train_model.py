"""
====================================================
NusantaraAI - SageMaker Model Training
====================================================
Jalankan script ini di SageMaker Notebook Instance:
  nusantaraai-ml-PROVINSI-NAMA (ml.t3.medium)

Algoritma:
  1. Collaborative Filtering (ALS via Surprise library)
  2. Content-Based Filtering (cosine similarity)
  3. Hybrid Model (weighted ensemble)
  4. Export model ke S3: models/hybrid_model.pkl

Install dependencies dulu di notebook terminal:
  pip install scikit-surprise scikit-learn boto3 pandas
====================================================
"""

import os
import json
import pickle
import warnings
import boto3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

# Coba import surprise (collaborative filtering)
try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import cross_validate, train_test_split as surprise_split
    SURPRISE_AVAILABLE = True
    print("✓ Surprise library tersedia (Collaborative Filtering aktif)")
except ImportError:
    SURPRISE_AVAILABLE = False
    print("⚠ Surprise tidak tersedia, pakai alternatif sederhana")

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ.get("S3_BUCKET", "nusantaraai-ml-jawatengah-namakamu")
S3_INPUT     = f"s3://{S3_BUCKET}/processed-data/"
S3_MODEL_OUT = f"s3://{S3_BUCKET}/models/"
REGION       = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=REGION)

print(f"[Training] Bucket: {S3_BUCKET}")
print(f"[Training] Model output: {S3_MODEL_OUT}")


# ════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA DARI S3
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 1: Load Data ===")

def download_parquet_from_s3(s3_path: str) -> pd.DataFrame:
    """Download file Parquet dari S3 ke DataFrame."""
    bucket = s3_path.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_path.replace("s3://", "").split("/")[1:])

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    parquet_files = [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    dfs = []
    for key in parquet_files:
        local_path = f"/tmp/{key.split('/')[-1]}"
        s3.download_file(bucket, key, local_path)
        dfs.append(pd.read_parquet(local_path))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load semua processed data
try:
    users_df        = download_parquet_from_s3(f"{S3_INPUT}user_features/")
    destinations_df = download_parquet_from_s3(f"{S3_INPUT}destination_features/")
    interactions_df = download_parquet_from_s3(f"{S3_INPUT}interaction_matrix/")
    print(f"  Users       : {len(users_df)} baris")
    print(f"  Destinations: {len(destinations_df)} baris")
    print(f"  Interactions: {len(interactions_df)} baris")
except Exception as e:
    print(f"⚠ Gagal load dari S3, pakai data lokal: {e}")
    # Fallback: load dari dataset/ folder (jika ada)
    users_df        = pd.read_csv("dataset/user_profiles.csv")
    destinations_df = pd.read_csv("dataset/destination_catalog.csv")
    interactions_df = pd.read_csv("dataset/user_interactions.csv")


# ════════════════════════════════════════════════════════════════
# STEP 2: CONTENT-BASED FILTERING
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 2: Content-Based Filtering ===")

class ContentBasedRecommender:
    """
    Rekomendasi berbasis konten destinasi wisata.
    Menggunakan cosine similarity pada fitur numerik destinasi.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.similarity_matrix = None
        self.destinations = None
        self.feature_cols = []

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Bangun matrix fitur dari DataFrame destinasi."""
        numeric_cols = [
            "rating_rata_rata", "harga_tiket_dewasa", "durasi_kunjungan_jam",
            "total_pengunjung_tahunan"
        ]
        # Pastikan kolom ada
        self.feature_cols = [c for c in numeric_cols if c in df.columns]

        # Tambah kolom fasilitas binary
        facility_cols = [c for c in df.columns if c.startswith("has_")]
        self.feature_cols.extend(facility_cols)

        # Tambah encoding kategori jika ada
        if "kategori_idx" in df.columns:
            self.feature_cols.append("kategori_idx")
        if "provinsi_idx" in df.columns:
            self.feature_cols.append("provinsi_idx")

        # Jika kolom tidak ada, encode secara manual
        df_work = df.copy()
        if "kategori_idx" not in df_work.columns and "kategori" in df_work.columns:
            le = LabelEncoder()
            df_work["kategori_idx"] = le.fit_transform(df_work["kategori"].fillna("Alam"))
            self.feature_cols.append("kategori_idx")

        if "provinsi_idx" not in df_work.columns and "provinsi" in df_work.columns:
            le = LabelEncoder()
            df_work["provinsi_idx"] = le.fit_transform(df_work["provinsi"].fillna("Bali"))
            self.feature_cols.append("provinsi_idx")

        features = df_work[self.feature_cols].fillna(0).values
        return self.scaler.fit_transform(features)

    def fit(self, destinations_df: pd.DataFrame):
        """Training: hitung similarity matrix."""
        self.destinations = destinations_df.copy().reset_index(drop=True)

        feature_matrix = self._build_feature_matrix(self.destinations)
        self.similarity_matrix = cosine_similarity(feature_matrix)

        print(f"  Feature matrix shape : {feature_matrix.shape}")
        print(f"  Similarity matrix    : {self.similarity_matrix.shape}")
        print(f"  Fitur yang digunakan : {self.feature_cols}")
        return self

    def get_similar_destinations(self, destination_id: str, top_n: int = 5) -> list:
        """Dapatkan destinasi mirip berdasarkan konten."""
        dst_col = "destination_id" if "destination_id" in self.destinations.columns else self.destinations.columns[0]

        if destination_id not in self.destinations[dst_col].values:
            return []

        idx = self.destinations[self.destinations[dst_col] == destination_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

        results = []
        for idx_dst, score in sim_scores:
            row = self.destinations.iloc[idx_dst]
            results.append({
                "destination_id": row.get(dst_col, f"DST{idx_dst:04d}"),
                "nama": row.get("nama_destinasi", ""),
                "kategori": row.get("kategori", ""),
                "rating": row.get("rating_rata_rata", 0),
                "similarity_score": round(float(score), 4),
            })
        return results


# Latih Content-Based Recommender
cb_model = ContentBasedRecommender()
cb_model.fit(destinations_df)

# Test
if len(destinations_df) > 0:
    test_dst = destinations_df.iloc[0].get("destination_id", "DST0001")
    similar = cb_model.get_similar_destinations(test_dst, top_n=3)
    print(f"  Test - Destinasi mirip dengan {test_dst}:")
    for s in similar:
        print(f"    → {s['nama']} ({s['kategori']}) | score: {s['similarity_score']}")


# ════════════════════════════════════════════════════════════════
# STEP 3: COLLABORATIVE FILTERING
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 3: Collaborative Filtering (SVD) ===")

class CollaborativeFilteringModel:
    """Matrix Factorization menggunakan SVD (Singular Value Decomposition)."""

    def __init__(self):
        self.model = None
        self.user_avg = {}
        self.global_avg = 3.5
        self.is_trained = False

    def fit(self, interactions_df: pd.DataFrame):
        """Latih model CF dari data interaksi."""
        required_cols = {"user_id", "destination_id"}

        # Normalisasi nama kolom
        score_col = None
        for col in ["final_score", "skor_interaksi", "total_score"]:
            if col in interactions_df.columns:
                score_col = col
                break

        if score_col is None:
            print("  ⚠ Tidak ada kolom skor, pakai nilai default 1.0")
            interactions_df = interactions_df.copy()
            interactions_df["score"] = 1.0
            score_col = "score"

        df = interactions_df[["user_id", "destination_id", score_col]].copy()
        df.columns = ["user_id", "destination_id", "rating"]
        df["rating"] = df["rating"].clip(0, 5).fillna(1.0)

        self.global_avg = df["rating"].mean()
        self.user_avg = df.groupby("user_id")["rating"].mean().to_dict()

        if SURPRISE_AVAILABLE and len(df) > 50:
            reader = Reader(rating_scale=(0, 5))
            data = Dataset.load_from_df(df, reader)
            trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

            self.model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
            self.model.fit(trainset)

            predictions = self.model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            print(f"  SVD RMSE: {rmse:.4f}")
        else:
            print(f"  ⚠ Pakai fallback (user average) karena data kecil atau Surprise tidak ada")

        self.is_trained = True
        self.df_train = df
        return self

    def predict(self, user_id: str, destination_id: str) -> float:
        """Prediksi rating untuk pasangan user-destinasi."""
        if not self.is_trained:
            return self.global_avg

        if SURPRISE_AVAILABLE and self.model:
            try:
                pred = self.model.predict(user_id, destination_id)
                return pred.est
            except Exception:
                pass

        return self.user_avg.get(user_id, self.global_avg)

    def recommend_for_user(self, user_id: str, all_destination_ids: list,
                          exclude_seen: bool = True, top_n: int = 10) -> list:
        """Rekomendasikan top-N destinasi untuk seorang user."""
        if not self.is_trained:
            return []

        # Filter destinasi yang sudah dikunjungi
        seen = set()
        if exclude_seen and hasattr(self, "df_train"):
            seen = set(
                self.df_train[self.df_train["user_id"] == user_id]["destination_id"].tolist()
            )

        candidates = [d for d in all_destination_ids if d not in seen]

        predictions = []
        for dst_id in candidates:
            score = self.predict(user_id, dst_id)
            predictions.append((dst_id, score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"destination_id": d, "cf_score": round(s, 4)}
                for d, s in predictions[:top_n]]


# Latih Collaborative Filtering
cf_model = CollaborativeFilteringModel()
cf_model.fit(interactions_df)


# ════════════════════════════════════════════════════════════════
# STEP 4: HYBRID MODEL
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 4: Hybrid Recommendation Model ===")

class HybridRecommender:
    """
    Gabungan CF + Content-Based dengan bobot adaptif.
    - User baru (cold start): bobot content-based lebih tinggi
    - User lama (banyak interaksi): bobot CF lebih tinggi
    """

    def __init__(self, cf_model, cb_model,
                 alpha_cf=0.6, alpha_cb=0.4):
        self.cf_model  = cf_model
        self.cb_model  = cb_model
        self.alpha_cf  = alpha_cf
        self.alpha_cb  = alpha_cb
        self.destinations = None
        self.all_dst_ids  = []
        self.version = "1.0.0"
        self.trained_at = datetime.now().isoformat()

    def fit(self, destinations_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.destinations = destinations_df.copy()

        # Tentukan kolom ID destinasi
        dst_id_col = "destination_id" if "destination_id" in destinations_df.columns \
                     else destinations_df.columns[0]
        self.all_dst_ids = destinations_df[dst_id_col].tolist()

        # Hitung statistik user (untuk adaptif bobot)
        if "user_id" in interactions_df.columns:
            self.user_interaction_counts = interactions_df.groupby("user_id").size().to_dict()
        else:
            self.user_interaction_counts = {}

        return self

    def _get_adaptive_weights(self, user_id: str):
        """Sesuaikan bobot CF vs CB berdasarkan riwayat user."""
        n = self.user_interaction_counts.get(user_id, 0)
        if n < 5:      # Cold start
            return 0.2, 0.8
        elif n < 20:   # Transisi
            return 0.5, 0.5
        else:          # User lama
            return 0.7, 0.3

    def recommend(self, user_id: str, top_n: int = 10,
                  user_preferences: dict = None) -> list:
        """
        Berikan rekomendasi untuk seorang user.

        Args:
            user_id: ID pengguna
            top_n: Jumlah rekomendasi
            user_preferences: {'kategori': ['Pantai', 'Gunung'],
                               'max_budget': 500000, 'provinsi': 'Bali'}
        """
        w_cf, w_cb = self._get_adaptive_weights(user_id)

        # 1. CF recommendations
        cf_recs = self.cf_model.recommend_for_user(
            user_id, self.all_dst_ids, top_n=len(self.all_dst_ids)
        )
        cf_scores = {r["destination_id"]: r["cf_score"] for r in cf_recs}

        # 2. CB recommendations (dari preferensi atau riwayat terbaik)
        cb_scores = {}
        if user_preferences and "favorit_destinasi" in user_preferences:
            for fav_dst in user_preferences["favorit_destinasi"]:
                similar = self.cb_model.get_similar_destinations(fav_dst, top_n=20)
                for s in similar:
                    dst_id = s["destination_id"]
                    cb_scores[dst_id] = max(
                        cb_scores.get(dst_id, 0),
                        s["similarity_score"]
                    )

        # 3. Normalisasi skor
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 0.5 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

        cf_norm = normalize(cf_scores)
        cb_norm = normalize(cb_scores)

        # 4. Gabungkan skor
        all_dst = set(list(cf_norm.keys()) + list(cb_norm.keys()))
        final_scores = {}
        for dst_id in all_dst:
            final_scores[dst_id] = (
                w_cf * cf_norm.get(dst_id, 0) +
                w_cb * cb_norm.get(dst_id, 0)
            )

        # 5. Filter berdasarkan preferensi user
        if user_preferences:
            filtered_scores = {}
            for dst_id, score in final_scores.items():
                dst_info = self.destinations[
                    self.destinations.get("destination_id", self.destinations.columns[0]) == dst_id
                ]
                if dst_info.empty:
                    filtered_scores[dst_id] = score
                    continue

                row = dst_info.iloc[0]
                valid = True

                # Filter kategori
                if "kategori" in user_preferences:
                    prefs = user_preferences["kategori"]
                    if prefs and row.get("kategori", "") not in prefs:
                        valid = False

                # Filter budget
                if "max_budget" in user_preferences and valid:
                    if row.get("harga_tiket_dewasa", 0) > user_preferences["max_budget"]:
                        valid = False

                # Filter provinsi
                if "provinsi" in user_preferences and valid:
                    if row.get("provinsi", "") != user_preferences["provinsi"]:
                        valid = False

                if valid:
                    filtered_scores[dst_id] = score

            if filtered_scores:
                final_scores = filtered_scores

        # 6. Ambil top-N
        sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # 7. Enrich dengan info destinasi
        results = []
        dst_id_col = "destination_id" if "destination_id" in self.destinations.columns \
                     else self.destinations.columns[0]
        dst_lookup = self.destinations.set_index(dst_id_col).to_dict("index")

        for dst_id, hybrid_score in sorted_recs:
            dst_info = dst_lookup.get(dst_id, {})
            results.append({
                "destination_id": dst_id,
                "nama_destinasi": dst_info.get("nama_destinasi", ""),
                "kategori": dst_info.get("kategori", ""),
                "provinsi": dst_info.get("provinsi", ""),
                "rating": dst_info.get("rating_rata_rata", 0),
                "harga_tiket": dst_info.get("harga_tiket_dewasa", 0),
                "hybrid_score": round(hybrid_score, 4),
                "cf_score": round(cf_norm.get(dst_id, 0), 4),
                "cb_score": round(cb_norm.get(dst_id, 0), 4),
                "bobot_cf": w_cf,
                "bobot_cb": w_cb,
            })

        return results

    def predict(self, user_id: str) -> list:
        """Wrapper predict untuk kompatibilitas Lambda."""
        return self.recommend(user_id, top_n=10)


# Latih Hybrid Model
hybrid_model = HybridRecommender(cf_model, cb_model)
hybrid_model.fit(destinations_df, interactions_df)

# Test rekomendasi
print("\n  Test rekomendasi:")
if len(interactions_df) > 0:
    test_user = interactions_df.iloc[0]["user_id"] if "user_id" in interactions_df.columns else "USR00001"
else:
    test_user = "USR00001"

recs = hybrid_model.recommend(test_user, top_n=5)
for r in recs:
    print(f"    → {r['nama_destinasi']} ({r['kategori']}) | score: {r['hybrid_score']}")


# ════════════════════════════════════════════════════════════════
# STEP 5: EVALUASI MODEL
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 5: Evaluasi Model ===")

def evaluate_model(model, interactions_df, sample_users=50):
    """Evaluasi dengan Precision@K dan Coverage."""
    if len(interactions_df) == 0:
        return {"precision_at_5": 0, "coverage": 0}

    score_col = next(
        (c for c in ["final_score", "skor_interaksi", "total_score"] if c in interactions_df.columns),
        None
    )
    if score_col is None:
        return {"precision_at_5": 0, "coverage": 0}

    # Sample users
    unique_users = interactions_df["user_id"].unique() if "user_id" in interactions_df.columns else []
    if len(unique_users) == 0:
        return {"precision_at_5": 0, "coverage": 0}

    sample = np.random.choice(unique_users, min(sample_users, len(unique_users)), replace=False)

    precisions = []
    all_recommended = set()

    for user_id in sample:
        # Ground truth: destinasi dengan skor tinggi
        user_interactions = interactions_df[
            interactions_df["user_id"] == user_id
        ]
        top_true = set(
            user_interactions.nlargest(5, score_col)["destination_id"].tolist()
        )

        # Prediksi
        preds = model.recommend(user_id, top_n=5)
        pred_ids = {p["destination_id"] for p in preds}
        all_recommended.update(pred_ids)

        # Precision@5
        hits = len(top_true & pred_ids)
        precisions.append(hits / 5.0)

    coverage = len(all_recommended) / max(len(model.all_dst_ids), 1)

    metrics = {
        "precision_at_5": round(np.mean(precisions), 4),
        "coverage": round(coverage, 4),
        "users_evaluated": len(sample)
    }
    return metrics

metrics = evaluate_model(hybrid_model, interactions_df)
print(f"  Precision@5 : {metrics['precision_at_5']}")
print(f"  Coverage    : {metrics['coverage']}")


# ════════════════════════════════════════════════════════════════
# STEP 6: SIMPAN MODEL KE S3
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 6: Simpan Model ke S3 ===")

# Package model dengan metadata
model_package = {
    "model": hybrid_model,
    "cf_model": cf_model,
    "cb_model": cb_model,
    "metadata": {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "n_users": len(users_df),
        "n_destinations": len(destinations_df),
        "n_interactions": len(interactions_df),
        "algorithm": "Hybrid CF+CB",
        "cf_algorithm": "SVD (Surprise)" if SURPRISE_AVAILABLE else "UserAverage",
        "cb_algorithm": "Cosine Similarity",
    }
}

# Simpan lokal dulu
local_model_path = "/tmp/hybrid_model.pkl"
with open(local_model_path, "wb") as f:
    pickle.dump(model_package, f)

file_size_mb = os.path.getsize(local_model_path) / (1024 * 1024)
print(f"  Model size  : {file_size_mb:.2f} MB")

# Upload ke S3
model_key = "models/hybrid_model.pkl"
s3.upload_file(local_model_path, S3_BUCKET, model_key)
print(f"  ✓ Model diupload ke s3://{S3_BUCKET}/{model_key}")

# Simpan juga metadata sebagai JSON (mudah dibaca)
metadata_key = "models/model_metadata.json"
s3.put_object(
    Bucket=S3_BUCKET,
    Key=metadata_key,
    Body=json.dumps(model_package["metadata"], indent=2),
    ContentType="application/json"
)
print(f"  ✓ Metadata disimpan ke s3://{S3_BUCKET}/{metadata_key}")

print("\n✅ Training selesai!")
print(f"   Model: s3://{S3_BUCKET}/models/hybrid_model.pkl")
