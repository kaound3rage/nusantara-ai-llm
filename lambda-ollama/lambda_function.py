"""
NusantaraAI - Lambda Recommendation Function
Runtime : Python 3.10
Handler : lambda_function.lambda_handler

Semua class model (HybridRecommender, CollaborativeFilteringModel,
ContentBasedRecommender) didefinisikan di sini agar pickle.load
bisa menemukan class-nya saat inference di Lambda.
"""

import os
import json
import time
import logging
import pickle
import numpy as np
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ─── Surprise stub (tidak dibutuhkan saat inference) ─────────────────────────
try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    import sys, types
    _stub = types.ModuleType("surprise")
    class _SVDStub: pass
    _stub.SVD     = _SVDStub
    _stub.Dataset = None
    _stub.Reader  = None
    sys.modules["surprise"] = _stub

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Environment Variables ───────────────────────────────────────────────────
MODEL_BUCKET  = os.environ.get("MODEL_BUCKET",  "nusantara-ai")
MODEL_KEY     = os.environ.get("MODEL_KEY",     "models/hybrid_model.pkl")
METADATA_KEY  = os.environ.get("METADATA_KEY",  "models/model_metadata.json")
AWS_REGION    = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
TOP_N_DEFAULT = int(os.environ.get("TOP_N_DEFAULT", "5"))

s3 = boto3.client("s3", region_name=AWS_REGION)

# ─── In-Memory Cache ─────────────────────────────────────────────────────────
_CACHE = {"model": None, "metadata": None}


# ════════════════════════════════════════════════════════════════
# MODEL CLASSES
# Harus ada di sini agar pickle.load bisa resolve class-nya
# ════════════════════════════════════════════════════════════════

class ContentBasedRecommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.similarity_matrix = None
        self.destinations = None
        self.feature_cols = []

    def _build_feature_matrix(self, df):
        numeric_cols = [
            "rating_rata_rata", "harga_tiket_dewasa",
            "durasi_kunjungan_jam", "total_pengunjung_tahunan"
        ]
        self.feature_cols = [c for c in numeric_cols if c in df.columns]
        facility_cols = [c for c in df.columns if c.startswith("has_")]
        self.feature_cols.extend(facility_cols)

        if "kategori_idx" in df.columns:
            self.feature_cols.append("kategori_idx")
        if "provinsi_idx" in df.columns:
            self.feature_cols.append("provinsi_idx")

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

    def fit(self, destinations_df):
        self.destinations = destinations_df.copy().reset_index(drop=True)
        feature_matrix = self._build_feature_matrix(self.destinations)
        self.similarity_matrix = cosine_similarity(feature_matrix)
        return self

    def get_similar_destinations(self, destination_id, top_n=5):
        dst_col = "destination_id" if "destination_id" in self.destinations.columns \
                  else self.destinations.columns[0]
        if destination_id not in self.destinations[dst_col].values:
            return []
        idx = self.destinations[self.destinations[dst_col] == destination_id].index[0]
        sim_scores = sorted(enumerate(self.similarity_matrix[idx]),
                            key=lambda x: x[1], reverse=True)
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


class CollaborativeFilteringModel:
    def __init__(self):
        self.model = None
        self.user_avg = {}
        self.global_avg = 3.5
        self.is_trained = False

    def fit(self, interactions_df):
        score_col = next(
            (c for c in ["final_score", "skor_interaksi", "total_score"]
             if c in interactions_df.columns), None
        )
        if score_col is None:
            interactions_df = interactions_df.copy()
            interactions_df["score"] = 1.0
            score_col = "score"

        df = interactions_df[["user_id", "destination_id", score_col]].copy()
        df.columns = ["user_id", "destination_id", "rating"]
        df["rating"] = df["rating"].clip(0, 5).fillna(1.0)

        self.global_avg = df["rating"].mean()
        self.user_avg   = df.groupby("user_id")["rating"].mean().to_dict()
        self.is_trained = True
        self.df_train   = df
        return self

    def predict(self, user_id, destination_id):
        if not self.is_trained:
            return self.global_avg
        if SURPRISE_AVAILABLE and self.model:
            try:
                return self.model.predict(user_id, destination_id).est
            except Exception:
                pass
        return self.user_avg.get(user_id, self.global_avg)

    def recommend_for_user(self, user_id, all_destination_ids,
                           exclude_seen=True, top_n=10):
        if not self.is_trained:
            return []
        seen = set()
        if exclude_seen and hasattr(self, "df_train"):
            seen = set(
                self.df_train[self.df_train["user_id"] == user_id]["destination_id"].tolist()
            )
        candidates = [d for d in all_destination_ids if d not in seen]
        predictions = sorted(
            [(dst_id, self.predict(user_id, dst_id)) for dst_id in candidates],
            key=lambda x: x[1], reverse=True
        )
        return [{"destination_id": d, "cf_score": round(s, 4)}
                for d, s in predictions[:top_n]]


class HybridRecommender:
    def __init__(self, cf_model, cb_model, alpha_cf=0.6, alpha_cb=0.4):
        self.cf_model  = cf_model
        self.cb_model  = cb_model
        self.alpha_cf  = alpha_cf
        self.alpha_cb  = alpha_cb
        self.destinations = None
        self.all_dst_ids  = []
        self.user_interaction_counts = {}
        self.version    = "1.0.0"

    def fit(self, destinations_df, interactions_df):
        self.destinations = destinations_df.copy()
        dst_id_col = "destination_id" if "destination_id" in destinations_df.columns \
                     else destinations_df.columns[0]
        self.all_dst_ids = destinations_df[dst_id_col].tolist()
        if "user_id" in interactions_df.columns:
            self.user_interaction_counts = \
                interactions_df.groupby("user_id").size().to_dict()
        return self

    def _get_adaptive_weights(self, user_id):
        n = self.user_interaction_counts.get(user_id, 0)
        if n < 5:    return 0.2, 0.8
        elif n < 20: return 0.5, 0.5
        else:        return 0.7, 0.3

    def recommend(self, user_id, top_n=10, user_preferences=None):
        w_cf, w_cb = self._get_adaptive_weights(user_id)

        cf_recs    = self.cf_model.recommend_for_user(
            user_id, self.all_dst_ids, top_n=len(self.all_dst_ids)
        )
        cf_scores  = {r["destination_id"]: r["cf_score"] for r in cf_recs}

        cb_scores  = {}
        if user_preferences and "favorit_destinasi" in user_preferences:
            for fav_dst in user_preferences["favorit_destinasi"]:
                for s in self.cb_model.get_similar_destinations(fav_dst, top_n=20):
                    dst_id = s["destination_id"]
                    cb_scores[dst_id] = max(cb_scores.get(dst_id, 0),
                                            s["similarity_score"])

        def normalize(d):
            if not d: return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi == lo: return {k: 0.5 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        cf_norm = normalize(cf_scores)
        cb_norm = normalize(cb_scores)

        final_scores = {
            dst_id: w_cf * cf_norm.get(dst_id, 0) + w_cb * cb_norm.get(dst_id, 0)
            for dst_id in set(list(cf_norm) + list(cb_norm))
        }

        sorted_recs = sorted(final_scores.items(),
                             key=lambda x: x[1], reverse=True)[:top_n]

        dst_id_col = "destination_id" if "destination_id" in self.destinations.columns \
                     else self.destinations.columns[0]
        dst_lookup = self.destinations.set_index(dst_id_col).to_dict("index")

        return [
            {
                "destination_id": dst_id,
                "nama_destinasi": dst_lookup.get(dst_id, {}).get("nama_destinasi", ""),
                "kategori":       dst_lookup.get(dst_id, {}).get("kategori", ""),
                "provinsi":       dst_lookup.get(dst_id, {}).get("provinsi", ""),
                "rating":         dst_lookup.get(dst_id, {}).get("rating_rata_rata", 0),
                "harga_tiket":    dst_lookup.get(dst_id, {}).get("harga_tiket_dewasa", 0),
                "hybrid_score":   round(score, 4),
                "cf_score":       round(cf_norm.get(dst_id, 0), 4),
                "cb_score":       round(cb_norm.get(dst_id, 0), 4),
                "bobot_cf":       w_cf,
                "bobot_cb":       w_cb,
            }
            for dst_id, score in sorted_recs
        ]

    def predict(self, user_id):
        return self.recommend(user_id, top_n=10)


# ════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════

def _download(bucket, key, local_path):
    try:
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        raise RuntimeError(f"S3 error [{code}]: s3://{bucket}/{key}") from e


def load_model_and_metadata():
    if _CACHE["model"] is not None:
        logger.info("[Cache HIT] Model sudah di memori")
        return _CACHE["model"], _CACHE["metadata"]

    logger.info("[Cache MISS] Cold start — download dari S3")
    t0 = time.time()

    _download(MODEL_BUCKET, MODEL_KEY,    "/tmp/hybrid_model.pkl")
    _download(MODEL_BUCKET, METADATA_KEY, "/tmp/model_metadata.json")

    with open("/tmp/hybrid_model.pkl", "rb") as f:
        package = pickle.load(f)

    model = package["model"] if isinstance(package, dict) and "model" in package \
            else package

    with open("/tmp/model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    _CACHE["model"]    = model
    _CACHE["metadata"] = metadata

    logger.info(f"Model v{metadata.get('version','?')} siap | {round((time.time()-t0)*1000)}ms")
    return model, metadata


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def parse_body(event):
    body = event.get("body") or "{}"
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            raise ValueError("Request body bukan JSON yang valid")
    return body if isinstance(body, dict) else {}


def get_method(event):
    return (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod", "POST")
    ).upper()


_HEADERS = {
    "Content-Type":                 "application/json",
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Api-Key",
    "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
}

def ok(data, status=200):
    return {"statusCode": status, "headers": _HEADERS,
            "body": json.dumps(data, ensure_ascii=False)}

def err(msg, status=400):
    return {"statusCode": status, "headers": _HEADERS,
            "body": json.dumps({"error": msg}, ensure_ascii=False)}


# ════════════════════════════════════════════════════════════════
# MAIN HANDLER
# ════════════════════════════════════════════════════════════════

def lambda_handler(event, context):
    t_start = time.time()
    method  = get_method(event)
    logger.info(f"[{method}] {event.get('rawPath', '')}")

    if method == "OPTIONS":
        return ok({}, 204)

    if method == "GET":
        try:
            _, meta = load_model_and_metadata()
            return ok({
                "status":         "healthy",
                "model_version":  meta.get("version"),
                "trained_at":     meta.get("trained_at"),
                "n_users":        meta.get("n_users"),
                "n_destinations": meta.get("n_destinations"),
                "metrics":        meta.get("metrics"),
                "cache":          "warm" if _CACHE["model"] else "cold",
            })
        except Exception as e:
            return err(str(e), 503)

    if method != "POST":
        return err(f"Method {method} tidak didukung", 405)

    try:
        body = parse_body(event)
    except ValueError as e:
        return err(str(e), 400)

    user_id = str(body.get("user_id", "")).strip()
    if not user_id:
        return err("Field 'user_id' wajib diisi", 400)

    try:
        top_n = int(body.get("top_n", TOP_N_DEFAULT))
        if not (1 <= top_n <= 20):
            return err("top_n harus antara 1 dan 20", 400)
    except (TypeError, ValueError):
        return err("top_n harus berupa angka bulat", 400)

    try:
        model, meta = load_model_and_metadata()
        recommendations = model.recommend(user_id=user_id, top_n=top_n)
    except RuntimeError as e:
        logger.error(f"[Model Error] {e}")
        return err(str(e), 503)
    except Exception as e:
        import traceback
        logger.error(f"[Inference Error]\n{traceback.format_exc()}")
        return err(f"{type(e).__name__}: {str(e)}", 500)

    latency = round((time.time() - t_start) * 1000)
    logger.info(f"[OK] user_id={user_id} | hasil={len(recommendations)} | {latency}ms")

    return ok({
        "user_id":         user_id,
        "recommendations": recommendations,
        "count":           len(recommendations),
        "model_version":   meta.get("version", "unknown"),
        "latency_ms":      latency,
    })