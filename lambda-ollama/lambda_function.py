"""
NusantaraAI - Lambda Recommendation Function
Runtime : Python 3.11
Handler : lambda_function.lambda_handler
"""

import os
import json
import time
import logging
import pickle
import boto3
from botocore.exceptions import ClientError

# ─── Logger (output masuk CloudWatch otomatis) ────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Environment Variables ───────────────────────────────────────────────────
MODEL_BUCKET   = os.environ.get("MODEL_BUCKET",   "nusantara-ai")
MODEL_KEY      = os.environ.get("MODEL_KEY",      "models/hybrid_model.pkl")
METADATA_KEY   = os.environ.get("METADATA_KEY",   "models/model_metadata.json")
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
TOP_N_DEFAULT  = int(os.environ.get("TOP_N_DEFAULT", "5"))

# ─── S3 Client ───────────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=AWS_REGION)

# ─── In-Memory Cache (warm invocation tidak download ulang) ──────────────────
_CACHE = {
    "model":    None,   # HybridRecommender instance
    "metadata": None,   # dict dari model_metadata.json
}


# ════════════════════════════════════════════════════════════════
# 1. MODEL LOADER
# ════════════════════════════════════════════════════════════════

def _download_from_s3(bucket: str, key: str, local_path: str) -> None:
    """Download satu file dari S3 ke /tmp."""
    try:
        logger.info(f"Downloading s3://{bucket}/{key} → {local_path}")
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        raise RuntimeError(f"S3 download gagal [{code}]: s3://{bucket}/{key}") from e


def load_model_and_metadata() -> tuple:
    """
    Load model dan metadata dari S3.
    Hasil di-cache di _CACHE agar warm invocation tidak download ulang.
    Return: (model, metadata_dict)
    """
    if _CACHE["model"] is not None:
        logger.info("[Cache HIT] Model sudah tersedia di memori")
        return _CACHE["model"], _CACHE["metadata"]

    logger.info("[Cache MISS] Cold start — download dari S3")
    t0 = time.time()

    # Download model
    local_model = "/tmp/hybrid_model.pkl"
    _download_from_s3(MODEL_BUCKET, MODEL_KEY, local_model)

    # Download metadata
    local_meta = "/tmp/model_metadata.json"
    _download_from_s3(MODEL_BUCKET, METADATA_KEY, local_meta)

    # Load model
    with open(local_model, "rb") as f:
        package = pickle.load(f)

    # package bisa berupa HybridRecommender langsung,
    # atau dict {"model": ..., "metadata": ...} dari train_model.py
    if isinstance(package, dict) and "model" in package:
        model = package["model"]
    else:
        model = package

    # Load metadata
    with open(local_meta, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Simpan ke cache
    _CACHE["model"]    = model
    _CACHE["metadata"] = metadata

    elapsed = round((time.time() - t0) * 1000)
    logger.info(f"Model v{metadata.get('version', '?')} siap | cold start: {elapsed} ms")

    return model, metadata


# ════════════════════════════════════════════════════════════════
# 2. INFERENCE
# ════════════════════════════════════════════════════════════════

def generate_recommendation(user_id: str, top_n: int = 5) -> list:
    """
    Panggil model.recommend() dan kembalikan list rekomendasi.
    Setiap item berisi nama destinasi dan skor hybrid.
    """
    model, _ = load_model_and_metadata()

    results = model.recommend(user_id=user_id, top_n=top_n)

    # Format output — hanya field yang relevan untuk response API
    recommendations = []
    for r in results:
        recommendations.append({
            "destination_id":  r.get("destination_id", ""),
            "nama_destinasi":  r.get("nama_destinasi", ""),
            "kategori":        r.get("kategori", ""),
            "provinsi":        r.get("provinsi", ""),
            "rating":          r.get("rating", 0),
            "harga_tiket":     r.get("harga_tiket", 0),
            "hybrid_score":    r.get("hybrid_score", 0),
        })

    return recommendations


# ════════════════════════════════════════════════════════════════
# 3. REQUEST PARSER
# ════════════════════════════════════════════════════════════════

def parse_body(event: dict) -> dict:
    """Ambil body JSON dari event API Gateway (v1 maupun v2)."""
    body = event.get("body") or "{}"
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            raise ValueError("Request body bukan JSON yang valid")
    return body if isinstance(body, dict) else {}


def get_http_method(event: dict) -> str:
    """Deteksi HTTP method dari payload API Gateway v1 atau v2."""
    # HTTP API v2
    method = (event.get("requestContext", {})
                   .get("http", {})
                   .get("method", ""))
    # REST API v1
    if not method:
        method = event.get("httpMethod", "POST")
    return method.upper()


# ════════════════════════════════════════════════════════════════
# 4. RESPONSE BUILDER
# ════════════════════════════════════════════════════════════════

_CORS_HEADERS = {
    "Content-Type":                 "application/json",
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Api-Key,Authorization",
    "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
}


def ok(data: dict, status: int = 200) -> dict:
    return {
        "statusCode": status,
        "headers":    _CORS_HEADERS,
        "body":       json.dumps(data, ensure_ascii=False),
    }


def err(message: str, status: int = 400) -> dict:
    return {
        "statusCode": status,
        "headers":    _CORS_HEADERS,
        "body":       json.dumps({"error": message}, ensure_ascii=False),
    }


# ════════════════════════════════════════════════════════════════
# 5. MAIN HANDLER
# ════════════════════════════════════════════════════════════════

def lambda_handler(event: dict, context) -> dict:
    """
    Entry point Lambda.

    Route:
      OPTIONS  *           → CORS preflight
      GET      /health     → status model + metadata
      POST     /recommend  → rekomendasi destinasi
    """
    t_start = time.time()
    method  = get_http_method(event)

    logger.info(f"[{method}] event keys: {list(event.keys())}")

    # ── CORS preflight ──────────────────────────────────────────
    if method == "OPTIONS":
        return ok({}, 204)

    # ── Health check ────────────────────────────────────────────
    if method == "GET":
        try:
            _, metadata = load_model_and_metadata()
            return ok({
                "status":       "healthy",
                "model_version": metadata.get("version"),
                "trained_at":   metadata.get("trained_at"),
                "n_users":      metadata.get("n_users"),
                "n_destinations": metadata.get("n_destinations"),
                "metrics":      metadata.get("metrics"),
                "cache":        "warm" if _CACHE["model"] else "cold",
            })
        except Exception as e:
            logger.error(f"Health check gagal: {e}")
            return err("Model tidak tersedia", 503)

    # ── Rekomendasi ─────────────────────────────────────────────
    if method != "POST":
        return err(f"Method {method} tidak didukung", 405)

    # Parse body
    try:
        body = parse_body(event)
    except ValueError as e:
        return err(str(e), 400)

    # Validasi user_id
    user_id = str(body.get("user_id", "")).strip()
    if not user_id:
        return err("Field 'user_id' wajib diisi", 400)

    # top_n opsional, default 5
    try:
        top_n = int(body.get("top_n", TOP_N_DEFAULT))
        if not (1 <= top_n <= 20):
            return err("top_n harus antara 1 dan 20", 400)
    except (TypeError, ValueError):
        return err("top_n harus berupa angka bulat", 400)

    # Generate rekomendasi
    try:
        recommendations = generate_recommendation(user_id, top_n)
    except RuntimeError as e:
        logger.error(f"[Model Error] {e}")
        return err("Model tidak tersedia, coba beberapa saat lagi", 503)
    except Exception as e:
        logger.exception(f"[Inference Error] user_id={user_id} — {e}")
        return err("Terjadi kesalahan internal", 500)

    latency_ms = round((time.time() - t_start) * 1000)
    logger.info(
        f"[OK] user_id={user_id} | "
        f"rekomendasi={len(recommendations)} | "
        f"latency={latency_ms}ms"
    )

    _, metadata = load_model_and_metadata()

    return ok({
        "user_id":         user_id,
        "recommendations": recommendations,
        "count":           len(recommendations),
        "model_version":   metadata.get("version", "unknown"),
        "latency_ms":      latency_ms,
    })
