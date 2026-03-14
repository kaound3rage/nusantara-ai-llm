"""
====================================================
NusantaraAI - Flask Application
====================================================
Aplikasi web untuk sistem rekomendasi wisata Indonesia.
Berjalan di Docker container di ECS Fargate.

Endpoints:
  GET  /health              - Health check
  GET  /                    - Info aplikasi
  POST /api/recommend       - Rekomendasi via ML model
  POST /api/chat            - Chat dengan LLM Ollama
  GET  /api/destinations    - Daftar destinasi
  POST /api/track           - Track interaksi user
====================================================
"""

import os
import json
import time
import pickle
import boto3
import logging
import requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from functools import wraps

# ─── Setup ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Konfigurasi dari environment variables
S3_BUCKET   = os.environ.get("S3_BUCKET",   "nusantaraai-ml-jawatengah-namakamu")
OLLAMA_URL  = os.environ.get("OLLAMA_URL",  "http://localhost:11434")
AWS_REGION  = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
MODEL_KEY   = os.environ.get("MODEL_KEY",   "models/hybrid_model.pkl")

# AWS clients
s3       = boto3.client("s3",       region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# Model cache
_model_cache = None
_model_loaded_at = None

# ─── Helper: Load Model ───────────────────────────────────────────────────────
def get_model():
    """Load model dari S3 dengan caching 1 jam."""
    global _model_cache, _model_loaded_at
    now = time.time()
    if _model_cache and _model_loaded_at and (now - _model_loaded_at < 3600):
        return _model_cache
    try:
        local = "/tmp/hybrid_model.pkl"
        s3.download_file(S3_BUCKET, MODEL_KEY, local)
        with open(local, "rb") as f:
            _model_cache = pickle.load(f)
        _model_loaded_at = now
        logger.info("Model berhasil dimuat dari S3")
    except Exception as e:
        logger.error(f"Gagal load model: {e}")
        _model_cache = None
    return _model_cache

# ─── Helper: Simpan interaksi ke DynamoDB ─────────────────────────────────────
def save_interaction(user_id: str, destination_id: str, tipe: str, skor: float):
    try:
        table = dynamodb.Table("UserInteractions")
        ts = datetime.now(timezone.utc).isoformat()
        table.put_item(Item={
            "user_id": user_id,
            "timestamp": ts,
            "destination_id": destination_id,
            "tipe_interaksi": tipe,
            "skor_interaksi": str(skor),
            "platform": "WebApp",
            "ttl": int(time.time()) + 86400 * 30,
        })
    except Exception as e:
        logger.warning(f"Gagal simpan interaksi: {e}")

# ─── Decorator: tambah CORS header ───────────────────────────────────────────
def cors_headers(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "OPTIONS":
            resp = Response("")
            resp.headers["Access-Control-Allow-Origin"]  = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            return resp
        result = f(*args, **kwargs)
        if isinstance(result, tuple):
            response, code = result[0], result[1]
        else:
            response, code = result, 200
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, code
    return decorated

# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Health check endpoint untuk ALB/ECS."""
    model_ok = get_model() is not None
    return jsonify({
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "ollama_url": OLLAMA_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200

@app.route("/")
def index():
    """Info aplikasi."""
    return jsonify({
        "app": "NusantaraAI",
        "version": "1.0.0",
        "description": "Sistem Rekomendasi Wisata Indonesia",
        "endpoints": {
            "POST /api/recommend": "Rekomendasi destinasi wisata",
            "POST /api/chat": "Chat dengan asisten wisata (LLM)",
            "GET /api/destinations": "Daftar semua destinasi",
            "POST /api/track": "Track interaksi user",
        }
    })

@app.route("/api/recommend", methods=["POST", "OPTIONS"])
@cors_headers
def recommend():
    """
    Endpoint rekomendasi destinasi wisata.
    Body: {"user_id": "USR00001", "top_n": 10, "preferences": {...}}
    """
    start = time.time()
    data = request.get_json() or {}
    user_id = data.get("user_id")
    top_n   = min(int(data.get("top_n", 10)), 20)
    prefs   = data.get("preferences", {})

    if not user_id:
        return jsonify({"error": "user_id wajib diisi"}), 400

    pkg = get_model()
    if not pkg or not pkg.get("model"):
        return jsonify({"error": "Model belum tersedia"}), 503

    try:
        recs = pkg["model"].recommend(user_id, top_n=top_n,
                                       user_preferences=prefs or None)
        elapsed = round((time.time() - start) * 1000, 2)
        return jsonify({
            "user_id": user_id,
            "recommendations": recs,
            "total": len(recs),
            "latency_ms": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST", "OPTIONS"])
@cors_headers
def chat():
    """
    Endpoint chat dengan LLM Ollama.
    Body: {"user_id": "USR00001", "message": "Rekomendasikan pantai di Bali"}
    """
    data       = request.get_json() or {}
    user_id    = data.get("user_id", "anonymous")
    message    = data.get("message", "").strip()
    session_id = data.get("session_id", f"SES{int(time.time())}")

    if not message:
        return jsonify({"error": "message wajib diisi"}), 400

    # Ambil rekomendasi ML untuk konteks
    recs = []
    pkg = get_model()
    if pkg and pkg.get("model") and user_id != "anonymous":
        try:
            recs = pkg["model"].recommend(user_id, top_n=5)
        except Exception:
            pass

    # Bangun prompt
    rec_context = ""
    if recs:
        rec_context = "\n\nRekomendasi untuk user ini:\n"
        for i, r in enumerate(recs[:3], 1):
            rec_context += f"{i}. {r.get('nama_destinasi','')} ({r.get('kategori','')})\n"

    prompt = f"""Kamu adalah pemandu wisata Indonesia bernama NusantaraAI. 
Jawab dengan ramah dan informatif dalam Bahasa Indonesia.{rec_context}

Pengguna: {message}
Asisten:"""

    # Panggil Ollama
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 400}},
            timeout=30
        )
        reply = resp.json().get("response", "Maaf, tidak ada respons dari AI.")
    except Exception as e:
        reply = f"Maaf, asisten AI sedang tidak tersedia: {str(e)}"

    return jsonify({
        "session_id": session_id,
        "user_id": user_id,
        "reply": reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/destinations", methods=["GET"])
def get_destinations():
    """Ambil daftar destinasi dari model."""
    pkg = get_model()
    if not pkg or not pkg.get("model"):
        return jsonify({"error": "Model tidak tersedia"}), 503

    model = pkg["model"]
    if hasattr(model, "destinations") and model.destinations is not None:
        cols = ["destination_id", "nama_destinasi", "kategori", "provinsi",
                "rating_rata_rata", "harga_tiket_dewasa", "durasi_kunjungan_jam"]
        available_cols = [c for c in cols if c in model.destinations.columns]
        destinations = model.destinations[available_cols].fillna("").to_dict("records")
    else:
        destinations = []

    return jsonify({
        "destinations": destinations,
        "total": len(destinations),
    })

@app.route("/api/track", methods=["POST", "OPTIONS"])
@cors_headers
def track_interaction():
    """
    Track interaksi user dengan destinasi.
    Body: {"user_id":"USR001","destination_id":"DST001","tipe":"view","skor":0.2}
    """
    data = request.get_json() or {}
    user_id    = data.get("user_id")
    dst_id     = data.get("destination_id")
    tipe       = data.get("tipe", "view")
    skor       = float(data.get("skor", 0.2))

    if not user_id or not dst_id:
        return jsonify({"error": "user_id dan destination_id wajib"}), 400

    save_interaction(user_id, dst_id, tipe, skor)
    return jsonify({"status": "ok", "message": "Interaksi berhasil disimpan"})

# ─── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint tidak ditemukan"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
