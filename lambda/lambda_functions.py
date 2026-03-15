"""
====================================================
NusantaraAI - Lambda: ML Prediction + LLM Chat
====================================================
File ini berisi 2 Lambda function:
  1. lambda_prediction  → /recommendations endpoint
  2. lambda_llm_chat    → /chat endpoint (Ollama)

Deploy masing-masing sebagai Lambda terpisah.
Runtime: Python 3.12
====================================================
"""

# ════════════════════════════════════════════════════════════════
# LAMBDA 1: nusantaraai-ml-prediction
# Handler: lambda_prediction.lambda_handler
# ════════════════════════════════════════════════════════════════

import json
import os
import time
import pickle
import boto3
import logging
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Cache model di memory (hindari download ulang tiap request) ───
_model_cache = None
_model_loaded_at = None
MODEL_CACHE_TTL = 3600  # refresh model tiap 1 jam

# ─── AWS Clients ───
_s3_client = None
_dynamo_resource = None

def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client

def get_dynamo():
    global _dynamo_resource
    if _dynamo_resource is None:
        _dynamo_resource = boto3.resource("dynamodb")
    return _dynamo_resource

def load_model():
    """Load model dari S3 dengan caching."""
    global _model_cache, _model_loaded_at

    now = time.time()
    if (_model_cache is not None and _model_loaded_at is not None and
            now - _model_loaded_at < MODEL_CACHE_TTL):
        return _model_cache

    bucket = os.environ.get("S3_BUCKET", "nusantara-ai")
    key    = os.environ.get("MODEL_KEY", "models/hybrid_model.pkl")
    local  = "/tmp/hybrid_model.pkl"

    logger.info(f"Downloading model dari s3://{bucket}/{key}")
    get_s3().download_file(bucket, key, local)

    with open(local, "rb") as f:
        pkg = pickle.load(f)

    _model_cache   = pkg
    _model_loaded_at = now
    logger.info("Model berhasil dimuat dan dicache")
    return pkg


def lambda_handler(event, context):
    """
    Handler Lambda untuk rekomendasi destinasi wisata.

    Request body:
    {
        "user_id": "USR00001",
        "top_n": 10,
        "preferences": {
            "kategori": ["Pantai", "Bahari"],
            "max_budget": 200000,
            "provinsi": "Bali"
        }
    }
    """
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,x-api-key",
            },
            "body": "",
        }

    start_time = time.time()

    try:
        # Parse request body
        body = {}
        if event.get("body"):
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]

        user_id = body.get("user_id")
        top_n   = min(int(body.get("top_n", 10)), 20)  # max 20
        prefs   = body.get("preferences", {})

        if not user_id:
            return _error_response(400, "user_id wajib diisi")

        # Load model
        pkg = load_model()
        model = pkg.get("model")

        if model is None:
            return _error_response(500, "Model tidak tersedia")

        # Dapatkan rekomendasi
        recommendations = model.recommend(
            user_id=user_id,
            top_n=top_n,
            user_preferences=prefs if prefs else None
        )

        elapsed = round((time.time() - start_time) * 1000, 2)

        result = {
            "user_id": user_id,
            "recommendations": recommendations,
            "total": len(recommendations),
            "model_version": pkg.get("metadata", {}).get("version", "1.0.0"),
            "latency_ms": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Simpan ke DynamoDB (log & audit)
        _save_to_dynamo(user_id, result)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result, ensure_ascii=False),
        }

    except Exception as e:
        logger.exception(f"Error dalam prediction: {e}")
        return _error_response(500, f"Internal error: {str(e)}")


def _save_to_dynamo(user_id: str, result: dict):
    """Simpan hasil prediksi ke DynamoDB MLPredictionResults."""
    try:
        table = get_dynamo().Table("MLPredictionResults")
        prediction_id = f"{user_id}-{int(time.time())}"
        table.put_item(Item={
            "prediction_id": prediction_id,
            "user_id": user_id,
            "recommendations": json.dumps(result["recommendations"][:5]),
            "timestamp": result["timestamp"],
            "latency_ms": str(result["latency_ms"]),
            "ttl": int(time.time()) + 86400,  # 24 jam
        })
        logger.info(f"Hasil disimpan ke DynamoDB: {prediction_id}")
    except Exception as e:
        logger.warning(f"Gagal simpan ke DynamoDB: {e}")


def _error_response(status_code: int, message: str) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"error": message}, ensure_ascii=False),
    }


# ════════════════════════════════════════════════════════════════
# LAMBDA 2: nusantaraai-llm-chat
# Handler: lambda_llm_chat.lambda_handler
# Deskripsi: Chat dengan LLM Ollama untuk saran wisata
# ════════════════════════════════════════════════════════════════

import urllib.request
import urllib.parse

_chat_dynamo = None
_recommendation_model_cache = None

def get_chat_dynamo():
    global _chat_dynamo
    if _chat_dynamo is None:
        _chat_dynamo = boto3.resource("dynamodb")
    return _chat_dynamo


def call_ollama(prompt: str, model: str = "llama2") -> str:
    """Panggil Ollama API di EC2."""
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    endpoint   = f"{ollama_url}/api/generate"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 500,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "Maaf, tidak ada respons dari LLM.")
    except Exception as e:
        logger.error(f"Error memanggil Ollama: {e}")
        return f"Maaf, terjadi kesalahan saat menghubungi asisten AI: {str(e)}"


def build_travel_prompt(user_message: str, recommendations: list,
                        session_history: list) -> str:
    """
    Bangun prompt untuk LLM dengan konteks wisata Indonesia.
    """
    # Ringkasan rekomendasi
    rec_text = ""
    if recommendations:
        rec_text = "\n\nRekomendasi destinasi wisata untuk user ini:\n"
        for i, r in enumerate(recommendations[:5], 1):
            rec_text += (
                f"{i}. {r.get('nama_destinasi','')}"
                f" ({r.get('kategori','')}, {r.get('provinsi','')})"
                f" - Rating: {r.get('rating','')}"
                f" - Harga: Rp{r.get('harga_tiket',0):,}\n"
            )

    # Riwayat percakapan (ringkas)
    history_text = ""
    if session_history:
        recent = session_history[-4:]  # 2 turn terakhir
        for h in recent:
            role = "Pengguna" if h["role"] == "user" else "Asisten"
            history_text += f"{role}: {h['content'][:200]}\n"

    prompt = f"""Kamu adalah pemandu wisata Indonesia bernama NusantaraAI yang ramah dan berpengetahuan luas tentang destinasi wisata di seluruh Indonesia. Jawab selalu dalam Bahasa Indonesia yang hangat dan informatif.
{rec_text}
Riwayat percakapan:
{history_text}
Pengguna: {user_message}
Asisten:"""

    return prompt


def lambda_handler_llm(event, context):
    """
    Handler Lambda untuk chat LLM Ollama.

    Request body:
    {
        "session_id": "SES123456",
        "user_id": "USR00001",
        "message": "Rekomendasikan pantai bagus di Bali",
        "include_recommendations": true
    }
    """
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": "",
        }

    try:
        body = {}
        if event.get("body"):
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]

        session_id = body.get("session_id", f"SES{int(time.time())}")
        user_id    = body.get("user_id", "anonymous")
        message    = body.get("message", "").strip()
        include_recs = body.get("include_recommendations", True)

        if not message:
            return _error_response(400, "message wajib diisi")

        # Load session history dari DynamoDB
        session_history = _load_session_history(session_id)

        # Ambil rekomendasi ML jika diminta
        recommendations = []
        if include_recs and user_id != "anonymous":
            try:
                pkg = load_model()
                if pkg and pkg.get("model"):
                    recommendations = pkg["model"].recommend(user_id, top_n=5)
            except Exception as e:
                logger.warning(f"Gagal load rekomendasi: {e}")

        # Bangun prompt dan panggil Ollama
        prompt    = build_travel_prompt(message, recommendations, session_history)
        llm_reply = call_ollama(prompt)

        # Update session history
        session_history.append({"role": "user",      "content": message})
        session_history.append({"role": "assistant", "content": llm_reply})
        _save_session_history(session_id, user_id, session_history)

        result = {
            "session_id": session_id,
            "user_id": user_id,
            "reply": llm_reply,
            "recommendations_used": len(recommendations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result, ensure_ascii=False),
        }

    except Exception as e:
        logger.exception(f"Error dalam LLM chat: {e}")
        return _error_response(500, str(e))


def _load_session_history(session_id: str) -> list:
    """Ambil riwayat chat dari DynamoDB."""
    try:
        table = get_chat_dynamo().Table("LLMChatSessions")
        resp = table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": session_id},
            ScanIndexForward=True,
            Limit=20
        )
        history = []
        for item in resp.get("Items", []):
            history.append({
                "role": item.get("role", "user"),
                "content": item.get("content", "")
            })
        return history
    except Exception as e:
        logger.warning(f"Gagal load session: {e}")
        return []


def _save_session_history(session_id: str, user_id: str, history: list):
    """Simpan 2 pesan terakhir ke DynamoDB."""
    try:
        table = get_chat_dynamo().Table("LLMChatSessions")
        for turn in history[-2:]:
            ts = datetime.now(timezone.utc).isoformat()
            table.put_item(Item={
                "session_id": session_id,
                "timestamp": ts,
                "user_id": user_id,
                "role": turn["role"],
                "content": turn["content"][:2000],
                "ttl": int(time.time()) + 86400 * 7,  # 7 hari
            })
    except Exception as e:
        logger.warning(f"Gagal simpan session: {e}")
