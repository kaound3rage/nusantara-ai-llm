# NusantaraAI — Lambda Recommendation API

Sistem inference rekomendasi destinasi wisata menggunakan AWS Lambda + API Gateway.  
Model: Hybrid Collaborative Filtering + Content-Based (disimpan di S3).

---

## Struktur Project

```
lambda-recommender/
├── lambda_function.py   ← kode utama Lambda
├── requirements.txt     ← dependensi untuk Lambda Layer
├── iam_policy.json      ← IAM policy yang dibutuhkan Lambda
└── README.md
```

---

## Konfigurasi Lambda

| Parameter | Nilai |
|-----------|-------|
| Runtime | Python 3.11 |
| Handler | `lambda_function.lambda_handler` |
| Memory | 512 MB |
| Timeout | 30 detik |
| Architecture | x86_64 |

### Environment Variables

| Key | Contoh Nilai | Keterangan |
|-----|-------------|------------|
| `MODEL_BUCKET` | `nusantara-ai` | Nama S3 bucket |
| `MODEL_KEY` | `models/hybrid_model.pkl` | Path ke file model |
| `METADATA_KEY` | `models/model_metadata.json` | Path ke metadata |
| `TOP_N_DEFAULT` | `5` | Jumlah rekomendasi default |

---

## Step 1 — Buat Lambda Layer

Lambda Layer dibutuhkan karena `scikit-learn`, `scikit-surprise`, dan `pandas`
terlalu besar untuk di-zip langsung bersama kode.

Jalankan perintah ini di terminal SageMaker atau EC2 (**bukan** di dalam Lambda):

```bash
# Buat folder struktur layer
mkdir -p lambda-layer/python

# Install dependensi ke dalam folder layer
pip install \
    scikit-learn==1.4.2 \
    scikit-surprise==1.1.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    pyarrow==16.1.0 \
    --target lambda-layer/python \
    --only-binary=:all: \
    --quiet

# Zip
cd lambda-layer
zip -r ../nusantaraai-layer.zip python/ -q
cd ..

echo "Layer size: $(du -sh nusantaraai-layer.zip | cut -f1)"
```

Upload layer ke AWS:

```bash
aws lambda publish-layer-version \
    --layer-name nusantaraai-ml-deps \
    --description "scikit-learn, surprise, pandas, numpy" \
    --zip-file fileb://nusantaraai-layer.zip \
    --compatible-runtimes python3.11 \
    --region us-east-1
```

Catat `LayerVersionArn` dari output — dibutuhkan di Step 3.

---

## Step 2 — Buat IAM Role untuk Lambda

### 2a. Buat Role via AWS Console

1. Buka **IAM → Roles → Create role**
2. Trusted entity: **AWS service → Lambda**
3. Attach policy bawaan: `AWSLambdaBasicExecutionRole`
4. Beri nama: `NusantaraAI-Lambda-Role`
5. Setelah role dibuat, tambahkan inline policy dari file `iam_policy.json`

### 2b. Buat Role via CLI

```bash
# Buat role
aws iam create-role \
    --role-name NusantaraAI-Lambda-Role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach policy bawaan
aws iam attach-role-policy \
    --role-name NusantaraAI-Lambda-Role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Attach inline policy S3
aws iam put-role-policy \
    --role-name NusantaraAI-Lambda-Role \
    --policy-name S3ReadModel \
    --policy-document file://iam_policy.json
```

---

## Step 3 — Deploy Lambda Function

### Via AWS Console

1. Buka **Lambda → Create function**
2. Pilih **Author from scratch**
3. Isi:
   - Function name: `nusantaraai-recommender`
   - Runtime: **Python 3.11**
   - Architecture: x86_64
   - Execution role: pilih `NusantaraAI-Lambda-Role`
4. Klik **Create function**
5. Di tab **Code**, upload `lambda_function.py` (atau paste isinya langsung)
6. Di tab **Configuration → General configuration**:
   - Memory: **512 MB**
   - Timeout: **30 seconds**
7. Di tab **Configuration → Environment variables**, tambahkan:
   ```
   MODEL_BUCKET   = nusantara-ai
   MODEL_KEY      = models/hybrid_model.pkl
   METADATA_KEY   = models/model_metadata.json
   TOP_N_DEFAULT  = 5
   ```
8. Di tab **Layers**, klik **Add a layer** → pilih layer dari Step 1

### Via AWS CLI

```bash
# Zip kode Lambda
zip lambda_deployment.zip lambda_function.py

# Ambil Layer ARN dari Step 1
LAYER_ARN="arn:aws:lambda:us-east-1:ACCOUNT_ID:layer:nusantaraai-ml-deps:1"
ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/NusantaraAI-Lambda-Role"

# Buat Lambda function
aws lambda create-function \
    --function-name nusantaraai-recommender \
    --runtime python3.11 \
    --handler lambda_function.lambda_handler \
    --role "$ROLE_ARN" \
    --zip-file fileb://lambda_deployment.zip \
    --layers "$LAYER_ARN" \
    --memory-size 512 \
    --timeout 30 \
    --environment Variables="{
        MODEL_BUCKET=nusantara-ai,
        MODEL_KEY=models/hybrid_model.pkl,
        METADATA_KEY=models/model_metadata.json,
        TOP_N_DEFAULT=5
    }" \
    --region us-east-1

# Update kode saja (jika function sudah ada):
aws lambda update-function-code \
    --function-name nusantaraai-recommender \
    --zip-file fileb://lambda_deployment.zip
```

---

## Step 4 — Setup API Gateway

### Via AWS Console

1. Buka **API Gateway → Create API**
2. Pilih **HTTP API** (lebih murah & cepat dari REST API)
3. Klik **Add integration → Lambda** → pilih `nusantaraai-recommender`
4. API name: `nusantaraai-api`
5. Di bagian **Routes**, tambahkan:
   - `POST /recommend`
   - `GET /health`
   - `OPTIONS /{proxy+}` (untuk CORS)
6. Di bagian **CORS**, aktifkan dan set:
   - Allow origins: `https://domain-kamu.id`
   - Allow methods: `POST, GET, OPTIONS`
   - Allow headers: `Content-Type, X-Api-Key`
7. Deploy ke stage `prod`

Endpoint akhir akan berbentuk:
```
https://XXXXXX.execute-api.us-east-1.amazonaws.com/prod/recommend
```

### Via AWS CLI (HTTP API)

```bash
# Buat HTTP API
API_ID=$(aws apigatewayv2 create-api \
    --name nusantaraai-api \
    --protocol-type HTTP \
    --cors-configuration \
        AllowOrigins='["*"]',AllowMethods='["POST","GET","OPTIONS"]',AllowHeaders='["Content-Type","X-Api-Key"]' \
    --query "ApiId" --output text)

echo "API ID: $API_ID"

# Buat integrasi ke Lambda
LAMBDA_ARN="arn:aws:lambda:us-east-1:ACCOUNT_ID:function:nusantaraai-recommender"

INTEGRATION_ID=$(aws apigatewayv2 create-integration \
    --api-id "$API_ID" \
    --integration-type AWS_PROXY \
    --integration-uri "$LAMBDA_ARN" \
    --payload-format-version "2.0" \
    --query "IntegrationId" --output text)

# Buat route POST /recommend
aws apigatewayv2 create-route \
    --api-id "$API_ID" \
    --route-key "POST /recommend" \
    --target "integrations/$INTEGRATION_ID"

# Buat route GET /health
aws apigatewayv2 create-route \
    --api-id "$API_ID" \
    --route-key "GET /health" \
    --target "integrations/$INTEGRATION_ID"

# Deploy ke stage prod
aws apigatewayv2 create-stage \
    --api-id "$API_ID" \
    --stage-name prod \
    --auto-deploy

# Izinkan API Gateway invoke Lambda
aws lambda add-permission \
    --function-name nusantaraai-recommender \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:us-east-1:ACCOUNT_ID:$API_ID/*/*"

echo "Endpoint: https://$API_ID.execute-api.us-east-1.amazonaws.com/prod"
```

---

## Contoh Request & Response

### POST /recommend

**Request:**
```json
{
    "user_id": "USR00042",
    "top_n": 5
}
```

**Response 200 OK:**
```json
{
    "user_id": "USR00042",
    "recommendations": [
        {
            "destination_id": "DST0012",
            "nama_destinasi": "Pantai Parangtritis",
            "kategori": "Pantai",
            "provinsi": "DI Yogyakarta",
            "rating": 4.5,
            "harga_tiket": 10000,
            "hybrid_score": 0.8721
        },
        {
            "destination_id": "DST0007",
            "nama_destinasi": "Candi Borobudur",
            "kategori": "Budaya",
            "provinsi": "Jawa Tengah",
            "rating": 4.8,
            "harga_tiket": 50000,
            "hybrid_score": 0.8134
        },
        {
            "destination_id": "DST0021",
            "nama_destinasi": "Gunung Merapi",
            "kategori": "Alam",
            "provinsi": "Jawa Tengah",
            "rating": 4.6,
            "harga_tiket": 25000,
            "hybrid_score": 0.7892
        },
        {
            "destination_id": "DST0003",
            "nama_destinasi": "Kebun Raya Bogor",
            "kategori": "Alam",
            "provinsi": "Jawa Barat",
            "rating": 4.3,
            "harga_tiket": 30000,
            "hybrid_score": 0.7345
        },
        {
            "destination_id": "DST0018",
            "nama_destinasi": "Taman Nasional Bromo",
            "kategori": "Alam",
            "provinsi": "Jawa Timur",
            "rating": 4.7,
            "harga_tiket": 29000,
            "hybrid_score": 0.7102
        }
    ],
    "count": 5,
    "model_version": "1.0.0",
    "latency_ms": 38
}
```

### GET /health

**Response 200 OK:**
```json
{
    "status": "healthy",
    "model_version": "1.0.0",
    "trained_at": "2026-03-15T09:01:22.895246",
    "n_users": 2000,
    "n_destinations": 29,
    "metrics": {
        "precision_at_5": 0.0,
        "coverage": 0.5862,
        "users_evaluated": 50
    },
    "cache": "warm"
}
```

### Error Response (400)

```json
{
    "error": "Field 'user_id' wajib diisi"
}
```

---

## Monitoring dengan CloudWatch

Log Lambda otomatis masuk ke CloudWatch Log Group:  
`/aws/lambda/nusantaraai-recommender`

### Filter log yang berguna

```bash
# Lihat semua error dalam 1 jam terakhir
aws logs filter-log-events \
    --log-group-name /aws/lambda/nusantaraai-recommender \
    --filter-pattern "ERROR" \
    --start-time $(date -d '1 hour ago' +%s000)

# Lihat cold start
aws logs filter-log-events \
    --log-group-name /aws/lambda/nusantaraai-recommender \
    --filter-pattern "cold start"

# Lihat semua request
aws logs filter-log-events \
    --log-group-name /aws/lambda/nusantaraai-recommender \
    --filter-pattern "[OK]"
```

### CloudWatch Alarm (error rate tinggi)

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "NusantaraAI-Lambda-Errors" \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --dimensions Name=FunctionName,Value=nusantaraai-recommender \
    --statistic Sum \
    --period 60 \
    --threshold 5 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --alarm-actions "arn:aws:sns:us-east-1:ACCOUNT_ID:notifikasi-error"
```

---

## Test Cepat via curl

```bash
# Ganti URL dengan endpoint API Gateway kamu
API_URL="https://XXXXXX.execute-api.us-east-1.amazonaws.com/prod"

# Health check
curl "$API_URL/health"

# Rekomendasi
curl -X POST "$API_URL/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "USR00001", "top_n": 5}'
```

---

## Mitigasi Cold Start

Cold start terjadi saat container Lambda baru dinyalakan —  
model harus di-download dari S3 (~3–10 detik pertama kali).

Setelah itu, **warm invocation** langsung pakai cache di memori (~30–100 ms).

Opsi untuk mengurangi cold start:

| Opsi | Cara | Biaya |
|------|------|-------|
| Provisioned Concurrency | Lambda → Configuration → Concurrency | Berbayar |
| Scheduled ping | EventBridge rule setiap 5 menit kirim GET /health | Hampir gratis |
| Increase memory | Memory lebih besar = CPU lebih cepat = cold start lebih pendek | Sedikit lebih mahal |

Rekomendasi untuk produksi: aktifkan **scheduled ping** dulu (gratis),
upgrade ke Provisioned Concurrency jika traffic sudah ramai.
