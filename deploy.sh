#!/bin/bash
# ====================================================
# NusantaraAI - Panduan Deploy Lengkap
# LKS Cloud Computing 2025
# ====================================================
# GANTI variabel ini sesuai data kamu:
NAMA="namakamu"
PROVINSI="jawatengah"
ACCOUNT_ID="123456789012"     # AWS Account ID kamu
REGION="us-east-1"
EMAIL="pesertamu@email.com"
KEYPAIR="nusantaraai-key"
IP_WHITELIST="203.0.113.0/24" # IP yang boleh akses logs/

BUCKET="nusantaraai-ml-${PROVINSI}-${NAMA}"
ECR_REPO="nusantaraai-ecr-${PROVINSI}-${NAMA}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "======================================================"
echo " NusantaraAI - Deploy Script"
echo " Bucket : $BUCKET"
echo " Region : $REGION"
echo "======================================================"

# ============================================================
# LANGKAH 0: Pre-requisite
# ============================================================
echo ""
echo "=== LANGKAH 0: Install Tools ==="
pip3 install boto3 pandas numpy scikit-learn scikit-surprise awscli
aws configure  # masukkan Access Key, Secret, Region us-east-1, format json

# ============================================================
# LANGKAH 1: Generate Dataset
# ============================================================
echo ""
echo "=== LANGKAH 1: Generate Dataset ==="
cd dataset/
python3 generate_dataset.py

echo "  Dataset berhasil dibuat:"
ls -lh dataset/

# ============================================================
# LANGKAH 2: Deploy CloudFormation - Networking
# ============================================================
echo ""
echo "=== LANGKAH 2: Deploy Networking Stack ==="
aws cloudformation deploy \
  --template-file cloudformation/01-networking.yaml \
  --stack-name nusantaraai-networking \
  --parameter-overrides \
    NamaPeserta=${NAMA} \
    NamaProvinsi=${PROVINSI} \
  --capabilities CAPABILITY_IAM \
  --region ${REGION}

echo "  ✓ Networking Stack deployed"

# Ambil output VPC
VPC_ID=$(aws cloudformation describe-stacks \
  --stack-name nusantaraai-networking \
  --query "Stacks[0].Outputs[?OutputKey=='VPCId'].OutputValue" \
  --output text --region ${REGION})
echo "  VPC ID: $VPC_ID"

# ============================================================
# LANGKAH 3: Buat Key Pair
# ============================================================
echo ""
echo "=== LANGKAH 3: Buat Key Pair EC2 ==="
aws ec2 create-key-pair \
  --key-name ${KEYPAIR} \
  --query "KeyMaterial" \
  --output text \
  --region ${REGION} > ${KEYPAIR}.pem
chmod 400 ${KEYPAIR}.pem
echo "  ✓ Key pair tersimpan di ${KEYPAIR}.pem"

# ============================================================
# LANGKAH 4: Deploy CloudFormation - Storage & Compute
# ============================================================
echo ""
echo "=== LANGKAH 4: Deploy Storage & Compute Stack ==="
aws cloudformation deploy \
  --template-file cloudformation/02-storage-compute.yaml \
  --stack-name nusantaraai-storage \
  --parameter-overrides \
    NamaPeserta=${NAMA} \
    NamaProvinsi=${PROVINSI} \
    IPWhitelistLogs=${IP_WHITELIST} \
    KeyPairName=${KEYPAIR} \
  --capabilities CAPABILITY_IAM \
  --region ${REGION}

echo "  ✓ Storage & Compute Stack deployed"

# Ambil IP Ollama
OLLAMA_IP=$(aws cloudformation describe-stacks \
  --stack-name nusantaraai-storage \
  --query "Stacks[0].Outputs[?OutputKey=='OllamaPublicIP'].OutputValue" \
  --output text --region ${REGION})
echo "  Ollama EC2 IP: $OLLAMA_IP"

# ============================================================
# LANGKAH 5: Upload Dataset ke S3
# ============================================================
echo ""
echo "=== LANGKAH 5: Upload Dataset ke S3 ==="

# Buat folder struktur
aws s3api put-object --bucket ${BUCKET} --key "raw-data/user-profiles/"
aws s3api put-object --bucket ${BUCKET} --key "raw-data/destination-catalog/"
aws s3api put-object --bucket ${BUCKET} --key "raw-data/user-interactions/"
aws s3api put-object --bucket ${BUCKET} --key "raw-data/transaction-history/"
aws s3api put-object --bucket ${BUCKET} --key "processed-data/"
aws s3api put-object --bucket ${BUCKET} --key "models/"
aws s3api put-object --bucket ${BUCKET} --key "logs/"
aws s3api put-object --bucket ${BUCKET} --key "scripts/"

# Upload dataset
aws s3 cp dataset/user_profiles.csv       s3://${BUCKET}/raw-data/user-profiles/
aws s3 cp dataset/destination_catalog.csv s3://${BUCKET}/raw-data/destination-catalog/
aws s3 cp dataset/user_interactions.csv   s3://${BUCKET}/raw-data/user-interactions/
aws s3 cp dataset/transaction_history.csv s3://${BUCKET}/raw-data/transaction-history/

# Upload scripts
aws s3 cp glue/etl_script.py       s3://${BUCKET}/scripts/
aws s3 cp sagemaker/train_model.py s3://${BUCKET}/scripts/

echo "  ✓ Dataset & scripts diupload ke S3"

# ============================================================
# LANGKAH 6: Deploy Services Stack (Glue, Lambda, API GW, SNS, Step Functions)
# ============================================================
echo ""
echo "=== LANGKAH 6: Deploy Services Stack ==="

# Buat zip Lambda
cd lambda/
zip prediction.zip lambda_functions.py
zip llm_chat.zip lambda_functions.py
aws s3 cp prediction.zip s3://${BUCKET}/lambda/
aws s3 cp llm_chat.zip   s3://${BUCKET}/lambda/
cd ..

aws cloudformation deploy \
  --template-file cloudformation/03-services.yaml \
  --stack-name nusantaraai-services \
  --parameter-overrides \
    NamaPeserta=${NAMA} \
    NamaProvinsi=${PROVINSI} \
    OllamaIP=${OLLAMA_IP} \
    EmailNotifikasi=${EMAIL} \
  --capabilities CAPABILITY_IAM \
  --region ${REGION}

echo "  ✓ Services Stack deployed"

# Ambil API endpoint
API_URL=$(aws cloudformation describe-stacks \
  --stack-name nusantaraai-services \
  --query "Stacks[0].Outputs[?OutputKey=='APIEndpoint'].OutputValue" \
  --output text --region ${REGION})
echo "  API URL: $API_URL"

# ============================================================
# LANGKAH 7: Jalankan Glue Crawler & ETL
# ============================================================
echo ""
echo "=== LANGKAH 7: Jalankan Glue Crawler ==="
aws glue start-crawler --name nusantaraai-crawler --region ${REGION}
echo "  Crawler berjalan... tunggu 2-3 menit"
sleep 120

# Cek status crawler
CRAWLER_STATUS=$(aws glue get-crawler \
  --name nusantaraai-crawler \
  --query "Crawler.State" \
  --output text --region ${REGION})
echo "  Status Crawler: $CRAWLER_STATUS"

echo ""
echo "=== Jalankan Glue ETL Job ==="
JOB_RUN_ID=$(aws glue start-job-run \
  --job-name nusantaraai-etl-job \
  --region ${REGION} \
  --query "JobRunId" \
  --output text)
echo "  ETL Job Run ID: $JOB_RUN_ID"
echo "  Tunggu 5-10 menit untuk ETL selesai..."

# Monitor ETL
while true; do
  STATUS=$(aws glue get-job-run \
    --job-name nusantaraai-etl-job \
    --run-id ${JOB_RUN_ID} \
    --query "JobRun.JobRunState" \
    --output text --region ${REGION})
  echo "  ETL Status: $STATUS"
  if [ "$STATUS" == "SUCCEEDED" ] || [ "$STATUS" == "FAILED" ]; then
    break
  fi
  sleep 30
done

# ============================================================
# LANGKAH 8: Training Model di EC2 (atau SageMaker Notebook)
# ============================================================
echo ""
echo "=== LANGKAH 8: Training Model ==="
echo "  Pilihan A: Jalankan di SageMaker Notebook"
echo "    1. Buka SageMaker Console"
echo "    2. Buka notebook: nusantaraai-ml-${PROVINSI}-${NAMA}"
echo "    3. Upload file sagemaker/train_model.py"
echo "    4. Jalankan script"
echo ""
echo "  Pilihan B: Jalankan lokal (untuk testing)"
echo "    export S3_BUCKET=${BUCKET}"
echo "    python3 sagemaker/train_model.py"

# Quick training lokal untuk testing
export S3_BUCKET=${BUCKET}
python3 sagemaker/train_model.py
echo "  ✓ Model diupload ke s3://${BUCKET}/models/hybrid_model.pkl"

# ============================================================
# LANGKAH 9: Build & Push Docker Image
# ============================================================
echo ""
echo "=== LANGKAH 9: Build & Push Docker Image ==="

# Buat ECR Repository
aws ecr create-repository \
  --repository-name ${ECR_REPO} \
  --region ${REGION} \
  --image-scanning-configuration scanOnPush=true \
  2>/dev/null || echo "  ECR Repository sudah ada"

# Login ke ECR
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build image
cd docker/
docker build -t ${ECR_REPO}:latest .

# Tag image
docker tag ${ECR_REPO}:latest ${ECR_URI}:latest

# Push ke ECR
docker push ${ECR_URI}:latest
cd ..

echo "  ✓ Image berhasil di-push ke ECR"

# ============================================================
# LANGKAH 10: Deploy ECS Cluster & Service
# ============================================================
echo ""
echo "=== LANGKAH 10: Deploy ECS ==="

# Buat cluster
aws ecs create-cluster \
  --cluster-name nusantaraai-cluster-${PROVINSI}-${NAMA} \
  --capacity-providers FARGATE \
  --region ${REGION}

# Buat task definition dari taskdef.json
aws ecs register-task-definition \
  --cli-input-json file://docker/taskdef.json \
  --region ${REGION}

echo "  ✓ ECS Cluster & Task Definition dibuat"
echo "  Buat ECS Service via Console (perlu ALB & Target Groups)"

# ============================================================
# LANGKAH 11: Test API
# ============================================================
echo ""
echo "=== LANGKAH 11: Test API ==="

# Ambil API Key
API_KEY=$(aws apigateway get-api-keys \
  --name-query "nusantaraai-api-key" \
  --include-values \
  --query "items[0].value" \
  --output text --region ${REGION})
echo "  API Key: $API_KEY"

echo ""
echo "  Test endpoint /recommendations:"
curl -s -X POST "${API_URL}/recommendations" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{"user_id":"USR00001","top_n":5}' | python3 -m json.tool

echo ""
echo "  Test endpoint /chat:"
curl -s -X POST "${API_URL}/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{"user_id":"USR00001","message":"Rekomendasikan pantai bagus di Indonesia"}' | python3 -m json.tool

# ============================================================
# LANGKAH 12: Trigger Step Functions Pipeline
# ============================================================
echo ""
echo "=== LANGKAH 12: Test Step Functions ==="
SFN_ARN=$(aws cloudformation describe-stacks \
  --stack-name nusantaraai-services \
  --query "Stacks[0].Outputs[?OutputKey=='StepFunctionArn'].OutputValue" \
  --output text --region ${REGION})

aws stepfunctions start-execution \
  --state-machine-arn ${SFN_ARN} \
  --input '{"trigger":"manual_test"}' \
  --region ${REGION}

echo "  ✓ Step Functions execution dimulai"

# ============================================================
# RINGKASAN
# ============================================================
echo ""
echo "======================================================"
echo " DEPLOY SELESAI!"
echo "======================================================"
echo " API URL      : ${API_URL}"
echo " API Key      : ${API_KEY}"
echo " Ollama IP    : ${OLLAMA_IP}"
echo " S3 Bucket    : ${BUCKET}"
echo ""
echo " Endpoints:"
echo "   POST ${API_URL}/recommendations"
echo "   POST ${API_URL}/chat"
echo "======================================================"
