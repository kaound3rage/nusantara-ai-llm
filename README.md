🌏 NusantaraAI
AI-Powered Indonesian Tourism Recommendation System

Test Project 

NusantaraAI adalah sistem rekomendasi destinasi wisata Indonesia yang memanfaatkan Machine Learning dan Large Language Model (LLM) untuk memberikan rekomendasi wisata yang relevan berdasarkan preferensi pengguna dan riwayat interaksi.

Project ini dirancang menggunakan arsitektur cloud AWS end-to-end, mulai dari data lake, ETL pipeline, model training, hingga deployment service.

📑 Table of Contents

Project Objective

System Workflow

Cloud Architecture

Project Structure

Dataset Description

Machine Learning Model

Application Service

Infrastructure Deployment

Security Configuration

Example Recommendation

Future Improvements

Author

🎯 Project Objective

Tujuan dari project ini adalah:

Membangun sistem rekomendasi wisata berbasis AI

Mengimplementasikan Machine Learning pipeline di AWS

Menggunakan Cloud Infrastructure as Code (IaC) dengan CloudFormation

Mengintegrasikan LLM untuk natural language recommendation

Membangun sistem yang scalable dan production-ready

🧠 System Workflow

Alur kerja sistem NusantaraAI:

User Interaction
       │
       ▼
Application Service (Docker / API)
       │
       ▼
Recommendation Engine
       │
       ├── Machine Learning Model (SageMaker)
       │
       └── LLM Response Generation (Ollama)
       │
       ▼
Recommendation Result

Pipeline data:

Dataset
   │
   ▼
Amazon S3 (Data Lake)
   │
   ▼
AWS Glue ETL
   │
   ▼
Processed Dataset
   │
   ▼
Amazon SageMaker
(Model Training)
   │
   ▼
Model Deployment
   │
   ▼
API Service
☁️ Cloud Architecture (AWS)

Project ini menggunakan beberapa layanan AWS utama:

Service	Fungsi
Amazon S3	Penyimpanan dataset dan model
AWS Glue	ETL pipeline untuk preprocessing data
Amazon SageMaker	Training dan deployment ML model
AWS Lambda	Serverless automation
Docker	Containerized application
AWS CloudFormation	Infrastructure as Code
AWS CodeDeploy / ECS	Deployment service
📂 Project Structure

Struktur repository project:

NUSANTARAAI/
│
├── cloudformation/
│   ├── 01-networking.yaml
│   ├── 02-storage-compute.yaml
│   └── 03-services.yaml
│
├── dataset/
│   ├── destination_catalog.csv
│   ├── transaction_history.csv
│   ├── user_profiles.csv
│   ├── user_interactions.csv
│   └── generate_dataset.py
│
├── docker/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── appspec.yml
│   └── taskdef.json
│
├── glue/
│   └── etl_script.py
│
├── lambda/
│
├── sagemaker/
│   └── train_model.py
│
├── deploy.sh
│
└── README.md
📊 Dataset Description

Dataset digunakan untuk mensimulasikan platform wisata digital.

Dataset	Deskripsi
destination_catalog.csv	Informasi destinasi wisata
user_profiles.csv	Profil pengguna
transaction_history.csv	Riwayat pemesanan wisata
user_interactions.csv	Aktivitas pengguna dalam sistem

Dataset dapat dihasilkan menggunakan script berikut:

python dataset/generate_dataset.py
🤖 Machine Learning Model

Model Machine Learning digunakan untuk:

Menganalisis preferensi pengguna

Mengidentifikasi pola interaksi wisata

Menghasilkan rekomendasi destinasi

Training model dilakukan menggunakan Amazon SageMaker melalui script:

sagemaker/train_model.py
🐳 Application Service

Aplikasi utama dijalankan menggunakan Docker container.

Struktur utama service:

docker/
 ├── app.py
 ├── Dockerfile
 └── requirements.txt

Build container:

docker build -t nusantara-ai .

Menjalankan container:

docker run -p 5000:5000 nusantara-ai
⚙️ Infrastructure Deployment

Infrastructure dibuat menggunakan AWS CloudFormation.

Template yang tersedia:

cloudformation/
01-networking.yaml
02-storage-compute.yaml
03-services.yaml

Deployment dapat dijalankan dengan:

bash deploy.sh

Stack ini akan membuat resource berikut:

VPC

Subnet

Amazon S3 Data Lake

Compute resources

Machine Learning infrastructure

🔐 Security Configuration

Beberapa kebijakan keamanan diterapkan pada sistem.

Public Access

raw-data

models

Private Access

processed-data

logs

Akses private dibatasi menggunakan:

IAM Role

Bucket Policy

IP Whitelisting

💡 Example Recommendation

Input pengguna:

Location: Bali
Budget: 2 juta
Interest: Pantai dan wisata alam

Output sistem:

Recommended Destinations:

1. Nusa Penida
2. Pantai Melasti
3. Pantai Pandawa
4. Air Terjun Sekumpul
🚀 Future Improvements

Beberapa pengembangan yang dapat dilakukan di masa depan:

Integrasi Retrieval Augmented Generation (RAG)

Penggunaan vector database untuk semantic search

Real-time recommendation system

Dashboard monitoring untuk ML pipeline