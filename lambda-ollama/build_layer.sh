#!/bin/bash
# ====================================================
# NusantaraAI - Build Lambda Layer (fixed)
# Jalankan di SageMaker terminal
# ====================================================

set -e

TARGET="python"
ZIP_NAME="nusantaraai-layer.zip"

echo "=== Bersihkan folder lama ==="
rm -rf "$TARGET" "$ZIP_NAME"
mkdir -p "$TARGET"

echo "=== Install library utama (binary wheels) ==="
pip install \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    pyarrow==16.1.0 \
    --target "$TARGET" \
    --only-binary=:all: \
    --quiet

echo "=== Install scikit-surprise (perlu build dari source) ==="
pip install \
    scikit-surprise==1.1.3 \
    --target "$TARGET" \
    --use-pep517 \
    --quiet

echo "=== Hapus file tidak perlu (kurangi ukuran zip) ==="
find "$TARGET" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name "tests"       -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name "*.pyc"               -delete 2>/dev/null || true
find "$TARGET" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Zip layer ==="
zip -r "$ZIP_NAME" "$TARGET"/ -q

SIZE=$(du -sh "$ZIP_NAME" | cut -f1)
echo ""
echo "✓ Layer selesai: $ZIP_NAME ($SIZE)"
echo ""
echo "Langkah selanjutnya — upload ke AWS:"
echo ""
echo "  aws lambda publish-layer-version \\"
echo "      --layer-name nusantaraai-ml-deps \\"
echo "      --zip-file fileb://$ZIP_NAME \\"
echo "      --compatible-runtimes python3.10 python3.11 \\"
echo "      --region us-east-1"
