#!/bin/bash
# ====================================================
# NusantaraAI - Build Lambda Layer (no scikit-surprise)
# Jalankan di SageMaker terminal
#
# scikit-surprise TIDAK dimasukkan ke layer karena:
#   1. Tidak ada pre-built wheel untuk Python 3.10/3.11
#   2. Model sudah menggunakan cf_algorithm: "UserAverage"
#      (bukan SVD), jadi surprise tidak dibutuhkan saat inference
#
# surprise hanya dibutuhkan saat TRAINING, bukan inference.
# ====================================================

set -e

TARGET="python"
ZIP_NAME="nusantaraai-layer.zip"

echo "=== Bersihkan folder lama ==="
rm -rf "$TARGET" "$ZIP_NAME"
mkdir -p "$TARGET"

echo "=== Install library (binary wheels only) ==="
pip install \
    "scikit-learn==1.4.2" \
    "pandas>=2.2.2" \
    "numpy==1.26.4" \
    "pyarrow==16.1.0" \
    --target "$TARGET" \
    --only-binary=:all: \
    --ignore-installed \
    --quiet

echo "=== Verifikasi isi folder ==="
ls "$TARGET" | grep -E "sklearn|pandas|numpy|pyarrow" || true

echo "=== Hapus file tidak perlu ==="
find "$TARGET" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name "tests"       -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name "*.pyc"               -delete       2>/dev/null || true
find "$TARGET" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Zip layer ==="
zip -r "$ZIP_NAME" "$TARGET"/ -q

SIZE=$(du -sh "$ZIP_NAME" | cut -f1)
echo ""
echo "✓ Layer selesai: $ZIP_NAME ($SIZE)"
echo ""
echo "Upload ke AWS:"
echo "  aws lambda publish-layer-version \\"
echo "      --layer-name nusantaraai-ml-deps \\"
echo "      --zip-file fileb://$ZIP_NAME \\"
echo "      --compatible-runtimes python3.10 python3.11 \\"
echo "      --region us-east-1"