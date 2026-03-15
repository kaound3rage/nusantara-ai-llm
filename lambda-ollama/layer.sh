#!/bin/bash
# ====================================================
# NusantaraAI - Build Lambda Layer (final)
# ====================================================

set -e

PYTHON_VERSION="python3.10"
TARGET="python/lib/${PYTHON_VERSION}/site-packages"
ZIP_NAME="nusantaraai-layer.zip"

echo "=== Bersihkan folder lama ==="
rm -rf python/ "$ZIP_NAME"
mkdir -p "$TARGET"

echo "=== Install dependencies ==="
pip install \
    "scikit-learn==1.4.2" \
    "numpy==1.26.4" \
    "pandas>=2.2.2" \
    --target "$TARGET" \
    --only-binary=:all: \
    --ignore-installed \
    --quiet

echo "=== Trim file tidak perlu ==="
find "$TARGET" -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name "tests"        -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -type d -name "test"         -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name "*.pyc"                -delete       2>/dev/null || true
find "$TARGET" -name "*.dist-info" -type d  -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name "*.egg-info"  -type d  -exec rm -rf {} + 2>/dev/null || true
rm -rf "$TARGET/sklearn/datasets"           2>/dev/null || true
rm -rf "$TARGET/numpy/f2py"                 2>/dev/null || true
rm -rf "$TARGET/numpy/testing"              2>/dev/null || true
rm -rf "$TARGET/pandas/tests"               2>/dev/null || true

echo "=== Ukuran setelah trim ==="
du -sh "$TARGET"

echo "=== Zip ==="
zip -r "$ZIP_NAME" python/ -q

SIZE=$(du -sh "$ZIP_NAME" | cut -f1)
echo ""
echo "✓ $ZIP_NAME ($SIZE)"
echo ""
echo "Upload:"
echo "  aws lambda publish-layer-version \\"
echo "      --layer-name nusantaraai-ml-deps \\"
echo "      --zip-file fileb://$ZIP_NAME \\"
echo "      --compatible-runtimes python3.10 \\"
echo "      --region us-east-1"
