#!/bin/bash
# Merge packed parquet chunks into a single Hive-partitioned directory.
#
# Supports two modes:
#   1. CHUNK MERGE (default): each chunk_N/ is a packed Hive directory with split=train/, split=validation/
#   2. PREPROCESS MERGE: each chunk_N/data.parquet (flat preprocessed output)
#
# Usage:
#   # Merge packed chunks (chunk_0/, chunk_1/, ...) into one directory:
#   ./scripts/merge_packed_chunks.sh \
#       --source /work/hdd/bfaq/jlyu3/lcm/preprocessed_data_packed \
#       --output /work/hdd/bfaq/jlyu3/lcm/preprocessed_data_packed_merged
#
#   # Merge preprocessed chunks (flat data.parquet per chunk):
#   ./scripts/merge_packed_chunks.sh --mode preprocess \
#       --source /work/hdd/bfaq/jlyu3/lcm/preprocessed_data \
#       --output /work/hdd/bfaq/jlyu3/lcm/preprocessed_data_merged

set -e

MODE="chunk"
SOURCE=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)   MODE="$2";   shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$SOURCE" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --source <dir> --output <dir> [--mode chunk|preprocess]"
    exit 1
fi

echo "=============================="
echo "Merge Parquet Data"
echo "=============================="
echo "  Mode:   $MODE"
echo "  Source: $SOURCE"
echo "  Output: $OUTPUT"
echo ""

if [ "$MODE" = "chunk" ]; then
    # Each chunk_N/ has split=train/data.parquet and split=validation/data.parquet
    mkdir -p "$OUTPUT/split=train" "$OUTPUT/split=validation"

    count=0
    for chunk_dir in "$SOURCE"/chunk_*; do
        [ -d "$chunk_dir" ] || continue
        chunk_name=$(basename "$chunk_dir")
        idx=${chunk_name#chunk_}

        for split in train validation; do
            src="$chunk_dir/split=$split/data.parquet"
            if [ -f "$src" ]; then
                dst="$OUTPUT/split=$split/data_chunk_${idx}.parquet"
                cp "$src" "$dst"
                echo "  $src -> $dst"
                count=$((count + 1))
            fi
        done
    done

    echo ""
    echo "Copied $count parquet files."

elif [ "$MODE" = "preprocess" ]; then
    # Flat merge: each chunk_N/data.parquet -> output/data_chunk_N.parquet
    mkdir -p "$OUTPUT"
    count=0

    for chunk_dir in "$SOURCE"/chunk_*; do
        [ -d "$chunk_dir" ] || continue
        chunk_name=$(basename "$chunk_dir")
        idx=${chunk_name#chunk_}

        src="$chunk_dir/data.parquet"
        if [ -f "$src" ]; then
            dst="$OUTPUT/data_chunk_${idx}.parquet"
            cp "$src" "$dst"
            echo "  $src -> $dst"
            count=$((count + 1))
        fi
    done

    echo ""
    echo "Copied $count parquet files."

else
    echo "Unknown mode: $MODE (expected chunk or preprocess)"
    exit 1
fi

# Print row counts if python/pyarrow is available
echo ""
echo "--- Statistics ---"
if command -v python3 &>/dev/null; then
    python3 -c "
import sys, os
try:
    import pyarrow.parquet as pq
except ImportError:
    print('pyarrow not available, skipping row counts')
    sys.exit(0)

out = '$OUTPUT'
total = 0
for root, dirs, files in os.walk(out):
    for f in sorted(files):
        if f.endswith('.parquet'):
            path = os.path.join(root, f)
            rows = pq.ParquetFile(path).metadata.num_rows
            total += rows
            rel = os.path.relpath(path, out)
            print(f'  {rel}: {rows:,} rows')
print(f'  Total: {total:,} rows')
"
else
    echo "  (python3 not found, skipping row counts)"
fi

echo ""
echo "Done. Output: $OUTPUT"
