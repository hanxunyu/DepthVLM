#!/usr/bin/env bash
# video_to_gif.sh
#
# High-quality MP4 -> animated GIF converter using ffmpeg's two-pass
# palette method (palettegen + paletteuse). This is the standard
# "highest quality" technique for video-to-GIF conversion.
#
# Usage:
#   ./video_to_gif.sh <video1> [video2 ...]
#   ./video_to_gif.sh --fps 20 --width 1280 a.mp4 b.mp4
#   ./video_to_gif.sh --output-dir gifs/ assets/*.mp4
#
# Run `./video_to_gif.sh --help` for full options.

set -euo pipefail

FPS=""
WIDTH=""
OUTPUT_DIR=""
DITHER="sierra2_4a"
STATS_MODE="full"
LOOP=0
KEEP_ORIGINAL_QUALITY=0

usage() {
    cat <<'EOF'
Usage: video_to_gif.sh [options] <video1> [video2 ...]

Convert MP4 (or any ffmpeg-supported video) to high-quality animated GIF
using a two-pass palette workflow.

Options:
  --fps <N>            Output frame rate. Default: original video fps.
  --width <N>          Output width in pixels (height auto-scaled, even).
                       Default: original width.
  --output-dir <DIR>   Output directory. Default: same dir as input file.
  --dither <NAME>      Dither algorithm: sierra2_4a (default, best quality),
                       floyd_steinberg, bayer, none.
  --stats-mode <NAME>  Palette stats mode: full (default, best for varied
                       scenes) or diff (better for static scenes).
  --no-loop            Play once. Default: infinite loop.
  --max-quality        Force NO downscaling and NO fps change.
                       (Equivalent to omitting --fps and --width.)
  -h, --help           Show this help.

Presets (recommended):
  # GitHub README (autoplay-able, small file)
  ./video_to_gif.sh --width 960 --fps 15 assets/clip.mp4

  # Highest quality (very large files, ~10-20x source size)
  ./video_to_gif.sh --max-quality assets/clip.mp4

  # Balanced HD
  ./video_to_gif.sh --width 1280 --fps 20 assets/clip.mp4

EOF
}

INPUTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)         FPS="$2"; shift 2 ;;
        --width)       WIDTH="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --dither)      DITHER="$2"; shift 2 ;;
        --stats-mode)  STATS_MODE="$2"; shift 2 ;;
        --no-loop)     LOOP=-1; shift ;;
        --max-quality) KEEP_ORIGINAL_QUALITY=1; FPS=""; WIDTH=""; shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; INPUTS+=("$@"); break ;;
        -*)            echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)             INPUTS+=("$1"); shift ;;
    esac
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    usage
    exit 1
fi

command -v ffmpeg >/dev/null 2>&1 || {
    echo "Error: ffmpeg is not installed. Install it with: brew install ffmpeg" >&2
    exit 1
}

# Build the shared video filter chain (everything before palettegen / paletteuse).
build_filters() {
    local filters=()
    [[ -n "$FPS" ]]   && filters+=("fps=${FPS}")
    [[ -n "$WIDTH" ]] && filters+=("scale=${WIDTH}:-2:flags=lanczos")
    if [[ ${#filters[@]} -eq 0 ]]; then
        # No-op filter so the filtergraph syntax stays valid.
        echo "null"
    else
        local IFS=,
        echo "${filters[*]}"
    fi
}

VIDEO_FILTERS=$(build_filters)

human_size() {
    # Cross-platform (macOS/Linux) human-readable file size.
    local f="$1"
    if [[ "$(uname)" == "Darwin" ]]; then
        local b
        b=$(stat -f%z "$f")
        awk -v b="$b" 'BEGIN {
            if (b >= 1048576) printf "%.2f MB", b/1048576;
            else if (b >= 1024) printf "%.2f KB", b/1024;
            else printf "%d B", b;
        }'
    else
        du -h "$f" | cut -f1
    fi
}

for INPUT in "${INPUTS[@]}"; do
    if [[ ! -f "$INPUT" ]]; then
        echo "Skipping (not found): $INPUT" >&2
        continue
    fi

    base=$(basename "$INPUT")
    name="${base%.*}"

    if [[ -n "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
        OUT="$OUTPUT_DIR/${name}.gif"
    else
        OUT="$(dirname "$INPUT")/${name}.gif"
    fi

    PALETTE=$(mktemp -t "vid2gif.XXXXXX")
    PALETTE="${PALETTE}.png"

    echo ""
    echo "=== ${INPUT} -> ${OUT} ==="
    echo "    input size : $(human_size "$INPUT")"
    echo "    filters    : ${VIDEO_FILTERS}"
    echo "    dither     : ${DITHER}    stats_mode: ${STATS_MODE}    loop: ${LOOP}"

    echo "[1/2] Generating optimal 256-color palette..."
    ffmpeg -hide_banner -v error -stats -y -i "$INPUT" \
        -vf "${VIDEO_FILTERS},palettegen=stats_mode=${STATS_MODE}" \
        "$PALETTE"

    echo "[2/2] Encoding GIF (this can take a while at full resolution)..."
    ffmpeg -hide_banner -v error -stats -y -i "$INPUT" -i "$PALETTE" \
        -lavfi "${VIDEO_FILTERS} [x]; [x][1:v] paletteuse=dither=${DITHER}" \
        -loop "$LOOP" \
        "$OUT"

    rm -f "$PALETTE"

    echo "    output size: $(human_size "$OUT")"
done

echo ""
echo "All conversions finished."
