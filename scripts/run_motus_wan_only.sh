#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_CONFIG="$ROOT_DIR/inference/real_world/Motus/utils/ac_one.yaml"
CKPT_DIR="$ROOT_DIR/pretrained_models/Motus"
WAN_ROOT="/home/zqy/ws/Wan2.2"
DEFAULT_IMAGE_REL="Wan2.2-TI2V-5B/examples/i2v_input.JPG"
IMAGE="$WAN_ROOT/$DEFAULT_IMAGE_REL"
INSTRUCTION="Pour water from kettle to flowers"
OUTPUT="$ROOT_DIR/outputs/motus_wan_only.png"
T5_CACHE_DIR="$ROOT_DIR/.cache/t5_embeds"
INFER_GPU="1"
T5_GPU=""
T5_DEVICE=""
T5_GPU_SET_BY_USER="0"
T5_DEVICE_SET_BY_USER="0"
FORCE_REENCODE="0"
NUM_INFERENCE_STEPS=""
OFFLOAD_VAE_TO_CPU="0"
LOW_VRAM="0"
T5_EMBED_OVERRIDE=""

usage() {
    cat <<EOF
Usage:
  bash scripts/run_motus_wan_only.sh
  bash scripts/run_motus_wan_only.sh --image /path/to/image.png --instruction "pick up the cube"

Options:
  --image PATH           Input image path (optional; default: $WAN_ROOT/$DEFAULT_IMAGE_REL)
  --instruction TEXT     Instruction text (optional; default: "$INSTRUCTION")
  --output PATH          Output image path (default: $OUTPUT)
  --config PATH          Motus yaml config (default: $MODEL_CONFIG)
  --ckpt-dir PATH        Motus checkpoint dir (default: $CKPT_DIR)
  --wan-root PATH        WAN root dir that contains Wan2.2-TI2V-5B/ (default: $WAN_ROOT)
  --infer-gpu ID         Physical GPU id for Motus inference (default: $INFER_GPU)
  --t5-gpu ID            Physical GPU id for T5 encoding (default: same as --infer-gpu)
  --t5-device DEVICE     Device for T5 encoding (e.g. cuda:1 or cpu; default: cuda:<t5-gpu>)
  --t5-embed PATH        Use an existing T5 embedding file directly (skip encoding step)
  --force-reencode       Ignore cached T5 embedding and regenerate it
  --num-steps N          Override denoising steps in config (lower is less VRAM, e.g. 4)
  --offload-vae-to-cpu   Move VAE to CPU during denoising (lower VRAM, slower)
  --low-vram             Shortcut: equivalent to --num-steps 4 --offload-vae-to-cpu
  --help                 Show this message

Notes:
  - Current Motus inference is single-GPU. This script uses the second GPU only for T5 encoding/caching.
  - ac_one.yaml should keep model.vlm.enabled: false for WAN-only inference.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --ckpt-dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --wan-root)
            WAN_ROOT="$2"
            shift 2
            ;;
        --infer-gpu)
            INFER_GPU="$2"
            shift 2
            ;;
        --t5-gpu)
            T5_GPU="$2"
            T5_GPU_SET_BY_USER="1"
            shift 2
            ;;
        --t5-device)
            T5_DEVICE="$2"
            T5_DEVICE_SET_BY_USER="1"
            shift 2
            ;;
        --t5-embed)
            T5_EMBED_OVERRIDE="$2"
            shift 2
            ;;
        --force-reencode)
            FORCE_REENCODE="1"
            shift 1
            ;;
        --num-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --offload-vae-to-cpu)
            OFFLOAD_VAE_TO_CPU="1"
            shift 1
            ;;
        --low-vram)
            LOW_VRAM="1"
            shift 1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$T5_GPU" ]]; then
    T5_GPU="$INFER_GPU"
fi

if [[ -z "$T5_DEVICE" ]]; then
    T5_DEVICE="cuda:$T5_GPU"
fi

if [[ "$LOW_VRAM" == "1" ]]; then
    if [[ -z "$NUM_INFERENCE_STEPS" ]]; then
        NUM_INFERENCE_STEPS="4"
    fi
    OFFLOAD_VAE_TO_CPU="1"
    if [[ "$T5_DEVICE_SET_BY_USER" == "0" ]]; then
        T5_DEVICE="cpu"
    fi
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Config not found: $MODEL_CONFIG" >&2
    exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint dir not found: $CKPT_DIR" >&2
    exit 1
fi

if [[ ! -f "$CKPT_DIR/mp_rank_00_model_states.pt" ]]; then
    echo "Checkpoint file missing: $CKPT_DIR/mp_rank_00_model_states.pt" >&2
    exit 1
fi

if [[ ! -d "$WAN_ROOT/Wan2.2-TI2V-5B" ]]; then
    echo "WAN model dir not found: $WAN_ROOT/Wan2.2-TI2V-5B" >&2
    exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "Image not found: $IMAGE" >&2
    echo "No bundled Motus sample image exists in this repo. Provide --image explicitly." >&2
    exit 1
fi

mkdir -p "$T5_CACHE_DIR"
mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$T5_EMBED_OVERRIDE" ]]; then
    T5_EMBED="$T5_EMBED_OVERRIDE"
else
    INSTRUCTION_HASH="$(printf '%s' "$INSTRUCTION" | sha1sum | awk '{print $1}')"
    T5_EMBED="$T5_CACHE_DIR/$INSTRUCTION_HASH.pt"
fi

echo "Root dir: $ROOT_DIR"
echo "Config: $MODEL_CONFIG"
echo "Checkpoint: $CKPT_DIR"
echo "WAN root: $WAN_ROOT"
echo "Inference GPU: $INFER_GPU"
if [[ "$T5_DEVICE" == cpu* ]]; then
    echo "T5 GPU: N/A (using $T5_DEVICE)"
else
    echo "T5 GPU: $T5_GPU"
fi
echo "T5 device: $T5_DEVICE"
echo "T5 cache: $T5_EMBED"
if [[ -n "$NUM_INFERENCE_STEPS" ]]; then
    echo "Inference steps override: $NUM_INFERENCE_STEPS"
fi
echo "Offload VAE to CPU: $OFFLOAD_VAE_TO_CPU"
echo "Output: $OUTPUT"

if [[ ! -f "$T5_EMBED" ]]; then
    if [[ -n "$T5_EMBED_OVERRIDE" ]]; then
        echo "Provided --t5-embed does not exist: $T5_EMBED" >&2
        exit 1
    fi
fi

if [[ -n "$T5_EMBED_OVERRIDE" ]]; then
    echo "Using provided T5 embedding: $T5_EMBED"
elif [[ "$FORCE_REENCODE" == "1" || ! -f "$T5_EMBED" ]]; then
    echo "Encoding T5 embedding on $T5_DEVICE"
    python "$ROOT_DIR/inference/real_world/Motus/encode_t5_instruction.py" \
        --instruction "$INSTRUCTION" \
        --output "$T5_EMBED" \
        --wan_path "$WAN_ROOT" \
        --device "$T5_DEVICE"
else
    echo "Using cached T5 embedding: $T5_EMBED"
fi

echo "Running Motus WAN-only inference on GPU $INFER_GPU"
EXTRA_INFER_ARGS=()
if [[ -n "$NUM_INFERENCE_STEPS" ]]; then
    EXTRA_INFER_ARGS+=(--num_inference_steps "$NUM_INFERENCE_STEPS")
fi
if [[ "$OFFLOAD_VAE_TO_CPU" == "1" ]]; then
    EXTRA_INFER_ARGS+=(--offload_vae_to_cpu)
fi

CUDA_VISIBLE_DEVICES="$INFER_GPU" \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python "$ROOT_DIR/inference/real_world/Motus/inference_example.py" \
    --model_config "$MODEL_CONFIG" \
    --ckpt_dir "$CKPT_DIR" \
    --wan_path "$WAN_ROOT" \
    --image "$IMAGE" \
    --instruction "$INSTRUCTION" \
    --t5_embeds "$T5_EMBED" \
    --output "$OUTPUT" \
    "${EXTRA_INFER_ARGS[@]}"
