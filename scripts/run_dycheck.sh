EXP_NAME=$1
GPU_ID=$2
SCENE=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

# using random port
PORT=$(( ( RANDOM % 64512 ) + 1024 ))

python train.py \
    -s /workspace/data/dycheck/$SCENE \
    --port $PORT \
    --expname $EXP_NAME/$SCENE \
    --configs arguments/dycheck/default.py

python render.py \
    --model_path output/$EXP_NAME/$SCENE  \
    --skip_train \
    --configs arguments/dycheck/default.py 

python metrics.py \
    --model_path output/$EXP_NAME/$SCENE