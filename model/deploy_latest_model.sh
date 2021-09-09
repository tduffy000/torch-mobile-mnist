SRC_ROOT=./artifacts
MODEL_DIR=$(ls ${SRC_ROOT} | sort -r | head -n 1)
MODEL_ASSET=mobile_model.pt

MODEL_SRC=$SRC_ROOT/$MODEL_DIR/$MODEL_ASSET

echo "deploying $SRC_ROOT/$MODEL_DIR/$MODEL_ASSET ..."

TGT_DIR=../android/app/src/main/assets
rm -rf $TGT_DIR
mkdir -p $TGT_DIR

cp $MODEL_SRC $TGT_DIR/
echo "deployed model version: $MODEL_DIR"
