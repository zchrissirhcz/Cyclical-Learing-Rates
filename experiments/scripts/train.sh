#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"


LOG_NAME="log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/${LOG_NAME}"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x

#--weights data/imagenet_models/${NET}.caffemodel \
time ./tools/solve.py \
  --solver models/cifar10_full_solver.prototxt \
  --log_dir experiments/vdl_logs
