#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e

EXAMPLE=data
DATA=data/cifar-10-batches-bin
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

CAFFE_ROOT=/home/chris/work/caffe-BVLC
$CAFFE_ROOT/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$CAFFE_ROOT/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
