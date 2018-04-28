#!/usr/bin/env python2
# coding: utf-8

"""
inspired and copied from:
    - fcn.berkeleyvision.org
    - py-faster-rcnn
"""

from __future__ import print_function
import _init_paths
import caffe
import argparse
import os
import sys
from datetime import datetime
import cv2
import math

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
import numpy as np
import perfeval

from visualdl import LogWriter #for visualization during training

def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a classification network')
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str, required=True)

    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--log_dir', dest='log_dir',
                        help='log dir for VisualDL meta data',
                        default=None, type=str, required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class SolverWrapper:
    def __init__(self, solver_prototxt, log_dir, pretrained_model=None):
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print('Loading pretrained model weights from {:s}'.format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)
        
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        self.cur_epoch = 0
        self.test_interval = 500  #用来替代self.solver_param.test_interval
        #self.test_interval = 2000  #用来替代self.solver_param.test_interval
        self.logw = LogWriter(log_dir, sync_cycle=100)
        with self.logw.mode('train') as logger:
            self.sc_train_acc = logger.scalar("Accuracy")
            self.sc_train_lr = logger.scalar("learning_rate")
        with self.logw.mode('val') as logger:
            self.sc_val_acc = logger.scalar("Accuracy")
            self.sc_val_lr = logger.scalar("learning_rate")
        

    def train_model(self):
        """执行训练的整个流程，穿插了validation"""
        cur_iter = 0
        test_batch_size, num_classes = self.solver.test_nets[0].blobs['prob'].shape
        num_test_images_tot = test_batch_size * self.solver_param.test_iter[0]
        lr_policy = self.solver_param.lr_policy
        memo_t = 25   # 2 * 25(each epoch is 25)
        while cur_iter < self.solver_param.max_iter:
            #self.solver.step(self.test_interval)
            for i in range(self.test_interval):
                self.solver.step(1)
                cur_iter += 1

                #loss = self.solver.net.blobs['loss'].data
                if (cur_iter==1 or cur_iter % memo_t==0):
                    acc = float(self.solver.net.blobs['accuracy'].data)
                    step = cur_iter
                    lr = self.get_lr(lr_policy, cur_iter)
                    #self.sc_train_loss.add_record(step, loss)
                    self.sc_train_acc.add_record(step, acc)
                    self.sc_train_lr.add_record(step, lr)
                    self.eval_on_val(num_classes, num_test_images_tot, test_batch_size)
            #self.eval_on_val(num_classes, num_test_images_tot, test_batch_size)
        
    def eval_on_val(self, num_classes, num_test_images_tot, test_batch_size):
        """在整个验证集上执行inference和evaluation"""
        self.solver.test_nets[0].share_with(self.solver.net)
        self.cur_epoch += 1
        scores = np.zeros((num_classes, num_test_images_tot), dtype=np.float32)
        gt_labels = np.zeros((1, num_test_images_tot), dtype=np.float32).squeeze()
        for t in range(self.solver_param.test_iter[0]):
            output = self.solver.test_nets[0].forward()
            probs = output['prob']
            labels = self.solver.test_nets[0].blobs['label'].data

            gt_labels[t*test_batch_size:(t+1)*test_batch_size] = labels.T.astype(np.float32)
            scores[:,t*test_batch_size:(t+1)*test_batch_size] = probs.T
        # TODO: 处理最后一个batch样本少于num_test_images_per_batch的情况
        
        ap, acc = perfeval.cls_eval(scores, gt_labels)
        print('====================================================================\n')
        print('\tDo validation after the {:d}-th training epoch\n'.format(self.cur_epoch))
        print('>>>>', end='\t')  #设定标记，方便于解析日志获取出数据
        for i in range(num_classes):
            print('AP[{:d}]={:.4f}'.format(i, ap[i]), end=', ')
        mAP = np.average(ap)
        print('mAP={:.4f}, Accuracy={:.4f}'.format(mAP, acc))
        print('\n====================================================================\n')
        step = self.solver.iter
        lr_policy = self.solver_param.lr_policy
        lr = self.get_lr(lr_policy, step)
        self.sc_val_acc.add_record(step, acc)
        self.sc_val_lr.add_record(step, lr)


    def get_lr(self, lr_policy, cur_iter):
        if lr_policy=="fixed":
            rate = self.solver_param.base_lr
        elif lr_policy=="step":
            cur_step = cur_iter / self.solver_param.stepsize
            rate = self.solver_param.base_lr * math.pow(self.solver_param.gamma, cur_step)
        elif lr_policy=="exp":
            rate = self.solver_param.base_lr * math.pow(self.solver_param.gamma, cur_iter)
        elif lr_policy=="triangular":
            cycle = cur_iter / (2*self.solver_param.stepsize)
            x = float(cur_iter - (2*cycle+1)*self.solver_param.stepsize)
            x = x / self.solver_param.stepsize
            rate = self.solver_param.base_lr + (self.solver_param.max_lr  - self.solver_param.base_lr)*max(0, 1-abs(x))
        return rate


if __name__ == '__main__':
    args = parse_args()
    solver_prototxt = args.solver
    log_dir = args.log_dir
    pretrained_model = args.pretrained_model

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    sw = SolverWrapper(solver_prototxt, log_dir, pretrained_model)
    sw.train_model()