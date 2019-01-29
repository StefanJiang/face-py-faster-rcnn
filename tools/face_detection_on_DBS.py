from __future__ import division
import _init_paths


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
# sys.path.insert(0,'/home/zjian30/test/face-py-faster-rcnn/caffe-fast-rcnn/python/caffe')
# import caffe_rcnn as caffe
import caffe
import argparse
import sys

NETS = {'vgg16': ('VGG16',
          '/home/zjian30/test/face-py-faster-rcnn/output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_80000.caffemodel')}


def vis_detections(im, class_name, dets, im_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    # save faces in the picture    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #for test
#         print('%f %f %f %f %f\n' % (bbox[0],bbox[1],bbox[2],bbox[3],score))
        im_face = im[int(bbox[1]):int(bbox[3]) ,int(bbox[0]): int(bbox[2]), :]
        fig_fc, ax_fc = plt.subplots(figsize=(12, 12))
        ax_fc.imshow(im_face, aspect='equal')
        plt.axis('off')
        plt.tight_layout()
        fig_fc.savefig(im_name)
    

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # cfg.TEST.BBOX_REG = False

    parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
            help='Use CPU mode (overrides --gpu)',
            action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            choices=NETS.keys(), default='vgg16')
    parser.add_argument('--name', type=str, default='DBS032', help='# of the patients')

    args = parser.parse_args()
    
    prototxt = '/home/zjian30/test/face-py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = NETS[args.demo_net][1]
    
    data_dir = '/home/zjian30/DBS_tmp/'
    p_name = args.name
    file_dir = data_dir + p_name + '/'
    imdir_list = os.listdir(file_dir)
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    CONF_THRESH = 0.65
    NMS_THRESH = 0.15

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    
    file_dir = data_dir + p_name + '/'
    imdir_list = os.listdir(file_dir)
    for i in range(len(imdir_list)):
        im_list = os.listdir(file_dir+imdir_list[i])
        for j in range(len(im_list)):
            im = cv2.imread(file_dir+imdir_list[i]+'/'+im_list[j])
            scores, boxes = im_detect(net, im)
            cls_ind = 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                    cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            keep = np.where(dets[:, 4] > CONF_THRESH)
            dets = dets[keep]
            if dets.size==0:
                print('no face found at img' + str(j) + 'in ' + file_dir+imdir_list[i])
            elif dets.size >= 10:
                print('more than one face found at img ' + str(im_list[j]) + ' in ' + file_dir+imdir_list[i])
            else:
                im_name = file_dir+imdir_list[i]+'/face_'+str(j)
                vis_detections(im, 'face', dets, im_name, thresh=CONF_THRESH)
    
    