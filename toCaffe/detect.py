from __future__ import print_function
import sys
sys.path.insert(0, "/mnt/e/code/DeepLearning/caffe/python")
sys.path.insert(1, "../")
import caffe
import argparse
import torch
import numpy as np
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time


class CaffeInference(caffe.Net):
    """docstring for ClassName"""

    def __init__(self, model_file, pretrained_file, mean=None, use_gpu=False, device_id=0):
        self.__mean = mean
        if use_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        else:
            caffe.set_mode_cpu()

        self.__net = caffe.Net(model_file, pretrained_file, caffe.TEST)

    def predict(self, img, input_name="data", output_name=["BboxHead_Concat", "ClassHead_Softmax", "LandmarkHead_Concat"]):
        img -= (self.__mean)
        if 3 == len(self.__mean):
            img = img.transpose(2, 0, 1)
            new_shape = [1, img.shape[0], img.shape[1], img.shape[2]]
        else:
            new_shape = [1, 1, img.shape[0], img.shape[1]]

        img = img.reshape(new_shape)
        self.__net.blobs[input_name].reshape(*new_shape)
        self.__net.blobs[input_name].data[...] = img

        self.__net.forward()

        res = []
        for item in output_name:
            res.append(self.__net.blobs[item].data)

        return (*res, img)


def demo(args, cfg):

    net = CaffeInference(args.deploy, args.trained_model, mean=(104, 117, 123), use_gpu=not args.cpu, device_id=0)
    print('Finished loading model!')

    device = torch.device("cpu" if args.cpu else "cuda")

    resize = 1

    # testing begin
    img_raw = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms, img = net.predict(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    loc = torch.tensor(loc)
    conf = torch.tensor(conf)
    landms = torch.tensor(landms)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            print(b)
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "test.jpg"
        cv2.imwrite(name, img_raw)
        if args.show_image:
            cv2.imshow("Demo", img_raw)
            cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='models/mobile0.25.caffemodel',
                        type=str, help='Trained caffemodel path')
    parser.add_argument('--deploy', default='models/mobile0.25.prototxt', help='Path of deploy file')
    parser.add_argument('--img_path', default='../curve/test.jpg', help='Path of test image')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='save detection results')
    parser.add_argument('--show_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    cfg = cfg_mnet
    demo(args, cfg)
