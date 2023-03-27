import sys, os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=3, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../hopenet_alpha2.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='../mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video',default='../')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',default="test")
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', default=1000,type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=60.)
    args = parser.parse_args()
    return args

# 获取最右的人脸
def rightmost_face(dets):
    largest_face = 0
    # print(dets)
    for det in dets:
        face_area = (det.rect.right()-det.rect.left())*(det.rect.bottom()-det.rect.top())
        if face_area > largest_face:
            largest_det = det
            largest_face = face_area

    return largest_det

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_folder = args.video_path
    video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    video_num = 0
    for video_path in video_paths:
        if not os.path.exists(args.video_path):
            sys.exit('Video does not exist')

        # ResNet50 structure
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Dlib face detection model
        cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)
        # cnn_face_detector = dlib.get_frontal_face_detector()

        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)

        print('Loading data.')

        transformations = transforms.Compose([transforms.Resize(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        model.cuda(gpu)

        print('Ready to test network.')

        # Test the Model
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        total = 0

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

        video = cv2.VideoCapture(video_path)

        # New cv2
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/video/output-%s.mp4' % args.output_string, fourcc, args.fps, (width, height))

        # # Old cv2
        # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
        # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
        #
        # # Define the codec and create VideoWriter object
        # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
        # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

        txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

        video_num += 1
        frame_num = 1
        # print(video_frames)

        while frame_num < video_frames:
            print("video {0} : frame {1}".format(video_num,frame_num))

            ret,frame = video.read()
            if ret == False:
                break
            
            # cv2.imwrite("../pic/"+f'{frame_num}'+".jpg",frame)
            # out.write(frame)
            cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # Dlib detect
            dets = cnn_face_detector(cv2_frame, 1)
            det = rightmost_face(dets)
            
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
            # Crop image
            img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw,dim = 1)
            pitch_predicted = F.softmax(pitch,dim = 1)
            roll_predicted = F.softmax(roll,dim = 1)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            # Print new frame with cube and axis
            # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
            # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            cv2.imwrite("../pic/"+f'{frame_num}'+".jpg",frame)
            out.write(frame)
            # Plot expanded bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)


            frame_num += 1

        out.release()
        video.release()

        # for idx, det in enumerate(dets):
        #     # Get x_min, y_min, x_max, y_max, conf
        #     x_min = det.rect.left()
        #     y_min = det.rect.top()
        #     x_max = det.rect.right()
        #     y_max = det.rect.bottom()
        #     conf = det.confidence

        #     if conf > 1.0:
        #         bbox_width = abs(x_max - x_min)
        #         bbox_height = abs(y_max - y_min)
        #         x_min -= 2 * bbox_width / 4
        #         x_max += 2 * bbox_width / 4
        #         y_min -= 3 * bbox_height / 4
        #         y_max += bbox_height / 4
        #         x_min = max(x_min, 0); y_min = max(y_min, 0)
        #         x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
        #         # Crop image
        #         img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
        #         img = Image.fromarray(img)

        #         # Transform
        #         img = transformations(img)
        #         img_shape = img.size()
        #         img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        #         img = Variable(img).cuda(gpu)

        #         yaw, pitch, roll = model(img)

        #         yaw_predicted = F.softmax(yaw,dim = 1)
        #         pitch_predicted = F.softmax(pitch,dim = 1)
        #         roll_predicted = F.softmax(roll,dim = 1)
        #         # Get continuous predictions in degrees.
        #         yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        #         pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        #         roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        #         # Print new frame with cube and axis
        #         # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
        #         # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
        #         # utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
        #         # cv2.imwrite("../pic/"+f'{frame_num}'+".jpg",frame)
        #         # out.write(frame)
        #         # Plot expanded bounding box
        #         # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        # # out.write(frame)
        # frame_num += 1

    # out.release()
    # video.release()
