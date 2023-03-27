import sys, os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from imutils import face_utils


import YawnDetectModule as yd

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

# 获取最大的人脸
def largest_face(dets):
    largest_face = 0
    # print(dets)
    for det in dets:
        face_area = (det.rect.right()-det.rect.left())*(det.rect.bottom()-det.rect.top())
        if face_area > largest_face:
            largest_det = det
            largest_face = face_area

    return largest_det

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def upper_lip(landmarks):                               # To find out the mean cordinates of the upper lip position
    top_lip = []                                        # Creates a new list

    for i in range(50,53):                              # Creating a range from "50" to "53" because they represent the upper lip in dlib face predictor
        top_lip.append(landmarks[i])                    # Appending the same in the top lip list created above

    for j in range(61,64):                              # Creating a range from "61" to "64" because they represent the upper lip in dlib face predictor
        top_lip.append(landmarks[j])

    top_lip_point = (np.squeeze(np.asarray(top_lip)))   # This function removes any cordinate that is not in x,y format
    top_mean = np.mean(top_lip_point,axis=0)            # Finds the mean of the upper lip cordinates so that it can be subtracted from the mean cordinates of lower lip

    return int(top_mean[1])                             # Return int value of the mean of the cordinates of upper lip

def low_lip(landmarks):                                 # To find out the mean cordinates of the lower lip position
    lower_lip = []                                      # Creates a new list

    for i in range(65,68):                              # Creating a range from "65" to "68" because they represent the lower lip in dlib face predictor
        lower_lip.append(landmarks[i])

    for j in range(56,59):                              # Creating a range from "56" to "59" because they represent the lower lip in dlib face predictor
        lower_lip.append(landmarks[j])

    lower_lip_point = (np.squeeze(np.asarray(lower_lip)))   # This function removes any cordinate that is not in x,y format
    lower_mean = np.mean(lower_lip_point, axis=0)           # Finds the mean of the lower lip cordinates so that it can be subtracted from the mean cordinates of lower lip

    return int(lower_mean[1])                           # Return int value of the mean of the cordinates of lower lip

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_folder = args.video_path
    video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]

    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks_GTX.dat")
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    video_num = 0
    yawn_num = 0
    no_yawn_num = 0

    for video_path in video_paths:
        if not os.path.exists(args.video_path):
            sys.exit('Video does not exist')

        # Dlib face detection model
        cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)
        # cnn_face_detector = dlib.get_frontal_face_detector()

        print('Ready to test network.')

        # Test the Model
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

        # txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

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
            if len(dets) == 0:
                continue

            det = largest_face(dets)
            
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()

            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 1)

            face_pic = frame[(y_min-100):(y_max+100) , (x_min - 100):(x_max + 100)]

            # 提取嘴巴特征点
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, det.rect)
            shape = face_utils.shape_to_np(shape)
            
            if(shape.all() == [0]):                         # If no landmarks are created, return nothing
                continue 
            top_lip = upper_lip(shape)                      # Creates a variable that stores the mean of the cordinates returned by the function "upper_lip" created above
            lower_lip = low_lip(shape)                      # Creates a variable that stores the mean of the cordinates returned by the function "lower_lip" created above
            distance = abs(top_lip - lower_lip)                 # Subtracts the mean values of the two lips to find distance


            mouth = shape[mStart:mEnd]
            
            mar = mouth_aspect_ratio(mouth)

            mouth_x = []
            mouth_y = []
            for i in range(len(mouth)):
                mouth_x.append(mouth[i][0])
                mouth_y.append(mouth[i][1])
            
            x_min = np.min(mouth_x)
            x_max = np.max(mouth_x)
            y_min = np.min(mouth_y)
            y_max = np.min(mouth_y)
            mouth_pic = frame[(y_min-100):(y_max+100) , (x_min - 100):(x_max + 100)]

            if distance > 35:
                yawn_num += 1
                print(" yawn_num = " + f'{yawn_num}' + " \t mar = " + f'{mar}')
                cv2.imwrite("../pic/yawn_face/"+f'{yawn_num}'+".jpg",face_pic)
                cv2.imwrite("../pic/yawn_mouth/"+f'{yawn_num}'+".jpg",mouth_pic)
            else :
                no_yawn_num += 1
                print(" no_yawn_num = " + f'{no_yawn_num}' + " \t mar = " + f'{mar}')
                cv2.imwrite("../pic/no_yawn_face/"+f'{no_yawn_num}'+".jpg",face_pic)
                cv2.imwrite("../pic/no_yawn_mouth/"+f'{no_yawn_num}'+".jpg",mouth_pic)

            out.write(frame)

            frame_num += 1

        out.release()
        video.release()

