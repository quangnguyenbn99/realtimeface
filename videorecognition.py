from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import facenet
import detect_face
import os
from os.path import join as pjoin
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
def main():
    listpath='/home/ubuntu/Desktop/PRfacenet/test/list_candidate'

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, '/home/ubuntu/Desktop/PRfacenet/src/align')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 32
            frame_interval = 3
            batch_size = 1000
            image_size = 160
            input_image_size = 160

            print('Loading feature extraction model')
            modeldir = '/home/ubuntu/Desktop/PRfacenet/casia_pre_trained_model/20170511-185253.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename = '/home/ubuntu/Desktop/PRfacenet/PR-classifier/PRcasia_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)


            #capture video frame
            video_capture = cv2.VideoCapture('/media/ubuntu/MULTIBOOT/q.mkv')
            c = 0
            # frame_width = 640
            # frame_height = 480
            # fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
            # out = cv2.VideoWriter('/home/ubuntu/Desktop/output.avi',fourcc, 20.0, (frame_width,frame_height),True)
            

            print('Start Recognition!')
            prevTime = 0
            while True:
                ret, frame = video_capture.read()

                if frame.shape[0] == 0:
                    break
                else:
                
                    num_rows, num_cols = frame.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 0, 1)
                    frame = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))        

                    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                    print(' %d      %d      '%(frame.shape[0],frame.shape[1]))
                    curTime = time.time()+1    # calc fps
                    timeF = frame_interval

                    if (c % timeF == 0):
                        find_results = []

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        print ('',len(bounding_boxes))
                        print (frame.shape)
                        if len(bounding_boxes)>0:
    	                    
    		                    	# if bounding_boxes
    	                    nrof_faces = bounding_boxes.shape[0]
    	                    print('Detected_Face: %d' % nrof_faces)

    	                    if nrof_faces > 0:
    	                        det = bounding_boxes[:, 0:4]
    	                        print (det)
    	                        img_size = np.asarray(frame.shape)[0:2]

    	                        cropped = []
    	                        scaled = []
    	                        scaled_reshape = []
    	                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

    	                        for i in range(0,nrof_faces):
    	                            emb_array = np.zeros((1, embedding_size))

    	                            bb[i][0] = det[i][0]#x topleft
    	                            bb[i][1] = det[i][1]#y topleft
    	                            bb[i][2] = det[i][2]#x bottom right
    	                            bb[i][3] = det[i][3]#y bottom right

    	                            print ("bb[i][0]:%d,bb[i][1]:%d,bb[i][2]:%d,bb[i][3]:%d" %(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))


    	                            # inner exception
    	                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
    	                            	break

    	                            if det[i][0]>=0 and det[i][1]>=0 and det[i][2]<=frame.shape[1]and bounding_boxes[i][3]<=frame.shape[0]:
    	                                print('face is inner of range!')

    	                                # continue

    	                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
    	                                cropped[i] = facenet.flip(cropped[i], False)

    	                                # print (cropped[i].shape)
    	                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
    	                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
    	                                                       interpolation=cv2.INTER_CUBIC)
    	                                scaled[i] = facenet.prewhiten(scaled[i])
    	                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
    	                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}

    	                               

    	                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
    	                                predictions = model.predict_proba(emb_array)
    	                                best_class_indices = np.argmax(predictions, axis=1)
    	                                print (len)
    	                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    	                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 1)    #boxing face #top left ++++ bottom right

    	                                #plot result idx under box
    	                                text_x = bb[i][0]
    	                                text_y = bb[i][1]

    	                                print('result: %s '%class_names[best_class_indices[0]])
    	                                cadiname=class_names[best_class_indices[0]]
    	                                print('accuracy:%.4f '%best_class_probabilities[0])
    	                                accuracy=best_class_probabilities[0]
    	                                
    	                                
    	                                if best_class_probabilities[0]>=0.6:
    	                                    
    	                                    f_text=open(os.path.join('%s/name_of_candidate.txt'%listpath),'w')
    	                                    
    	                                    f_text.write(class_names[best_class_indices[0]]+'\n')
    	                                    cv2.putText(frame,class_names[best_class_indices[0]],(text_x, text_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7, (0, 255, 255), thickness=1, lineType=1)
    	                                    cv2.putText(frame,'%3s'%accuracy,(text_x, text_y-30),cv2.FONT_HERSHEY_DUPLEX,0.5, (0, 255, 255), thickness=1, lineType=1)
    	                                if best_class_probabilities[0]<=0.6:
    	                                    cv2.putText(frame,'Unknown',(text_x, text_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7, (0, 255, 255), thickness=1, lineType=1)
    	                                
    	                            else:
    	                                print('Nobody is in the camera zone!!!!')
                            
                    sec = curTime - prevTime
                    prevTime = curTime
                    fps = 1 / (sec)

                    str = 'FPS: %2.3f' % fps
                    text_fps_x = len(frame[0]) - 320
                    text_fps_y = 20
                    cv2.putText(frame, str, (text_fps_x, text_fps_y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                    cv2.imshow('Video', frame)

                if cv2.waitKey(33) == 27:
                    break

            video_capture.release()
            cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
