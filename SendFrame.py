#!/usr/bin/python
# -*- coding: UTF-8 -*-
import imagezmq
import cv2
import numpy as np
import sys


sender = {}

def quit():
    print("[Py] quit.")
    sys.exit(0)

def register(device):
    print("[Py] register device {}".format(device))
    for i in range(10):
        sender[5555+device*10+i] = imagezmq.ImageSender(connect_to='tcp://localhost:{}'.format(5555+device*10+i))

def send(frame, frame_id, device, res_type):
    print("[Py] Group{} send {}th mat({}x{}) to port {}".format(device, frame_id, frame.shape[0], frame.shape[1], 5555+device*10+res_type))
    sender[5555+device*10+res_type].send_image(frame_id, frame)


if __name__ == '__main__':
    # test code
    while True:
        register(0)
        frame = cv2.imread('./test_img.jpeg')
        send(frame, 0, 0, 0)
        time.sleep(1)