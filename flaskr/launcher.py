# import time
import cv2
import ctypes
# import numpy as np
from base_camera import BaseCamera

# ll = ctypes.cdll.LoadLibrary
# lib = ll("libMTMC.so")


class Launcher(BaseCamera):

    def __init__(self, device, res_type):
        super(Launcher, self).__init__(device, res_type)


    # def run(self):
    #     lib.Run()


    def frames(self, image_hub):
        """ 监听端口，阻塞接收图像 """
        while True:  # main loop
            frame_id, frame = image_hub.recv_image()
            print('recv->frame')
            image_hub.send_reply(b'OK')  # this is needed for the stream to work with REQ/REP pattern

            yield frame_id, frame