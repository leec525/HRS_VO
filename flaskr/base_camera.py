""" 由于网页同时只能加载6张图像,若使用阻塞策略，
    当图像过多时将无法一次性初始化所有BaseCamera,
    因此停用CameraEvent,令网页无阻塞请求。
"""
import time
import threading
import imagezmq
import numpy as np


from greenlet import getcurrent as get_ident


class CameraEvent:
    """ An Event-like class that signals all active clients when a new frame is
        available. 
        用于在所属相机的新一帧到来后, 通知所有正在查看相机输出的客户端, 也就是BaseCamera的get_frame();
        需要为每个相机都attach一个CameraEvent
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available.
           提醒有新的帧可用了
        """
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    """ 
    约定: port = 5555+deviceId*10+res_type, 为每个相机预留10个res_type
    """
    threads = {}  # background thread that reads frames from camera
    frame = {}  # current frame is stored here by background thread
    frame_id = {}
    last_access = {}  # time of last client access to the camera
    #event = {}


    def __init__(self, device, res_type):
        self.unique_name = (device, res_type)
        # BaseCamera.event[self.unique_name] = CameraEvent()

        BaseCamera.frame[self.unique_name] = (-1, np.zeros((480,640,3), np.uint8))

        if self.unique_name not in BaseCamera.threads:
            # 在Web的生命周期内被第一个浏览器初始化
            BaseCamera.threads[self.unique_name] = None

        if BaseCamera.threads[self.unique_name] is None:

            BaseCamera.last_access[self.unique_name] = time.time()

            # start background frame thread
            BaseCamera.threads[self.unique_name] = threading.Thread(target=self._thread)
            BaseCamera.threads[self.unique_name].start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)
        

    @staticmethod
    def frames(self, image_hub):
        """Generator that returns frames."""
        raise RuntimeError('Must be implemented by subclasses.')


    def _thread(self):
        """background thread."""
        device = self.unique_name[0]
        res_type = self.unique_name[1]
        port = 5555+device*10+res_type
        image_hub = imagezmq.ImageHub(open_port='tcp://*:{}'.format(port))

        print("Open port: {}.".format(port))
        
        frames_iterator = self.frames(image_hub)
        for frame_id,frame in frames_iterator:
            BaseCamera.frame[self.unique_name] = frame_id, frame
            if frame_id == 1:
                self.start_time = time.time()
            # BaseCamera.event[self.unique_name].set()
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            # if time.time() - BaseCamera.last_access[unique_name] > 10:
            #     frames_iterator.close()
            #     print('Stopping camera thread due to inactivity.')
            #     break


    def get_frame(self):
        """Return the current frame to flask."""
        BaseCamera.last_access[self.unique_name] = time.time()
        # # wait for a signal from the camera thread
        # BaseCamera.event[self.unique_name].wait()     # 如果新的一帧没有到来，就等待
        # BaseCamera.event[self.unique_name].clear()    # 同时更新
        

        return BaseCamera.frame[self.unique_name]

    #def pause(self):
