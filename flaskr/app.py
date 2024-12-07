""" 页面逻辑处理
    1. 进入参数配置页面
    2. 点击提交按钮开始跑程序，并显示运行结果
    3. 在运行过程中点下暂停/开始按键
"""

import os
from importlib import import_module
from flask import Flask, request, render_template, Response, make_response, url_for, redirect, flash
from gevent import pywsgi
from flask_bootstrap import Bootstrap4
from forms import MtmctForm, DispForm
import time
import cv2
import numpy as np
from threading import Thread
from launcher import Launcher

app = Flask(__name__)
bootstrap = Bootstrap4(app)
app.config['SECRET_KEY'] = 'hard to guess string'


@app.route('/', methods=["GET", "POST"])
def index():
    """ 项目参数配置页面 """
    form = MtmctForm()

    # 在配置页面上点击了提交
    if form.validate_on_submit():
        # ll = ctypes.cdll.LoadLibrary
        # lib = ll("libMTMC.so")
        # p = Thread(target = lib.Run)
        # p.start()
        resp = make_response(redirect(url_for('display_index')))
        resp.set_cookie('setting_path', form.setting_path.data)
        print('set cookie: setting path: \'{}\''.format(form.setting_path.data))
        return resp

    form.setting_path.data = request.cookies.get('setting_path')
    print('get cookie: setting path: \'{}\''.format(request.cookies.get('setting_path')))
    return render_template('param_form.html', form=form)


@app.route('/display', methods=["GET", "POST"])
def display_index():
    """ 项目运行结果显示页面 """
    run_form = DispForm()
    # 在显示页面上点击了暂停/开始按键
    if run_form.validate_on_submit():
        return render_template('./display/run.html', form=run_form)
    return render_template('./display/run.html', form=run_form)


def gen(camera_stream):
    total_time = 0
    print("camera_stream {}.{} initialized.".format(camera_stream.unique_name[0], camera_stream.unique_name[1]))
    while True:
        frame_id, frame = camera_stream.get_frame()

        # if frame_id != -1 and unique_name[1] == 0:
        #     time_now = time.time()
        #     total_time = time_now - camera_stream.start_time
        #     fps = frame_id / total_time

        #     cv2.putText(frame, "[%d] FPS: %.2f" % (frame_id,fps), (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
        #                     (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/result_feed/<node_id>/<res_type>')
def result_feed(node_id, res_type):
    # 原理见 https://blog.csdn.net/u012655441/article/details/124798348
    return Response(gen(Launcher(int(node_id), int(res_type))),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)