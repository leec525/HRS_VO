#include "FrameDrawer.hpp"
#include "Group.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

using namespace std;

namespace HRS_VO
{

    FrameDrawer::FrameDrawer(int group_size) : mnGroupSize(group_size)
    {
        mState = Group::NO_IMAGES_YET;
        // 初始化图像显示画布
        // 包括：图像、特征点连线形成的轨迹（初始化时）、框（跟踪时的MapPoint）、圈（跟踪时的特征点）
        // ！！！固定画布大小为640*480
        // 显示当前帧的跟踪点及上一帧对应的匹配点，两帧图像显示在一个窗口，进行图像拼接？
        // mvLeftImgs.resize(mnGroupSize, cv::Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0)));
        // mvRightImgs.resize(mnGroupSize, cv::Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0)));
        // mvLeftLastImgs.resize(mnGroupSize, cv::Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0)));
        // mvRightLastImgs.resize(mnGroupSize, cv::Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0)));
        // mvLeftDrawPad.resize(mnGroupSize, cv::Mat(480, 640*2, CV_8UC3, cv::Scalar(0,0,0)));
        // mvRightDrawPad.resize(mnGroupSize, cv::Mat(480, 640*2, CV_8UC3, cv::Scalar(0,0,0)));

        mvnLeftFrameIds.resize(mnGroupSize);
        mvnRightFrameIds.resize(mnGroupSize);

        cout << "[C++] rigister" << flush;
        Py_Initialize();
        ImportNumPySupport();
        cout << " & initialize." << endl;

        cout << "[C++] python initialized." << endl;
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./')");
        PyRun_SimpleString("import os");
        PyRun_SimpleString("print(os.listdir(sys.path[-1]))");

        cout << "[C++] func->import modules from " << flush;

        // 导入脚本内的函数
        PyObject *pModule = PyImport_ImportModule("SendFrame");
        if (pModule == nullptr)
            cout << endl
                 << "path failed." << endl;

        cout << "\"SendFrame.py\": " << flush;
        PyObject *pFunRegister = PyObject_GetAttrString(pModule, "register");
        cout << "register, " << flush;
        mpFunSendFrame = PyObject_GetAttrString(pModule, "send");
        cout << "send, " << flush;
        mpFunQuit = PyObject_GetAttrString(pModule, "quit");
        cout << "quit." << endl;

        // 注册PyFrameSender
        for (int i = 0; i < group_size; i++)
        {
            // 'i' 表示int变量
            PyEval_CallObject(pFunRegister, Py_BuildValue("(i)", i));

            if (PyErr_Occurred())
                PyErr_Print();
        }
    }

    void FrameDrawer::Update(Group *pGroup)
    {
        // unique_lock<mutex> lock(mMutex);

        int grpId = pGroup->mnGroupIdx;

        mvnLeftFrameIds[grpId] = pGroup->mCurrentRowPair.left_frame_cnt;
        mvnRightFrameIds[grpId] = pGroup->mCurrentRowPair.right_frame_cnt;

        vector<cv::Mat> vDispFrames;
        vector<string> vDispNames;
        vDispFrames.assign(pGroup->mvDispFrames.begin(),
                           pGroup->mvDispFrames.end());
        vDispNames.assign(pGroup->mvsDispNames.begin(),
                          pGroup->mvsDispNames.end());

        for (int i = 0; i < vDispFrames.size(); i++)
            SendFrame(grpId, vDispFrames[i], i);

        mState = static_cast<int>(pGroup->mLastProcessedState);
    }

    bool FrameDrawer::Image2Numpy_UBYTE(cv::Mat &srcImage,
                                        PyObject *&pPyArray)
    {
        // step 0 检查图像是否非空
        if (srcImage.empty())
            return false;

        // step 1 生成临时的图像数组，图像数组暂时是缓存在成员变量里。检查合法性
        // 获取图像的尺寸
        size_t w = srcImage.size().width,
               h = srcImage.size().height,
               c = srcImage.channels();

        // 生成
        uchar *mpb8ImgTmpArray = new unsigned char[w * h * c];

        size_t iRows = srcImage.rows,
               iCols = srcImage.cols * c;

        // 判断这个图像是否是连续存储的，如果是连续存储的，那么意味着我们可以把它看成是一个一维数组，从而加速存取速度
        if (srcImage.isContinuous())
        {
            iCols *= iRows;
            iRows = 1;
        }

        // 指向图像中某个像素所在行的指针
        unsigned char *p;

        int id = -1;
        for (int i = 0; i < iRows; i++)
        {
            // get the pointer to the ith row
            p = srcImage.ptr<uchar>(i);
            // operates on each pixel
            for (int j = 0; j < iCols; j++)
            {
                mpb8ImgTmpArray[++id] = p[j]; // 连续空间
            }
        }

        // step 2 生成三维的numpy
        npy_intp Dims[3] = {(int)h,
                            (int)w,
                            (int)c}; // 注意这个维度数据！

        // cout<<"numpy maked."<<endl;
        pPyArray = PyArray_SimpleNewFromData(
            3,                // 有几个维度
            Dims,             // 数组在每个维度上的尺度
            NPY_UBYTE,        // numpy数组中每个元素的类型
            mpb8ImgTmpArray); // 用于构造numpy数组的初始数据

        return true;
    }

    void FrameDrawer::SendFrame(int grp_id, cv::Mat frame, int type)
    {
        // TODO: 是否最后一帧
        cv::Mat res_frame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        if (frame.channels() == 1)
        {
            vector<cv::Mat> matParts;
            matParts.reserve(3);
            matParts.push_back(frame);
            matParts.push_back(frame);
            matParts.push_back(frame);
            cv::merge(matParts, res_frame);
        }
        else
            res_frame = frame;

        // step 1 将图片转换成为NumPy的数组的形式
        PyObject *pPyImageArray = nullptr;
        if (!Image2Numpy_UBYTE(res_frame, pPyImageArray))
            return;
        if (!pPyImageArray)
            return;

        PyObject *mpPyArgList = PyTuple_New(4);
        PyTuple_SetItem(mpPyArgList, 0, pPyImageArray);
        PyTuple_SetItem(mpPyArgList, 1, Py_BuildValue("i", mvnLeftFrameIds[grp_id]));
        PyTuple_SetItem(mpPyArgList, 2, Py_BuildValue("i", grp_id));
        PyTuple_SetItem(mpPyArgList, 3, Py_BuildValue("i", type));

        // step 3 运行
        {
            unique_lock<mutex> lock(mPyMutex);
            PyEval_CallObject(mpFunSendFrame, mpPyArgList);
            if (PyErr_Occurred())
                PyErr_Print();
        }
        // cout<<"done"<<endl;
    }

    bool FrameDrawer::Finish()
    {
        PyObject *mpPyArgs = nullptr;
        PyEval_CallObject(mpFunQuit, mpPyArgs);
        if (PyErr_Occurred())
            PyErr_Print();

        cout << "[C++] FrameDrawer released." << endl;

        return true;
    }

} // namespace