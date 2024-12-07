#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>

#include <VideoCapture.hpp>
#include "Group.hpp"

#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace HRS_VO;

VideoCapture::VideoCapture(int camera, Group *pGroup, unsigned char group_loc, string save_path = "~/record", unsigned char save_mode = NO_SAVE) : mSaveMode(save_mode), mSavePath(save_path), mpGroup(pGroup), mnGroupLoc(group_loc), mCamera(camera)
{
    mpCap = new cv::VideoCapture(camera);
    // 相机设备序列号超过7，会报错  VIDEOIO ERROR: V4L: index 8 is not correct!
    // 检查相机是否正确打开，如果没有，再次打开相机
    if (!(mpCap->isOpened()))
        mpCap->open("/dev/video" + to_string(camera));

    mpCap->set(cv::CAP_PROP_FRAME_WIDTH, pGroup->msCamGroupConfig.width);
    mpCap->set(cv::CAP_PROP_FRAME_HEIGHT, pGroup->msCamGroupConfig.height);
    mpCap->set(cv::CAP_PROP_FPS, pGroup->msCamGroupConfig.fps);

    mnCount = 0;

    mSize = cv::Size(pGroup->msCamGroupConfig.width, pGroup->msCamGroupConfig.height);

    // 设置保存路径
    if (mSaveMode & SAVE_VIDEO)
    {
#if CV_VERSION_MAJOR >= 4
        mVideoWriter.open(mSavePath + "/cam" + to_string(mCamera) + "_left.avi", cv::CAP_OPENCV_MJPEG, CAMERA_FPS, cv::Size(IMAGE_WIDTH / 2, IMAGE_HIGHT)); // opencv4.0+
#else
        mVideoWriter.open(mSavePath + "/cam" + to_string(mCamera) + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), CAMERA_FPS, mSize); // opencv3.0
#endif

        // if ( !(mVideoWriter.isOpened() && ((mnGroupLoc & 0x03) || mVideoWriterRight.isOpened())) )
        // {
        //     //MessageBoxA(NULL, "Save Failure", "Save", MB_OK);
        //     cout << "Can't open video save path!" << endl;
        //     mSaveMode = NO_SAVE;
        // }
    }

    if (mSaveMode & SAVE_FIGURE)
    {
        struct stat buffer;
        if (!stat(mSavePath.c_str(), &buffer) == 0)
        {
            // MessageBoxA(NULL, "Save Failure", "Save", MB_OK);
            cout << "Can't open figure save path!" << endl;
            mSaveMode = NO_SAVE;
        }
    }
}

VideoCapture::~VideoCapture()
{
    if (mSaveMode & SAVE_VIDEO)
    {
        // 写入文件关闭
        mVideoWriter.release();
    }
}

bool VideoCapture::WaitforReady()
{
    cv::Mat img;
    string win_name;
    win_name = "cam" + to_string(mCamera) + "_Video";

    while (mpCap->isOpened())
    {
        mpCap->read(img); // 从摄像头中读取当前这一帧
        cv::imshow(win_name, img);

        int k_value = cv::waitKey(10);
        if (k_value == 's')
        {
            cout << "开始..." << endl;
            cv::destroyWindow(win_name); /*显示窗口销毁*/
            return true;
        }
    }
    cout << "Camera " << mCamera << " is not opened!" << endl;
    return false;
}

bool VideoCapture::Capture()
{
    cv::Mat img;
    mpCap->read(img); // 从摄像头中读取当前这一帧
    // display
    string win_name = "cam" + to_string(mCamera) + "_Video";
    cv::imshow(win_name, img);
    // cv::waitKey(10);

    if (mnGroupLoc == BOTH_CAMERA) // stereo
    {
        Stereoframe frame;

        if (Segmentation(img, frame))
        {
            mpGroup->UpdateFrame(frame);
        }
        else
        {
            cerr << "segmentation false!" << endl;
            return false;
        }
    }
    else // mono
    {
        mpGroup->UpdateFrame(img, mnCount, mnGroupLoc);
    }

    // 保存
    if (mSaveMode & SAVE_VIDEO)
        mVideoWriter.write(img);

    // if(mSaveMode & SAVE_FIGURE)
    // {
    //     string path_left = mSavePath+"/cam"+to_string(mCamera)+"_leftImage/"+to_string(frame.count)+".jpg";
    //     cv::imwrite(path_left, frame.left);
    //     if(mnGroupLoc == BOTH_CAMERA)
    //     {
    //         string path_right = mSavePath+"/cam"+to_string(mCamera)+"_rightImage/"+to_string(frame.count)+".jpg";
    //         cv::imwrite(path_right, frame.right);
    //     }
    // }

    mnCount++;

    return true;
}

bool VideoCapture::Segmentation(const cv::Mat &img_src, Stereoframe &frame)
{
    // cv::resize(img_src, img_src, cv::Size(1280, 480));  //set size of image
    frame.left = img_src(cv::Rect(0, 0, IMAGE_WIDTH / 2, IMAGE_HIGHT));                // split left image
    frame.right = img_src(cv::Rect(IMAGE_WIDTH / 2, 0, IMAGE_WIDTH / 2, IMAGE_HIGHT)); // split right image
    frame.count = mnCount;

    return true;
}

bool VideoCapture::isOpened()
{
    return mpCap->isOpened();
}

void VideoCapture::Close()
{
    mpCap->release();
}