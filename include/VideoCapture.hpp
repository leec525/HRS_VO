#ifndef VIDEOCAPTURE_H
#define VIDEOCAPTURE_H

#include <Group.hpp>

#define IMAGE_WIDTH 1280
#define IMAGE_HIGHT 480
#define CAMERA_FPS 30

#define NO_SAVE 0
#define SAVE_VIDEO 1
#define SAVE_FIGURE 2

#define CAMERA1 1
#define CAMERA2 2
#define BOTH_CAMERA 3

namespace HRS_VO
{

    class VideoCapture
    {
    public:
        /**************************************************************
         * @brief  初始化相机
         * @param camera    相机号
         * @param pGroup    相机所要绑定的group
         * @param group_loc 绑定到该group的哪一边(CAMERA1|CAMERA2|BOTH_CAMERA)
         * @param save_path 图像的保存路径，单个相机默认保存在该路径下的left文件夹下
         * @param save_mode   是否保存(NO_SAVE|SAVE_VIDEO|SAVE_FIGURE)
         * @return 初始化完毕则返回true
         *************************************************************/
        VideoCapture(int camera, Group *pGroup, unsigned char group_loc, std::string save_path, unsigned char save_mode);
        ~VideoCapture();

        /**************************************************************
         * @brief  摄像头刚打开时会有一段时间黑屏，需要等相机稳定了再开始
         * @return 初始化完毕则返回true
         *************************************************************/
        bool WaitforReady();

        /**************************************************************
         * @brief  获取相机图像
         *************************************************************/
        bool Capture();

        int getCameraID() { return mCamera; }

        /**************************************************************
         * @brief  返回相机的打开状态
         *************************************************************/
        bool isOpened();

        /**************************************************************
         * @brief  关闭相机
         *************************************************************/
        void Close();

    protected:
        unsigned int mnCount;
        int mCamera;

        cv::Size mSize;

        cv::VideoCapture *mpCap; // both_camera, camera1 -> mpCap; camera2 -> mpCapRight
        cv::VideoWriter mVideoWriter;
        unsigned char mDisplay;
        unsigned char mSaveMode;
        std::string mSavePath;

        Group *mpGroup;
        unsigned char mnGroupLoc;

        /**************************************************************
         * @brief  将多目相机的画面进行分割
         * @param img_src 相机直出的图像
         * @param frame 左右图像的结构体
         *************************************************************/
        bool Segmentation(const cv::Mat &img_src, Stereoframe &frame);
    };
}
#endif