#ifndef GROUP_H
#define GROUP_H

#define LEFT_IMAGE 1
#define RIGHT_IMAGE 2
#define BOTH_IMAGE 4
#define MEAN_THREASHOLD 10 // led初始化时灰度阈值
// #define BLOCK_SIZE 20  //二进制描述符的位数
// #define BINARY_THRESHOLD 12  //二进制描述符的阈值
#define BOARDER 20 // 边缘宽度
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 480
#define CAMERA_FPS 30

#define ROWRANGE 10

#define SYNC true
#define ASYNC false

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <queue>

#include "HRSmatcher.hpp"
#include "FrameDrawer.hpp"

typedef std::queue<std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector2d>> para_queue;
typedef std::queue<std::pair<std::vector<Eigen::Matrix4d>, std::vector<Eigen::Matrix4d>>> trace_queue;

namespace HRS_VO
{

    struct Stereoframe
    {
        unsigned int count;
        cv::Mat left, right;
        std::vector<double> left_mean, right_mean;
        cv::Mat left_binary, right_binary;
        std::vector<int> Minham;
        // 时间戳
        Stereoframe *next, *prev;
        std::vector<cv::Mat> left_rotation, right_rotation; // 用于存储每一行的旋转矩阵
    };

    struct StereoRow
    {
        int left_cnt, right_cnt; // 总行数
        int left_frame_cnt, right_frame_cnt;
        int left_row, right_row; // 当前帧的行数
        Stereoframe *left_stereo_ptr, *right_stereo_ptr;
    };

    struct CamConfiguration
    {
        cv::Mat K1, K2;                         // 相机内参矩阵
        cv::Mat D1, D2;                         // 相机畸变参数矩阵
        cv::Mat R_1_cl, R_2_cl;                 // 相机相对于簇中心的旋转
        cv::Mat t_1_cl, t_2_cl;                 // 相机相对于簇中心的平移
        cv::Mat T_1_cl, T_2_cl, T_cl_1, T_cl_2; // 相机n相对于簇中心的位姿矩阵
        int width;
        int height;
        int fps;
        float focal_length;
        float baseline;
        int type; // false: single; true: stereo
    };

    typedef struct Result
    {
        bool flag;                          // 匹配成功标志
        cv::Point2d curr_px_coord;          // 在当前帧上的图像坐标
        cv::Point2d matched_px_coord;       // 在上一帧上的匹配点坐标
        Eigen::Vector4d last_matched_coord; // 上一帧匹配点在相机坐标系下的齐次坐标
        double st;                          // 运动偏移
        double z_last;                      // 上一帧匹配点的深度
        double confidence_st;               // st的置信度
        int index;                          // 匹配点所在原始行的行索引
    } Result;

    class HRSmatcher;
    class FrameDrawer;

    class Group
    {
    public:
        /** @brief  拷贝
         */
        Group(Group &g);

        /** @brief  构造
         */
        Group(int id);

        /** @brief  组内相机有新的图像
         * @param  img  图像
         * @param  cnt  第几帧
         * @param  camera   组内的哪个相机，LEFT_IMAGE/RIGHT_IMAGE (暂时只考虑两个的情况)
         * @return 更新完毕则返回true
         */
        void UpdateFrame(cv::Mat &img, int cnt, unsigned char camera);

        /** @brief  利用外部灯光进行相机行同步
         */
        void Initialize();

        // TODO: 写注释，改函数的名字更贴合一点，run()
        /** @brief  分线程跟踪算法
         * @param  para_queue  返回方程参数
         * @param  pTrace      传入运动估计结果
         * @return
         */
        void Tracking(para_queue &ret_params, trace_queue &pTrace);

        /** @brief  跟踪算法
         * @param  CurrImgBinPair      first:当前帧图像  second:图像的二进制描述符
         * @param  LastStereoImgPair   first:上一帧图像  second:上一帧图像的匹配图像
         * @param  row                 当前行图像的行索引
         * @param  res                 存储结果结构体的数组
         */
        void Track(const std::pair<cv::Mat, cv::Mat> &CurrImgBinPair,
                   const std::pair<cv::Mat, cv::Mat> &LastStereoImgPair,
                   const int row,
                   HRSmatcher *pMatcher,
                   std::vector<Result> &res);

        /** @brief  设置组内相机的参数：内参矩阵、相对于簇中心的位姿矩阵
         * @return 设置完毕则返回true
         */
        bool SetModel(const std::string config_path);

        /** @brief  加载显示模块 */
        void SetFrameDrawer(FrameDrawer *pFrameDrawer);

        /** @brief  计算单应矩阵
         * @return 返回矫正后的图像矩阵
         */
        void Homography(para_queue &ret_params, int src_row, cv::Mat &H);

        /** @brief  双目图像更新
         * @param  frame  双目帧
         * @return 更新完毕则返回true
         */
        void UpdateFrame(Stereoframe &frame);

        /** @brief  处理左右图像的下一行
         * @return 处理完则返回true
         * // 这两个函数后面需要重构成一个函数，left和right作为函数参数
         */
        bool NextLeft();
        bool NextRight();

        /** @brief  准备处理下一行
         */
        void next();

        /** @brief 画图
         */
        void DrawFrame(cv::Mat &leftDrawPad, cv::Mat &rightDrawPad);

        /** @brief
         */
        void Update(Eigen::Matrix4d &pose_pair);

        // Group Variables
        int mnGroupIdx;
        CamConfiguration msCamGroupConfig; // 组内相机的参数
        // Result msleftRes, msrightRes;    //单相机跟踪函数返回的结果
        // cv::Mat mRefRotation;  //参考行的旋转矩阵

        // State Variables
        bool mbStart;                      // 系统开始跟踪的标志
        bool mbLeftInit, mbRightInit;      // 初始化状态，true:正在初始化，false:初始化完成
        int mnInitLeftCnt, mnInitRightCnt; // LED灯初始化结果返回的左右相机发生边缘效应时的总行数

        // Tracking states
        /// 跟踪状态类型
        enum eTrackingState
        {
            SYSTEM_NOT_READY = -1, // 系统没有准备好的状态 TODO:还没用上
            NO_IMAGES_YET = 0,     // 当前无图像
            NOT_INITIALIZED = 1,   // 有图像但是没有完成初始化
            OK = 2,                // 正常时候的工作状态
            LOST = 3               // 系统已经跟丢了的状态 TODO:没用上
        };

        /// 跟踪状态
        eTrackingState mState;
        /// 上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到
        eTrackingState mLastProcessedState;

        // Frame Variables
        int mnLeftCount, mnRightCount;
        std::queue<Stereoframe> mqCandidateFramePairs; // 如果多线程接收图像，需要赋予原子操作

        std::pair<int, int> mMatchedRowIdxPair;

        // Row Variables
        StereoRow mCurrentRowPair;
        StereoRow mLastRowPair;

        // TODO: modified
        std::vector<Result> mvLeftMatchedRes, mvRightMatchedRes;

        std::vector<std::vector<Eigen::Matrix4d>> mLeftRowPoses;
        std::vector<std::vector<Eigen::Matrix4d>> mRightRowPoses;

        int mnLeftProcCnt, mnRightProcCnt; // 当前正在处理的帧
        std::vector<cv::Mat> mvDispFrames; // 用于可视化
        std::vector<std::string> mvsDispNames;

    protected:
        /** @brief 注册想要显示的图像，传入顺序即显示顺序
         * @param pFrame  当前要显示的帧图像
         * @param name    显示窗口名字
         */
        void SetDisplay(cv::Mat &pFrame, std::string name);

        void ClearDisplay();

        // 存全局位姿，当前帧和上一帧每行位姿
        std::vector<cv::Mat> mvFormerPose, mvCurrentPose;

        // HRSmatcher
        HRSmatcher *mpLeftMatcher;
        HRSmatcher *mpRightMatcher;

        // 帧绘制器
        FrameDrawer *mpFrameDrawer;
    };

} // namespace HRS_VO

#endif