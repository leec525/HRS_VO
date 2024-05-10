#include <iostream>
#include <thread>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "VideoCapture.hpp"
#include "Group.hpp"

#include "FrameDrawer.hpp"

#define GROUP_SIZE 3

using namespace std;
using namespace HRS_VO;

void ImageCapture(vector<Group *> &vpGroup);
bool WaitForNewParam(vector<para_queue> params);

vector<para_queue> vqParamReturn;
vector<trace_queue> vqTrace;

int main(int argc, char **argv)
{
    // 1. 配置
    // 1.1）创建相机组Group对象
    vector<Group *> vpGroup;
    vpGroup.resize(GROUP_SIZE);
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        vpGroup[i] = new Group(i); // 相机组
        cout << "/home/simulator/LIC/hrs_vo/camera_models/cam_config_group" + to_string(i) + ".xml" << endl;
        vpGroup[i]->SetModel("/home/simulator/LIC/hrs_vo/camera_models/cam_config_group" + to_string(i) + ".xml");
    }
    cout << "SetModel done" << endl;

    // 1.2）创建画图对象
    FrameDrawer *pFrameDrawer = new FrameDrawer(GROUP_SIZE);
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        vpGroup[i]->SetFrameDrawer(pFrameDrawer);
    }
    cout << "FrameDrawer done" << endl;

    // Thread1: 开一个线程用于获取所有相机的图像，并塞入所在Group
    thread *CaptureThread = new thread(&ImageCapture, ref(vpGroup));

    // Thread2: 开Group线程，先进行相机同步
    vector<thread *> vpGroupThread;
    vpGroupThread.resize(GROUP_SIZE);
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        vpGroupThread[i] = new thread(&Group::Initialize, vpGroup[i]);
        // vpGroupThread[i] = new thread(&vpGroup[i]::Initialize);
    }
    cout << "Initializing..." << endl;
    // 等待初始化完毕
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        vpGroupThread[i]->join();
        // 若灯闪后长时间没反应，说明有相机同步失败了
    }
    cout << "初始化完毕! 开始跟踪." << endl;

    // 2. 开启跟踪线程
    // TODO：改！
    vqParamReturn.reserve(GROUP_SIZE);
    vqTrace.reserve(GROUP_SIZE);
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        para_queue param_ret;
        trace_queue Trace;
        vqParamReturn.push_back(param_ret);
        vqTrace.push_back(Trace);
        vpGroupThread[i] = new thread(&Group::Tracking, vpGroup[i], ref(vqParamReturn[i]), ref(vqTrace[i]));
    }
    cout << "跟踪进行中..." << endl;

    // 3. 位姿计算
    // 一个存储从初始位置开始的位姿矩阵的容器
    vector<Eigen::Matrix4d> pose;
    pose.reserve(100000); // 全局位姿矩阵
    while (1)
    {
        while (WaitForNewParam(vqParamReturn))
            ;
        Eigen::MatrixXd A(GROUP_SIZE * 6, 6);
        Eigen::VectorXd b(GROUP_SIZE * 6);
        vector<vector<Eigen::Matrix4d>> vvLastLeftPose, vvLastRightPose;
        vvLastLeftPose.reserve(GROUP_SIZE);
        vvLastRightPose.reserve(GROUP_SIZE);

        Eigen::Matrix4d FusionPose = Eigen::Matrix4d::Zero();

        for (int i = 0; i < GROUP_SIZE; i++)
        {
            // 聚合A, b
            A << A, vqParamReturn[i].front().first;
            b << b, vqParamReturn[i].front().second;
            vqParamReturn[i].pop();

            //
            vvLastLeftPose.push_back(vqTrace[i].front().first);
            vvLastRightPose.push_back(vqTrace[i].front().second);
            vqTrace[i].pop();
        }

        // 位姿求解
        // 线性方程Ax=b，当系数矩阵A超定时，最小二乘解为 x = (A^T * A)^(-1) * A^T *b
        // 高斯牛顿法系数矩阵可能不正定，解可能不稳定
        Eigen::MatrixXd A_T = A.transpose();
        Eigen::MatrixXd J = A_T * A;
        Eigen::MatrixXd J_inv = J.inverse();
        Eigen::VectorXd y = J_inv * A_T * b;
        cout << "y=" << y << endl;

        Eigen::Matrix4d RelativeMotion;
        RelativeMotion << 1.0, -y[2], y[1], y[3],
            y[2], 1.0, -y[0], y[4],
            -y[1], y[0], 1.0, y[5],
            0.0, 0.0, 0.0, 1.0;

        // 计算每个相机的全局T：从初始位姿累乘得到当前帧当前行的位姿矩阵
        // 当前行的位姿矩阵 = 匹配行的位姿矩阵 * 帧间相对位姿矩阵
        for (int i = 0; i < GROUP_SIZE; i++)
        {
            Eigen::Matrix4d T_cl_1, T_cl_2;
            cv::cv2eigen(vpGroup[i]->msCamGroupConfig.T_cl_1, T_cl_1);
            cv::cv2eigen(vpGroup[i]->msCamGroupConfig.T_cl_2, T_cl_2);

            for (int j = 0; j < 3; j++)
            {
                Eigen::Matrix4d CurrLeftPose, CurrRightPose;

                // 由当前相机行的位姿矩阵还原出的相机簇位姿
                CurrLeftPose = vvLastLeftPose[i][j] * RelativeMotion * T_cl_1;
                CurrRightPose = vvLastRightPose[i][j] * RelativeMotion * T_cl_2;
                // TODO：融合更新，待改进融合算法
                FusionPose += CurrLeftPose + CurrRightPose;
            }
        }

        // 直接计算算数平均是不可靠的，应根据相机匹配的可靠性加权算平均
        // TODO：加权平均
        FusionPose /= GROUP_SIZE * 3 * 2;

        // TODO: 更新
        for (int i = 0; i < GROUP_SIZE; i++)
        {
            vpGroup[i]->Update(FusionPose);
            // 每个Group得到两个全局位姿
        }
        // 融合每个Group的全局位姿，得到一个统一的全局位姿，存储进pose

        pose.push_back(FusionPose);
    }

    // 等待提取图像线程退出
    CaptureThread->join();

    // 等待Group线程退出
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        vpGroupThread[i]->join();
    }

    system("pause"); // 暂停
}

void ImageCapture(vector<Group *> &vpGroup)
{
    cout << "CaptureThread launch: [" << flush;
    cout << vpGroup.size() << "] groups to be cap." << endl;

    string save_path = "/home/simulator/LIC/hrs_vo/dataset";
    vector<VideoCapture> vCap;
    vCap.reserve(GROUP_SIZE * 2 - 1);

    // 打开相机并与Group进行绑定，但是不同的相机其实需要不同的capture方式及cap号，因此多个相机不应该使用for，而应该使用配置文件，或一行一行地写代码配置
    auto pGroupIt = vpGroup.begin();
    int capCnt = 0;
    int grpCnt = 0;
    for (; pGroupIt != vpGroup.end(); pGroupIt++, grpCnt++)
    {
        if (vpGroup[grpCnt]->msCamGroupConfig.type == 1)
        {
            // opencv4会识别音频设备，相机视频索引为02468，需要跳过音频索引13579
            vCap.push_back(VideoCapture(capCnt * 2, vpGroup[grpCnt], CAMERA1, save_path, NO_SAVE));       // 打开第i个相机
            vCap.push_back(VideoCapture((capCnt + 1) * 2, vpGroup[grpCnt], CAMERA2, save_path, NO_SAVE)); // 打开第i+1个相机
            vCap[capCnt].WaitforReady();
            vCap[capCnt + 1].WaitforReady();
            capCnt += 2;
        }
        else if (vpGroup[grpCnt]->msCamGroupConfig.type == 2)
        {
            vCap.push_back(VideoCapture(capCnt * 2, vpGroup[grpCnt], BOTH_CAMERA, save_path, NO_SAVE)); // 打开第i个相机
            vCap[capCnt].WaitforReady();
            capCnt++;
        }
        else
        {
            cerr << "wrong type" << endl;
        }
    }

    while (1)
    {
        auto capIt = vCap.begin();
        // pGroupIt = vpGroup.begin();
        for (; capIt != vCap.end(); capIt++)
        {
            if (capIt->isOpened())
            {
                capIt->Capture();
            }
            else
            {
                cerr << "camera " << capIt->getCameraID() << " doesn't open" << endl;
            }
        }
    }
}

bool WaitForNewParam(vector<para_queue> params)
{
    for (int i = 0; i < GROUP_SIZE; i++)
    {
        if (vqParamReturn[i].empty())
            return true;
    }
    return false;
}
