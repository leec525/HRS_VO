/**
 * @file Group.cpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 相机组跟踪问题
 * @version 0.1
 * @date 2023-05-26
 *
 */

#include "Group.hpp"
#include "HRSmatcher.hpp"
#include "HomographyMap.hpp"
#include "FrameDrawer.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <chrono>
#include <thread>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 4
#include <opencv2/imgproc/types_c.h> // opencv4.0+
#endif

using namespace std;
using namespace chrono;
using namespace HRS_VO;

Group::Group(Group &g) : mnGroupIdx(g.mnGroupIdx)
{
}

Group::Group(int id) : mnGroupIdx(id)
{
    mState = NO_IMAGES_YET;
    mbRightInit = true;
    mbLeftInit = true;

    mnRightCount = 0;
    mnLeftCount = 0;

    mnLeftProcCnt = 0;
    mnRightProcCnt = 0;

    mLeftRowPoses.resize(10000);
    mRightRowPoses.resize(10000);
    for (int i = 0; i < 10000; i++)
    {
        mLeftRowPoses[i].reserve(msCamGroupConfig.height);
        mRightRowPoses[i].reserve(msCamGroupConfig.height);
    }
}

void Group::Initialize()
{
    while (1)
    {
        if (mState == NO_IMAGES_YET)
            continue; // 等待第一个RowPair到来

        // 别名，为了代码简洁
        int left_row = mCurrentRowPair.left_row;
        int right_row = mCurrentRowPair.right_row;

        Eigen::Matrix4d T_1_cl, T_2_cl;
        cv::cv2eigen(msCamGroupConfig.T_1_cl, T_1_cl);
        cv::cv2eigen(msCamGroupConfig.T_2_cl, T_2_cl);

        Stereoframe *pLeftStereo = mCurrentRowPair.left_stereo_ptr;
        Stereoframe *pRightStereo = mCurrentRowPair.right_stereo_ptr;
        Stereoframe *pLastLeftStereo = mLastRowPair.left_stereo_ptr;
        Stereoframe *pLastRightStereo = mLastRowPair.right_stereo_ptr;

        cv::Mat CurrentLeftRow = pLeftStereo->left.row(left_row);
        cv::Mat CurrentRightRow = pRightStereo->right.row(right_row);

        // 计算灰度均值
        cv::Scalar leftrow_mean, rightrow_mean;
        leftrow_mean = cv::mean(CurrentLeftRow);
        rightrow_mean = cv::mean(CurrentRightRow);
        // cv::Scalar leftrow_stdDev, rightrow_stdDev;
        // cv::meanStdDev(CurrentLeftRow, leftrow_mean, leftrow_stdDev); // 没有用上方差
        // cv::meanStdDev(CurrentRightRow, rightrow_mean, rightrow_stdDev);

        pLeftStereo->left_mean.push_back(leftrow_mean[0]);
        pRightStereo->right_mean.push_back(rightrow_mean[0]);

        // 识别灯光
        if (pLastLeftStereo != nullptr) // 如果不是第一帧
        {
            bool left_init_indicator = (pLeftStereo->left_mean[left_row] - pLastLeftStereo->left_mean[left_row]) > MEAN_THREASHOLD;
            bool right_init_indicator = (pRightStereo->right_mean[right_row] - pLastRightStereo->right_mean[right_row]) > MEAN_THREASHOLD;

            if (mbLeftInit && left_init_indicator)
            {
                mbLeftInit = false;
                mnInitLeftCnt = mCurrentRowPair.left_cnt;
            }
            if (mbRightInit && right_init_indicator)
            {
                mbRightInit = false;
                mnInitRightCnt = mCurrentRowPair.right_cnt;
            }

            // TODO:LED灯以固定频率闪烁，返回相机组各自正在曝光的行索引，根据行索引拟合直线得到相机组哪些行在同一时刻被曝光
            if (!mbLeftInit && !mbRightInit)
            {
                // 位姿初始化，并通知main
                mCurrentRowPair.left_frame_cnt = 0;
                mCurrentRowPair.right_frame_cnt = 0;
                // 将簇中心位姿初始化为单位矩阵，相机位姿矩阵初始化*相机相对于簇中心的位姿变化

                mLeftRowPoses[0].push_back(T_1_cl);
                mRightRowPoses[0].push_back(T_2_cl);

                // 开始追踪，追踪成功
                mState = OK;
                break; // 退出线程，初始化完毕
            }
        }

        // 准备处理下一行
        next();
    }
}

void Group::Tracking(para_queue &ret_params, trace_queue &qTrace)
{
    Eigen::Matrix3d R_1_cl, R_2_cl;
    Eigen::Vector3d t_1_cl, t_2_cl;
    Eigen::Matrix4d T_1_cl, T_2_cl, T_cl_1, T_cl_2;
    cv::cv2eigen(msCamGroupConfig.R_1_cl, R_1_cl);
    cv::cv2eigen(msCamGroupConfig.R_2_cl, R_2_cl);
    cv::cv2eigen(msCamGroupConfig.t_1_cl, t_1_cl);
    cv::cv2eigen(msCamGroupConfig.t_2_cl, t_2_cl);
    cv::cv2eigen(msCamGroupConfig.T_1_cl, T_1_cl);
    cv::cv2eigen(msCamGroupConfig.T_2_cl, T_2_cl);
    cv::cv2eigen(msCamGroupConfig.T_cl_1, T_cl_1);
    cv::cv2eigen(msCamGroupConfig.T_cl_2, T_cl_2);

    while (mState == OK)
    {
        // 1. 预备变量
        // (1.1) 别名，为了代码简洁
        int left_row = mCurrentRowPair.left_row;
        int right_row = mCurrentRowPair.right_row;

        Stereoframe *pLeftStereo = mCurrentRowPair.left_stereo_ptr; // 左行所在的图像对
        Stereoframe *pRightStereo = mCurrentRowPair.right_stereo_ptr;
        Stereoframe *pLastLeftStereo = mLastRowPair.left_stereo_ptr;
        Stereoframe *pLastRightStereo = mLastRowPair.right_stereo_ptr;

        // 2. 开线程Track
        pair<cv::Mat, cv::Mat> leftImgBinPair(pLeftStereo->left, pLeftStereo->left_binary);
        pair<cv::Mat, cv::Mat> rightImgBinPair(pRightStereo->right, pRightStereo->right_binary);
        // first:当前帧图像的上一帧图像, second:上一帧的匹配图像
        pair<cv::Mat, cv::Mat> leftLastImagePair(pLastLeftStereo->left, pLastLeftStereo->right);
        pair<cv::Mat, cv::Mat> rightLastImagePair(pLastRightStereo->right, pLastRightStereo->left);

        vector<Result> vLeftRes, vRightRes; // 存储一行图像跟踪的三个点的Result

        thread *pLeftTrackThread = new thread(&Group::Track, this,
                                              move(leftImgBinPair),
                                              move(leftLastImagePair),
                                              left_row,
                                              mpLeftMatcher,
                                              ref(vLeftRes));
        thread *pRightTrackThread = new thread(&Group::Track, this,
                                               move(rightImgBinPair),
                                               move(rightLastImagePair),
                                               right_row,
                                               mpRightMatcher,
                                               ref(vRightRes));

        pLeftTrackThread->join();
        pRightTrackThread->join();

        // 将每行的匹配结果聚合成整帧的匹配结果
        // TODO:怎么聚合？vLeftRes只存了一行的三个点的Result
        mvLeftMatchedRes.insert(mvLeftMatchedRes.end(), vLeftRes.begin(), vLeftRes.end());
        mvRightMatchedRes.insert(mvRightMatchedRes.end(), vRightRes.begin(), vRightRes.end());

        // double theta_x, theta_y, theta_z, T_x, T_y, T_z;
        // Eigen::Matrix<double,6,1> Y;  //相机簇的位姿矩阵
        // Y(0,0) = theta_x;
        // Y(1,0) = theta_y;
        // Y(2,0) = theta_z;
        // Y(3,0) = T_x;
        // Y(4,0) = T_y;
        // Y(5,0) = T_z;

        Eigen::MatrixXd A66 = Eigen::MatrixXd::Zero(0, 6);
        Eigen::VectorXd B6 = Eigen::VectorXd::Zero(0, 1);
        vector<Eigen::Matrix4d> vLeftPose(3), vRightPose(3);
        for (int i = 0; i < vLeftRes.size(); i++)
        {
            Eigen::Vector4d D1 = T_cl_1 * vLeftRes[i].last_matched_coord;
            Eigen::Vector4d D2 = T_cl_2 * vRightRes[i].last_matched_coord;
            Eigen::MatrixXd A_group = Eigen::MatrixXd::Zero(2, 6); // 相机1、相机2 求位姿的系数矩阵
            A_group(0, 0) = T_1_cl(0, 2) * D1[1] - T_1_cl(0, 1) * D1[2];
            A_group(0, 1) = -T_1_cl(0, 2) * D1[0] + T_1_cl(0, 0) * D1[2];
            A_group(0, 2) = T_1_cl(0, 1) * D1[0] - T_1_cl(0, 0) * D1[1];
            A_group(0, 3) = T_1_cl(0, 0) * D1[4];
            A_group(0, 4) = T_1_cl(0, 1) * D1[4];
            A_group(0, 5) = T_1_cl(0, 2) * D1[4];
            A_group(1, 0) = D2[1] * R_2_cl(0, 2) - D2[2] * R_2_cl(0, 1);
            A_group(1, 1) = -D2[0] * R_2_cl(0, 2) + D2[2] * R_2_cl(0, 0);
            A_group(1, 2) = D2[0] * R_2_cl(0, 1) - D2[1] * R_2_cl(0, 0);
            A_group(1, 3) = R_2_cl(0, 0);
            A_group(1, 4) = R_2_cl(0, 1);
            A_group(1, 5) = R_2_cl(0, 2);

            Eigen::Vector2d B;
            B[0] = vLeftRes[i].last_matched_coord[0] + vLeftRes[i].confidence_st * vLeftRes[i].st * vLeftRes[i].z_last - T_1_cl.row(0) * D1;
            B[1] = vRightRes[i].last_matched_coord[0] + vRightRes[i].confidence_st * vRightRes[i].st * vRightRes[i].z_last - T_2_cl.row(0) * D2;

            A66 << A66, A_group;
            B6 << B6, B;

            vLeftPose.push_back(mLeftRowPoses[mCurrentRowPair.left_frame_cnt - 1][vLeftRes[i].index]);
            vRightPose.push_back(mRightRowPoses[mCurrentRowPair.right_frame_cnt - 1][vRightRes[i].index]);
        }

        pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector2d> param_ret;
        param_ret = make_pair(A66, B6);
        ret_params.push(param_ret);

        // qTrace传入上一帧匹配点所在行的相机位姿矩阵
        qTrace.push(make_pair(vLeftPose, vRightPose));

        next();
    }
}

void Group::Track(const pair<cv::Mat, cv::Mat> &CurrImgBinPair,
                  const pair<cv::Mat, cv::Mat> &LastStereoImgPair,
                  const int row,
                  HRSmatcher *pMatcher,
                  vector<Result> &res)
{
    // TODO: 如果匹配失败就res.flag = false; return

    // 1. get shift
    // (1.1) 相机前后帧匹配
    // 计算单应变换后的图像
    cv::Mat warpedImage, warpedImageIndex;
    auto warped_range = pMatcher->HomographyTrans(LastStereoImgPair.first, row, warpedImage, warpedImageIndex);
    cv::Mat warpedBinary;
    // 计算单应变换后的行图像的二进制描述符
    pMatcher->BinaryDescriptor(warpedImage, warpedBinary, warped_range);
    auto matchedRowDistPair = pMatcher->RowMatching(CurrImgBinPair.second.row(row), warpedBinary, warped_range);
    // auto timestamp1 = system_clock::now();
    // auto RM_duration = duration_cast<microseconds>(timestamp1 - start);
    // cout << "相机前后帧行匹配花费了" << double(RM_duration.count()) * microseconds::period::num / microseconds::period::den << "秒" << endl;

    // 相机每行跟踪多个点得到多个线性约束方程，利用vector实现
    vector<int> PointsX;
    PointsX.push_back(round(IMAGE_WIDTH / 4));
    PointsX.push_back(round(IMAGE_WIDTH / 2));
    PointsX.push_back(round(IMAGE_WIDTH * 2 / 3));

    for (int i = 0; i < PointsX.size(); i++)
    {
        Eigen::Vector2d pxCurr, pxLast; // 相机跟踪的像素点坐标, 前后帧匹配行上的匹配像素点坐标
        pxCurr[1] = row;
        pxCurr[0] = PointsX[i]; // 加入径向畸变后一行跟踪多个点,不同的X坐标对应不同的点

        double confidence_st, confidence_sd, st, sd; // st的置信度, sd的置信度
        auto matchedPointScoreStIndexTuple = pMatcher->PointMatching(CurrImgBinPair.first.row(row),
                                                                     warpedImage.row(matchedRowDistPair.first),
                                                                     matchedRowDistPair.first, -20, 20,
                                                                     pxCurr, warpedImageIndex);
        pxLast = get<0>(matchedPointScoreStIndexTuple);
        confidence_st = get<1>(matchedPointScoreStIndexTuple);
        int index = get<2>(matchedPointScoreStIndexTuple);
        // auto timestamp2 = system_clock::now();
        // auto PM_duration = duration_cast<microseconds>(timestamp2 - timestamp1);
        // cout << "相机前后帧点匹配花费了" << double(PM_duration.count()) * microseconds::period::num / microseconds::period::den << "秒" << endl;
        st = pxCurr[0] - pxLast[0];

        // 相机深度提取，将单应变换后的当前图像与单应变换后的对应图像进行点匹配，求双目视差
        cv::Mat warpedMatched, warpedMatchedIndex;
        Eigen::Vector2d pxLastMatched; // 双目匹配行上的匹配像素点坐标
        auto warped_matchedright_range = pMatcher->HomographyTrans(LastStereoImgPair.second, row, warpedMatched, warpedMatchedIndex);
        auto matchedPointScoreSdIndexTuple = pMatcher->PointMatching(warpedImage.row(matchedRowDistPair.first),
                                                                     warpedMatched.row(matchedRowDistPair.first),
                                                                     matchedRowDistPair.first, 10, 70,
                                                                     pxLast, warpedMatchedIndex);
        pxLastMatched = get<0>(matchedPointScoreSdIndexTuple);
        confidence_sd = get<1>(matchedPointScoreSdIndexTuple);
        sd = pxLast[0] - pxLastMatched[0];

        // 2. 利用径向畸变
        double z_last = msCamGroupConfig.focal_length * msCamGroupConfig.baseline / sd; // 计算上一帧匹配点的深度

        // 径向畸变矫正
        cv::Mat pxLastDistorted, pxLeftUndistorted;
        cv::eigen2cv(pxLast, pxLastDistorted);
        cv::undistortPoints(pxLastDistorted, pxLeftUndistorted, pMatcher->mK, pMatcher->mD);
        Eigen::Vector2d px_last;
        cv::cv2eigen(pxLeftUndistorted, px_last);

        // 将像素坐标转换为归一化平面坐标转换为相机坐标系下坐标
        Eigen::Vector3d ptLast;
        ptLast[0] = px_last[0];
        ptLast[1] = px_last[1];
        ptLast[2] = 1;
        Eigen::Matrix3d K;
        cv::cv2eigen(pMatcher->mK, K);
        Eigen::Vector3d ptLastUndistorted = z_last * K.inverse() * ptLast;

        Eigen::Vector4d ptLastMatched;
        ptLastMatched[0] = ptLastUndistorted[0];
        ptLastMatched[1] = ptLastUndistorted[1];
        ptLastMatched[2] = ptLastUndistorted[2];
        ptLastMatched[3] = 1;

        Result PointRes;
        PointRes.flag = true;
        PointRes.last_matched_coord = ptLastMatched;
        PointRes.st = st;
        PointRes.z_last = z_last;
        PointRes.confidence_st = confidence_st;
        PointRes.index = index;
        PointRes.curr_px_coord.x = pxCurr[0];
        PointRes.curr_px_coord.y = pxCurr[1]; // 存储当前帧图像行的跟踪点坐标（即特征点）
        PointRes.matched_px_coord.x = pxLast[0];
        PointRes.matched_px_coord.y = pxLast[1];
        res.push_back(PointRes);
    }
}

bool Group::SetModel(const string config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        // 从配置文件读取参数
        fs["K1"] >> msCamGroupConfig.K1;
        fs["K2"] >> msCamGroupConfig.K2;
        fs["D1"] >> msCamGroupConfig.D1;
        fs["D2"] >> msCamGroupConfig.D2;
        fs["R_1_cl"] >> msCamGroupConfig.R_1_cl;
        fs["R_2_cl"] >> msCamGroupConfig.R_2_cl;
        fs["t_1_cl"] >> msCamGroupConfig.t_1_cl;
        fs["t_2_cl"] >> msCamGroupConfig.t_2_cl;
        fs["focal_length"] >> msCamGroupConfig.focal_length;
        fs["baseline"] >> msCamGroupConfig.baseline;

        fs["width"] >> msCamGroupConfig.width;
        fs["height"] >> msCamGroupConfig.height;
        fs["fps"] >> msCamGroupConfig.fps;

        int tmp_type; // TODO：这个定义了没用到？
        fs["type"] >> msCamGroupConfig.type;

        // 配置转移矩阵
        msCamGroupConfig.T_1_cl = cv::Mat::zeros(4, 4, CV_64F);
        msCamGroupConfig.T_2_cl = cv::Mat::zeros(4, 4, CV_64F);

        msCamGroupConfig.T_1_cl(cv::Range(0, 2), cv::Range(0, 2)) = msCamGroupConfig.R_1_cl;
        msCamGroupConfig.T_1_cl(cv::Range(0, 2), cv::Range(2, 3)) = msCamGroupConfig.t_1_cl;
        msCamGroupConfig.T_1_cl.at<double>(3, 3) = 1.0;

        msCamGroupConfig.T_2_cl(cv::Range(0, 2), cv::Range(0, 2)) = msCamGroupConfig.R_2_cl;
        msCamGroupConfig.T_2_cl(cv::Range(0, 2), cv::Range(2, 3)) = msCamGroupConfig.t_2_cl;
        msCamGroupConfig.T_2_cl.at<double>(3, 3) = 1.0;

        // 预求逆
        cv::invert(msCamGroupConfig.T_1_cl, msCamGroupConfig.T_cl_1, cv::DECOMP_SVD);
        cv::invert(msCamGroupConfig.T_2_cl, msCamGroupConfig.T_cl_2, cv::DECOMP_SVD);

        mpLeftMatcher = new HRSmatcher(msCamGroupConfig.K1, msCamGroupConfig.D1);
        mpRightMatcher = new HRSmatcher(msCamGroupConfig.K2, msCamGroupConfig.D2);

        return true;
    }
    cerr << "ERROR: Config file opened error!" << endl;
    return false;
}

void Group::SetFrameDrawer(FrameDrawer *pFrameDrawer)
{
    mpFrameDrawer = pFrameDrawer;
}

void Group::UpdateFrame(Stereoframe &frame)
{
    // 转换为灰度图像
    if (frame.left.channels() == 3)
    {
        cv::cvtColor(frame.left, frame.left, CV_BGR2GRAY);
        cv::cvtColor(frame.right, frame.right, CV_BGR2GRAY);
    }

    frame.next = nullptr; // 不知道为什么push进队列后会赋奇怪的地址

    if (mState == NO_IMAGES_YET) // 若为第一帧
    {
        mLastRowPair.left_stereo_ptr = nullptr;
        mLastRowPair.right_stereo_ptr = nullptr; // 不知道为什么初始化时会赋奇怪的地址

        frame.prev = nullptr;
        mqCandidateFramePairs.push(frame);

        cout << &mqCandidateFramePairs.front() << endl;
        mCurrentRowPair.left_stereo_ptr = &mqCandidateFramePairs.front();
        mCurrentRowPair.right_stereo_ptr = &mqCandidateFramePairs.front();

        mCurrentRowPair.left_cnt = 0;
        mCurrentRowPair.right_cnt = 0;
        mCurrentRowPair.left_row = 0;
        mCurrentRowPair.right_row = 0;

        mState = NOT_INITIALIZED;
    }
    else
    {
        // auto Bin_start = system_clock::now();
        //  计算当前图像的二进制描述符
        pair<int, int> img_range(0, msCamGroupConfig.height);
        mpLeftMatcher->BinaryDescriptor(frame.left, frame.left_binary, img_range);
        mpRightMatcher->BinaryDescriptor(frame.right, frame.right_binary, img_range);
        // auto timestamp3 = system_clock::now();
        // auto BD_duration = duration_cast<microseconds>(timestamp3 - Bin_start);
        // cout << "提取二进制描述符花费了" << double(BD_duration.count()) * microseconds::period::num / microseconds::period::den << "秒" << endl;

        // 链接前后pair
        Stereoframe *prev_ptr = &mqCandidateFramePairs.back();
        frame.prev = prev_ptr;
        mqCandidateFramePairs.push(frame);
        prev_ptr->next = &mqCandidateFramePairs.back();
    }

    mnLeftCount++;
    mnRightCount++;
}

void Group::UpdateFrame(cv::Mat &img, int cnt, unsigned char camera)
{
    // 转换为灰度图像
    cv::Mat img_gray;
    if (img.channels() == 3)
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    if (mState == NO_IMAGES_YET)
    {
        // 若为第一帧

        if (mqCandidateFramePairs.empty())
        {
            Stereoframe new_stereo_frame;
            new_stereo_frame.count = cnt; // cnt=0
            new_stereo_frame.prev = nullptr;
            mqCandidateFramePairs.push(new_stereo_frame);
        }

        if (camera == LEFT_IMAGE)
        {
            mLastRowPair.left_stereo_ptr = nullptr;

            mqCandidateFramePairs.front().left = img_gray;
            mCurrentRowPair.left_stereo_ptr = &mqCandidateFramePairs.front();
            mCurrentRowPair.left_cnt = 0;
            mCurrentRowPair.left_row = 0;

            mnLeftCount++;
        }
        else
        {
            mLastRowPair.right_stereo_ptr = nullptr;

            mqCandidateFramePairs.front().right = img_gray;
            mCurrentRowPair.right_stereo_ptr = &mqCandidateFramePairs.front();
            mCurrentRowPair.right_cnt = 0;
            mCurrentRowPair.right_row = 0;

            mnRightCount++;
        }

        if (!mqCandidateFramePairs.front().left.empty() && !mqCandidateFramePairs.front().right.empty())
        {
            mState = NOT_INITIALIZED;
            mqCandidateFramePairs.back().next = nullptr;
        }
    }
    else
    {
        // 若不是第一帧

        if (camera == LEFT_IMAGE)
        {
            if (!mqCandidateFramePairs.back().left.empty())
            {
                // 若这帧图像属于新的pair

                Stereoframe new_stereo_frame;
                new_stereo_frame.count = cnt;
                new_stereo_frame.left = img_gray;
                new_stereo_frame.prev = &mqCandidateFramePairs.back();

                mqCandidateFramePairs.push(new_stereo_frame);
                mqCandidateFramePairs.back().next = nullptr;

                pair<int, int> img_range(0, msCamGroupConfig.height);
                mpLeftMatcher->BinaryDescriptor(img_gray, new_stereo_frame.left_binary, img_range);
            }
            else
            {
                mqCandidateFramePairs.back().left = img_gray;

                // 链接前后pair
                Stereoframe *prev_ptr = mqCandidateFramePairs.back().prev;
                prev_ptr->next = &mqCandidateFramePairs.back();
            }

            mnLeftCount++;
        }

        else // camera == RIGHT_IMAGE
        {
            if (!mqCandidateFramePairs.back().right.empty())
            {
                // 若这帧图像属于新的pair

                Stereoframe new_stereo_frame;
                new_stereo_frame.count = cnt;
                new_stereo_frame.prev = &mqCandidateFramePairs.back();
                new_stereo_frame.right = img_gray;

                mqCandidateFramePairs.push(new_stereo_frame);
                mqCandidateFramePairs.back().next = nullptr;

                pair<int, int> img_range(0, msCamGroupConfig.height);
                mpRightMatcher->BinaryDescriptor(img_gray, new_stereo_frame.right_binary, img_range);
            }
            else
            {
                mqCandidateFramePairs.back().right = img_gray;
                // 链接前后pair
                Stereoframe *prev_ptr = mqCandidateFramePairs.back().prev;
                prev_ptr->next = &mqCandidateFramePairs.back();
            }

            mnRightCount++;
        }
    }
}

bool Group::NextLeft()
{
    if (mbLeftInit)
        mLeftRowPoses[0].push_back(Eigen::Matrix4d::Zero());

    if (mCurrentRowPair.left_row == msCamGroupConfig.height - 1) // 帧结束
    {
        // 进入下一帧图像
        mCurrentRowPair.left_row = 0;
        mCurrentRowPair.left_cnt++;
        mCurrentRowPair.left_frame_cnt++;

        // 阻塞，直至下组pair到达
        while (mCurrentRowPair.left_stereo_ptr->next == nullptr)
            ;

        // 删除图像和上上帧pair，释放空间
        if (mLastRowPair.left_stereo_ptr != nullptr) // 若不是第一帧
        {
            mLastRowPair.left_stereo_ptr->left.release();
            if (mLastRowPair.left_stereo_ptr->right.empty())
                mqCandidateFramePairs.pop();
        }
        // 匹配结果清空
        vector<Result> tmp1, tmp2;
        mvLeftMatchedRes.swap(tmp1);
        mvRightMatchedRes.swap(tmp2);

        // 进入下一行
        mLastRowPair.left_stereo_ptr = mCurrentRowPair.left_stereo_ptr;
        mCurrentRowPair.left_stereo_ptr = mCurrentRowPair.left_stereo_ptr->next;

        // 初始化阶段
        if (mbLeftInit)
        {
            vector<Eigen::Matrix4d> tmp;
            mLeftRowPoses[0].swap(tmp);
        }

        return true;
    }
    else
    {
        mCurrentRowPair.left_row++;
        mCurrentRowPair.left_cnt++;
        return false;
    }
}

bool Group::NextRight()
{
    if (mbRightInit)
        mRightRowPoses[0].push_back(Eigen::Matrix4d::Zero());

    if (mCurrentRowPair.right_row == msCamGroupConfig.height - 1)
    {
        // 进入下一帧图像
        mCurrentRowPair.right_row = 0;
        mCurrentRowPair.right_cnt++;
        mCurrentRowPair.right_frame_cnt++;

        // 阻塞，直至下组pair到达
        while (mCurrentRowPair.right_stereo_ptr->next == nullptr)
            ;

        // 删除图像和上上帧pair，释放空间
        if (mLastRowPair.right_stereo_ptr != nullptr)
        {
            mLastRowPair.right_stereo_ptr->right.release();
            if (mLastRowPair.right_stereo_ptr->right.empty())
                mqCandidateFramePairs.pop();
        }
        // 匹配结果清空
        vector<Result> tmp1, tmp2;
        mvLeftMatchedRes.swap(tmp1);
        mvRightMatchedRes.swap(tmp2);

        // 进入下一行
        mLastRowPair.right_stereo_ptr = mCurrentRowPair.right_stereo_ptr;
        mCurrentRowPair.right_stereo_ptr = mCurrentRowPair.right_stereo_ptr->next;

        if (mbRightInit)
        {
            vector<Eigen::Matrix4d> tmp;
            mRightRowPoses[0].swap(tmp);
        }

        return true;
    }
    else
    {
        mCurrentRowPair.right_row++;
        mCurrentRowPair.right_cnt++;
        return false;
    }
}

void Group::next()
{
    bool bDraw = false; // 新的一帧就画图
    // 初始化阶段
    if (mState == NOT_INITIALIZED)
    {
        // 若相机已经检测到同步信号，就不再更新行，等待另外的相机同步
        if (mbLeftInit && NextLeft())
            bDraw = true;
        if (mbRightInit && NextRight())
            bDraw = true;

        // 同步的行不会相差过远
        int distance = abs(mCurrentRowPair.left_cnt - mCurrentRowPair.right_cnt);
        if (distance > msCamGroupConfig.height / 2) // 超过半张照片还没检测到就别再继续了
        {
            cerr << "ERROR: Initializing: Failed to detect sync signal, overtime." << endl;
            exit(0);
        }
    }
    // Tracking阶段
    else
    {
        if (NextLeft() || NextRight())
            bDraw = true;
    }

    if (bDraw)
    {
        cv::Mat LeftDrawPad(msCamGroupConfig.height, 640 * 2, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat RightDrawPad(msCamGroupConfig.height, 640 * 2, CV_8UC3, cv::Scalar(0, 0, 0));
        // cv::Mat LeftDrawPad = cv::Mat::zeros(msCamGroupConfig.height, 640*2, CV_8UC3);
        // cv::Mat RightDrawPad = cv::Mat::zeros(msCamGroupConfig.height, 640*2, CV_8UC3);
        DrawFrame(LeftDrawPad, RightDrawPad);
        SetDisplay(LeftDrawPad, "leftRes");
        SetDisplay(RightDrawPad, "rightRes");
        mpFrameDrawer->Update(this);
        ClearDisplay();
    }
}

void Group::DrawFrame(cv::Mat &leftDrawPad, cv::Mat &rightDrawPad)
{
    // 别名
    cv::Mat leftImg = mCurrentRowPair.left_stereo_ptr->left;
    cv::Mat rightImg = mCurrentRowPair.right_stereo_ptr->right;
    cv::Mat lastLeftImg = mLastRowPair.left_stereo_ptr->left;
    cv::Mat lastRightImg = mLastRowPair.right_stereo_ptr->right;

    cv::Mat leftim, rightim, leftlastim, rightlastim; // 原始图像

    vector<Result> vLeftMatchedRes; // 左相机当前帧的跟踪点匹配关系：当前帧关键点 上一帧匹配点
    vector<Result> vRightMatchedRes;
    pair<int, int> initMatchesPair; // 存储左右相机初始化结果的行索引

    // Copy variables within scoped mutex 复制作用域互斥对象内的变量
    //  step 1：将成员变量赋值给局部变量（包括图像、状态、其它的提示）
    if (mState == SYSTEM_NOT_READY)
        mState = NO_IMAGES_YET;

    // NOTICE 这里使用copyTo进行深拷贝是因为后面会把单通道灰度图像转为3通道图像
    leftImg.copyTo(leftim);
    rightImg.copyTo(rightim);
    lastLeftImg.copyTo(leftlastim);
    lastRightImg.copyTo(rightlastim);

    // 相机初始化还没完成的时候
    if (mState == NOT_INITIALIZED)
    {
        // 获取左右相机当前帧图像，利用LED灯进行初始化，图像会产生边缘效应，将初始化返回的行用红线显示
        vLeftMatchedRes = mvLeftMatchedRes;
        vRightMatchedRes = mvRightMatchedRes;
        initMatchesPair = make_pair(mnInitLeftCnt, mnInitRightCnt);
    }
    else if (mState == OK)
    {
        // 当系统处于运动追踪状态时
        vLeftMatchedRes = mvLeftMatchedRes;
        vRightMatchedRes = mvRightMatchedRes;
    }

    if (leftim.channels() < 3) // this should be always true
        cvtColor(leftim, leftim, CV_GRAY2BGR);
    if (rightim.channels() < 3) // this should be always true
        cvtColor(rightim, rightim, CV_GRAY2BGR);
    if (leftlastim.channels() < 3) // this should be always true
        cvtColor(leftlastim, leftlastim, CV_GRAY2BGR);
    if (rightlastim.channels() < 3) // this should be always true
        cvtColor(rightlastim, rightlastim, CV_GRAY2BGR);

    // Draw
    //  step 2：绘制初始化轨迹连线，绘制特征点边框（特征点用小框圈住）
    //  step 2.1：初始化时，当前帧的特征坐标与初始帧的特征点坐标连成线，形成轨迹
    if (mState == NOT_INITIALIZED) // INITIALIZING
    {
        // 初始化阶段画布显示当前组相机的左右相机当前帧图像，并将发生边缘效应的行标红
        // 将左右图像中产生边缘效应的行标红
        int leftRowIndex = initMatchesPair.first;
        cv::rectangle(leftImg, cv::Point(0, leftRowIndex), cv::Point(leftImg.cols - 1, leftRowIndex), cv::Scalar(0, 0, 255), -1);
        int rightRowIndex = initMatchesPair.second;
        cv::rectangle(rightImg, cv::Point(0, rightRowIndex), cv::Point(rightImg.cols - 1, rightRowIndex), cv::Scalar(0, 0, 255), -1);
        // 将左右相机当前帧拼接到一张画布上
        leftim.copyTo(leftDrawPad(cv::Rect(0, 0, leftim.cols, leftim.rows)));
        rightim.copyTo(leftDrawPad(cv::Rect(leftim.cols, 0, rightim.cols, rightim.rows)));
    }
    else if (mState == OK) // TRACKING
    {
        // Draw keypoints
        const float r = 5;
        const int n = vLeftMatchedRes.size();
        for (unsigned int j = 0; j < n; j++)
        {
            // 绘制当前帧特征点到上一帧匹配特征点的连线
            if (vLeftMatchedRes[j].flag == true)
            {
                // 在当前帧特征点附近正方形选择四个点
                cv::Point2f pt1, pt2;
                pt1.x = vLeftMatchedRes[j].curr_px_coord.x - r;
                pt1.y = vLeftMatchedRes[j].curr_px_coord.y - r;
                pt2.x = vLeftMatchedRes[j].curr_px_coord.x + r;
                pt2.y = vLeftMatchedRes[j].curr_px_coord.y + r;
                // 通道顺序为bgr，地图中MapPoints用绿色圆点表示，并用绿色小方框圈住
                cv::rectangle(leftim, pt1, pt2, cv::Scalar(0, 255, 0));
                cv::circle(leftim, vLeftMatchedRes[j].curr_px_coord, 2, cv::Scalar(0, 255, 0), -1);
                // 在上一帧匹配特征点附近正方形选择四个点
                cv::Point2f pt3, pt4;
                pt3.x = vLeftMatchedRes[j].matched_px_coord.x - r;
                pt3.y = vLeftMatchedRes[j].matched_px_coord.y - r;
                pt4.x = vLeftMatchedRes[j].matched_px_coord.x + r;
                pt4.y = vLeftMatchedRes[j].matched_px_coord.y + r;
                // 通道顺序为bgr，地图中MapPoints用蓝色圆点表示，并用蓝色小方框圈住
                cv::rectangle(leftlastim, pt3, pt4, cv::Scalar(255, 0, 0));
                cv::circle(leftlastim, vLeftMatchedRes[j].matched_px_coord, 2, cv::Scalar(255, 0, 0), -1);

                // 将前后两帧图像拼接在一张画布上
                leftlastim.copyTo(leftDrawPad(cv::Rect(0, 0, leftlastim.cols, leftlastim.rows)));
                leftim.copyTo(leftDrawPad(cv::Rect(leftlastim.cols, 0, leftim.cols, leftim.rows)));
                // 将当前帧跟踪点与上一帧匹配点进行连线
                cv::Point leftMatchedPt(vLeftMatchedRes[j].matched_px_coord.x, vLeftMatchedRes[j].matched_px_coord.y);
                cv::Point leftTrackPt(leftlastim.cols + vLeftMatchedRes[j].curr_px_coord.x, vLeftMatchedRes[j].curr_px_coord.y);
                cv::line(leftDrawPad, leftMatchedPt, leftTrackPt, cv::Scalar(0, 255, 0));
            }

            if (vRightMatchedRes[j].flag == true)
            {
                // 在当前帧特征点附近正方形选择四个点
                cv::Point2f pt5, pt6;
                pt5.x = vRightMatchedRes[j].curr_px_coord.x - r;
                pt5.y = vRightMatchedRes[j].curr_px_coord.y - r;
                pt6.x = vRightMatchedRes[j].curr_px_coord.x + r;
                pt6.y = vRightMatchedRes[j].curr_px_coord.y + r;
                // 通道顺序为bgr，地图中MapPoints用绿色圆点表示，并用绿色小方框圈住
                cv::rectangle(rightim, pt5, pt6, cv::Scalar(0, 255, 0));
                cv::circle(rightim, vRightMatchedRes[j].curr_px_coord, 2, cv::Scalar(0, 255, 0), -1);
                // 在上一帧匹配特征点附近正方形选择四个点
                cv::Point2f pt7, pt8;
                pt7.x = vRightMatchedRes[j].matched_px_coord.x - r;
                pt7.y = vRightMatchedRes[j].matched_px_coord.y - r;
                pt8.x = vRightMatchedRes[j].matched_px_coord.x + r;
                pt8.y = vRightMatchedRes[j].matched_px_coord.y + r;
                // 通道顺序为bgr，地图中MapPoints用蓝色圆点表示，并用蓝色小方框圈住
                cv::rectangle(rightlastim, pt7, pt8, cv::Scalar(255, 0, 0));
                cv::circle(rightlastim, vRightMatchedRes[j].matched_px_coord, 2, cv::Scalar(255, 0, 0), -1);

                // 将前后两帧图像拼接在一张画布上
                rightlastim.copyTo(rightDrawPad(cv::Rect(0, 0, rightlastim.cols, rightlastim.rows)));
                rightim.copyTo(rightDrawPad(cv::Rect(rightlastim.cols, 0, rightim.cols, rightim.rows)));
                // 将当前帧跟踪点与上一帧匹配点进行连线
                cv::Point rightMatchedPt(vRightMatchedRes[j].matched_px_coord.x, vRightMatchedRes[j].matched_px_coord.y);
                cv::Point rightTrackPt(rightlastim.cols + vRightMatchedRes[j].curr_px_coord.x, vRightMatchedRes[j].curr_px_coord.y);
                cv::line(rightDrawPad, rightMatchedPt, rightTrackPt, cv::Scalar(0, 255, 0));
            }
        }
        // 遍历所有的特征点
    }
}

void Group::Update(Eigen::Matrix4d &pose)
{
    Eigen::Matrix4d T_1_cl, T_2_cl, T_cl_1, T_cl_2;
    cv::cv2eigen(msCamGroupConfig.T_1_cl, T_1_cl);
    cv::cv2eigen(msCamGroupConfig.T_2_cl, T_2_cl);
    // cv::cv2eigen(msCamGroupConfig.T_cl_1, T_cl_1);
    // cv::cv2eigen(msCamGroupConfig.T_cl_2, T_cl_2);

    mLeftRowPoses[mCurrentRowPair.left_frame_cnt].push_back(pose * T_1_cl);
    mRightRowPoses[mCurrentRowPair.right_frame_cnt].push_back(pose * T_2_cl);

    mpLeftMatcher->mpHomographyMap->Update(mLeftRowPoses[mCurrentRowPair.left_frame_cnt][mCurrentRowPair.left_row], mCurrentRowPair.left_row);
    mpLeftMatcher->mpHomographyMap->UpdateRref(mLeftRowPoses[mCurrentRowPair.left_frame_cnt][mCurrentRowPair.left_row]);
    mpRightMatcher->mpHomographyMap->Update(mRightRowPoses[mCurrentRowPair.right_frame_cnt][mCurrentRowPair.right_row], mCurrentRowPair.right_row);
    mpRightMatcher->mpHomographyMap->UpdateRref(mRightRowPoses[mCurrentRowPair.right_frame_cnt][mCurrentRowPair.right_row]);
}

void Group::SetDisplay(cv::Mat &pFrame, string name)
{
    mvDispFrames.push_back(pFrame);
    mvsDispNames.push_back(name);
}

void Group::ClearDisplay()
{
    vector<cv::Mat> tmp1;
    vector<string> tmp2;
    mvDispFrames.swap(tmp1);
    mvsDispNames.swap(tmp2);
}