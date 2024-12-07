/**
 * @file HomographyMap.cpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 计算单应矩阵的查找表
 * @version 0.1
 * @date 2023-05-16
 *
 */

/**
为了创建F_GS，拆分H = k*R_ref*R_row.inv()*k.inv(),该扭曲通过创建两个查找表将每个滚动快门行映射到参考姿势:
    1) 查找表以使先前帧不失真并通过应用R_row.inv()*k.inv()调整其旋转
    2) K*R_ref的类似表。
有助于避免参考姿势T_ref改变时的冗余计算
自适应地选择参考姿态，使得参考和当前行之间的运动在预定义阈值内
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <math.h>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
// #include "funset.hpp"
// #include "common.hpp"
#include <Group.hpp>

#include <HomographyMap.hpp>

using namespace std;
using namespace Eigen;
using namespace HRS_VO;

namespace HRS_VO
{

    // Matrix3d Rref = Matrix3d::Identity();

    HomographyMap::HomographyMap(cv::Mat K, double R_threshold) : mK(K), mdRrefThreshold(R_threshold)
    {
        // mbStart = true;
        mbInit = true;
        mbNewFrame = false;
    }

    void HomographyMap::Update(const Eigen::Matrix4d &Pose, const int &row_index)
    {
        if (row_index == IMAGE_HEIGHT - 1)
        {
            mbNewFrame = true;
        }

        // 根据标志位判断是否要将currentmap变为formermap
        if (mbNewFrame)
        {
            // current->former  :swap
            mvFormerMap.swap(mvCurrentMap);
            // clear current    :swap(vector<> tmp)
            std::vector<cv::Mat> tmp;
            mvCurrentMap.swap(tmp);

            mbNewFrame = false;
        }

        // 更新currentmap
        cv::Mat T;
        cv::eigen2cv(Pose, T);
        cv::Mat R = T(cv::Rect(0, 0, 3, 3)).clone();
        cv::Mat KR = mK * R;
        cv::Mat KR_inv = KR.inv();
        mvCurrentMap.push_back(KR_inv);
    }

    void HomographyMap::UpdateRref(const Eigen::Matrix4d &Pose)
    {
        cv::Mat T;
        cv::eigen2cv(Pose, T);
        cv::Mat R = T(cv::Rect(0, 0, 3, 3)).clone();

        // 若还处于初始化期间，就直接将输入的R作为Rref
        if (mbInit)
        {
            mRref = R;
            mKRref = mK * mRref;
            mbInit = false;
        }
        else
        {
            // 判断参考行与当前行之间的旋转是否大于阈值
            // 2cos(rot_diff)=trace(R1.inv()*R2)-1
            double rot_diff;
            cv::Mat Rref_inv = mRref.inv();
            cv::Mat diff_r = Rref_inv * R;
            cv::Scalar tr = cv::trace(diff_r);
            rot_diff = acos((tr[0] - 1.0) / 2.0);
            // 用Eigen矩阵
            // double rot_diff = acos(((R * mRref.transpose()).trace()-1.0)/2.0);
            if (rot_diff > mdRrefThreshold)
            {
                mRref = R;
                mKRref = mK * mRref;
            }
        }
    }

    cv::Mat HomographyMap::GetHomography(int prev_row)
    {
        // 行索引为row_idx的行，利用Rref计算旋转矩阵
        return mKRref * mvFormerMap[prev_row];
    }

    // 析构函数
    HomographyMap::~HomographyMap()
    {
    }

}
