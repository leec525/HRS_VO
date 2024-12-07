/**
 * @file HomographyMap.hpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 计算单应矩阵的查找表
 * @version 0.1
 * @date 2023-04-18
 *  
 */

#ifndef HOMOGRAPHYMAP_H
#define HOMOGRAPHYMAP_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace HRS_VO
{

class HomographyMap
{
    
public:
    /** @brief   构造
     * @param   R_threshold 当前行i与参考行j之间的最大允许旋转
     */
    HomographyMap(cv::Mat K, double R_threshold=0.06);

    

    /** @brief    创建查找表1:R_row.inv()*k.inv()
     * @param Pose      当前帧当前行的位姿矩阵
     * @param row_index 当前行索引
     * @return   存储前一帧图像每一行的R_row.inv()*k.inv()
     */
    void Update(const Eigen::Matrix4d& Pose, const int & row_index);

    /** @brief    更新参考行j
     * @param R        当前行i的旋转矩阵
     * @param row_idx  当前行的行索引
     * @return   当当前行i和参考行j旋转差值超过阈值则更新j
     */
    void UpdateRref(const Eigen::Matrix4d& Pose);

    /** @brief    计算单应矩阵
     * @param row_idx  当前行的行索引
     * @return   前一帧到当前参考行的单应矩阵
     */
    cv::Mat GetHomography(int row_idx);

    /** @brief    析构
     */
    ~HomographyMap();

protected:
    bool mbInit; //系统正在初始化
    //bool mbStart;
    bool mbNewFrame;

    double mdRrefThreshold;
    cv::Mat mK,mRref, mKRref;
    std::vector<cv::Mat> mvFormerMap,mvCurrentMap;

};




} // namespace HRS_VO

#endif // HOMOGRAPHYMAP_H