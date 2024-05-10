/**
 * @file HRSmatcher.hpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 处理数据关联问题
 * @version 0.1
 * @date 2023-03-29
 *  
 */

#ifndef HRSMATCHER_H
#define HRSMATCHER_H

#define BOARDER 20          // 图像的边缘宽度
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 480
#define ROWRANGE 10         // 行匹配的范围

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "HomographyMap.hpp"
// #include "KeyFrame.h"
// #include "Frame.h"

namespace HRS_VO
{

class HomographyMap;
class Group;

class HRSmatcher
{    
public:

    /** @brief 构造
     * @param K   相机内参矩阵
     * @param D   相机畸变参数矩阵
     */
    HRSmatcher(cv::Mat K, cv::Mat D);


    /** @brief  计算单应矩阵，对行图像进行单应变换
     * @param  last_img    前一帧图像
     * @param  src_row     当前图像当前行的行索引
     * @param  warped_img  进行单应变换后的行图像
     * @param  index_img   单应变换后坐标点的原始行
     * @return 单应变换后图像的范围
     */
    std::pair<int,int> HomographyTrans(const cv::Mat& last_img, const int & src_row, cv::Mat& warped_img, cv::Mat& index_img);


    /** @brief Computes the Hamming distance between two Binary descriptors 计算当前帧当前行和候选图像行的二进制描述符距离
     * @param[in] binary_desc1     一个二进制描述符
     * @param[in] binary_desc2     另外一个二进制描述
     * @return dist   描述子的汉明距离
     */
    int DescriptorDistance(const cv::Mat& binary_desc1, const cv::Mat& binary_desc2);


    /** @brief  赋二进制描述符
     * @param  src_img    输入图像
     * @param  bin_desc   二进制描述符
     * @param  range      要计算二进制描述符的行范围
     * @return 每行二进制描述符赋完后则返回
     */
    void BinaryDescriptor(const cv::Mat& src_img, cv::Mat& bin_desc, const std::pair<int,int>& range); 


    /** @brief 自定义拉普拉斯滤波函数
     * 一维拉普拉斯平滑滤波[1 -2 1]
     * @param[in] image_input     输入需要滤波的图像
     * @param[in] image_output    输出滤波后的图像
     */
    //void myLaplacianfilter(cv::Mat& image_input, cv::Mat& image_output); 


     /** @brief  行匹配
     * @param  src_row_binary     当前图像的行描述符
     * @param  candidate_binary   匹配图像的行描述符
     * @param  warped_range       单应变换图像的范围
     * @return 匹配行的行索引及其汉明距离（用二进制描述符进行匹配，则返回的是匹配的汉明距离）
     */
    std::pair<int, int> RowMatching(const cv::Mat& src_row_binary, const cv::Mat& candidate_binary, const std::pair<int,int>& warped_range);


    /** @brief  检测一个点是否在图像边框内
     * @param  px  匹配像素点
     * @return 是否在边框内
     */
    inline bool inside(const Eigen::Vector2d& px);


    /** @brief  计算匹配点
     * @param  src_row      当前帧图像行
     * @param  dst_row      当前帧匹配行
     * @param  matched_row  匹配行的行索引
     * @param  min_range    搜索范围下限
     * @param  max_range    搜索范围上限
     * @param  px_src       当前行像素点
     * @param  index_img    匹配点的原始行，用于指示匹配点的位姿矩阵
     * @return 匹配点坐标
     */
    std::tuple<Eigen::Vector2d, double, int> PointMatching(const cv::Mat& src_row, const cv::Mat& dst_row, 
                                    const int & matched_row, const int min_range, const int max_range,
                                    const Eigen::Vector2d& px_src, const cv::Mat& index_img);


    /** @brief  计算NCC评分
     * @param  src_row    当前帧图像行
     * @param  dst_row    当前帧匹配行
     * @param  px_src     当前行像素点
     * @param  px_dst     匹配像素点
     * @return NCC评分
     */
    double NCC(const cv::Mat& src_row, const cv::Mat& dst_row, const Eigen::Vector2d& px_src, const Eigen::Vector2d& px_dst);


    /** @brief  计算两个像素点之间的距离
     * @param  px_src    当前像素点
     * @param  px_dst    匹配像素点
     * @return 两个像素点的距离
     */
    double distance(const Eigen::Vector2d& px_src, const Eigen::Vector2d& px_dst);


    /** @brief  计算SAD匹配得分
     * @param  src_row    当前帧图像行
     * @param  dst_row    当前帧匹配行
     * @param  px_src     当前行像素点
     * @param  px_dst     匹配像素点
     * @return score  SAD匹配得分
     */
    double SAD(const cv::Mat &src_row, const cv::Mat &dst_row, const Eigen::Vector2d &px_src, const Eigen::Vector2d &px_dst);
    

    /** @brief  计算距离分布曲线的置信度量峰值比（PKR）
     * @param  distances   距离和匹配得分数组对
     * @return PFR
     */
    double PKR(const std::vector<std::pair<double, double>>& distances);


    /** @brief  析构
     */
    ~HRSmatcher();


public:

    // 要用到的一些阈值
    static const int BLOCK_SIZE;        ///< 二进制描述符位数
    static const int BINARY_THRESHOLD;  ///< 赋二进制描述符设的阈值
    cv::Mat mK, mD;
    
    HomographyMap* mpHomographyMap;

protected:
    
};

}// namespace HRS_VO

#endif // HRSMATCHER_H
