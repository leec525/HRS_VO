/**
 * @file HRSmatcher.cpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 处理数据关联问题
 * @version 0.1
 * @date 2023-05-06
 *  
 */

#include <HRSmatcher.hpp>
#include "HomographyMap.hpp"

#include <limits.h>

#include <Eigen/Dense>
#include <iostream>
//#include <algorithm>
//#include <queue>
#include <vector>
#include <cmath>
#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
//#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include<stdint.h>

using namespace std;
using namespace HRS_VO;

namespace HRS_VO
{
    //要用到的一些阈值
    const int HRSmatcher::BLOCK_SIZE = 20;
    const int HRSmatcher::BINARY_THRESHOLD = 0;

    HRSmatcher::HRSmatcher(cv::Mat K, cv::Mat D):mK(K), mD(D)
    {
        mpHomographyMap = new HomographyMap(K);
    }

    pair<int,int> HRSmatcher::HomographyTrans(const cv::Mat& last_img, const int & src_row, cv::Mat& warped_img, cv::Mat& index_img)
    {
        //纯旋转运动下的单应矩阵
        int img_width = last_img.cols;
        int img_height = last_img.rows;
        warped_img = cv::Mat::zeros(img_height, img_width, last_img.type());
        index_img = cv::Mat::zeros(img_height, img_width, last_img.type());
        int min_warped_row = img_height;
        int max_warped_row = 0;

        int candidate_row_start = max(0, src_row - ROWRANGE);
        int candidate_row_end = min(src_row + ROWRANGE, last_img.rows-1);
        for(int r = candidate_row_start; r <= candidate_row_end; r++ )
        {
            cv::Mat Hr = mpHomographyMap->GetHomography(r); 

            // 逐点进行单应变换
            cv::Mat invH = Hr.inv();    // Compute inverse of H to transform coordinates from img2 to img1
            for (int i = 0; i < img_width; i++)
            {
                cv::Mat coord(3, 1, CV_64F);  //要进行单应变换的像素在归一化坐标系中的坐标
                coord.at<double>(0, 0) = (double)i;
                coord.at<double>(1, 0) = (double)r;
                coord.at<double>(2, 0) = 1;

                cv::Mat warped_coord = invH * coord;
                warped_coord /= warped_coord.at<double>(2, 0);

                int warped_row = warped_coord.at<double>(1, 0);
                int warped_col = warped_coord.at<double>(0, 0);

                //检查变换后的行号是否在图像的范围内，如果不在就忽略
                if (warped_row >= 0 && warped_row < img_height)
                {
                    //使用双线性插值法计算出该像素在原图像中的灰度值
                    //因为在变换前后，像素的x坐标不一定对应着整数列号
                    double x = warped_coord.at<double>(0, 0);
                    //使用std::floor()函数向下取整得到该像素在原图像中的最左像素的列号left
                    int left = std::max((int)std::floor(x), 0);
                    //使用std::ceil()函数向上取整得到该像素在原图像中的最右像素的列号right
                    int right = std::min((int)std::ceil(x), img_width - 1);

                    //将变换前后像素在x方向上距离最近的两个点的灰度值按照它们在x方向上的距离比例进行加权平均
                    double alpha = x - left;

                    warped_img.at<uchar>(warped_row, warped_col) = (1 - alpha) * last_img.at<uchar>(r, left) +
                        alpha * last_img.at<uchar>(r, right);
                    index_img.at<uchar>(warped_row, warped_col) = r;

                    // 这样写很笨，因为单应矩阵是线性的，中间大量点是无需判断的
                    if(warped_row < min_warped_row)
                        min_warped_row = warped_row;
                    else if(warped_row > max_warped_row)
                        max_warped_row = warped_row;
                }
            }
        }
        
        return make_pair(min_warped_row, max_warped_row);
    } 

    int HRSmatcher::DescriptorDistance(const cv::Mat& binary_desc1, const cv::Mat& binary_desc2)
    {
        int dist=0;

        int cols = binary_desc1.cols;
        for(int i = 0; i< cols; i++)
        {
            dist += __builtin_popcount(binary_desc1.at<uchar>(0, i) ^ binary_desc2.at<uchar>(0, i));
        }

        return dist;
    }

    void HRSmatcher::BinaryDescriptor(const cv::Mat& src_img, cv::Mat& bin_desc, const pair<int,int>& range)
    {
        //生成的二进制描述符的位数 640/20=32
        //BLOCK_SIZE图像行分成的块大小，20
        int bin_cols = static_cast<int>(round(src_img.cols / BLOCK_SIZE));

        bin_desc = cv::Mat::zeros(src_img.rows, bin_cols, CV_8UC1); // 初始化二进制描述符

        for ( int r = range.first; r < range.second; r++ )
        {
            cv::Mat M = src_img.row(r);
            //M = src_img.rowRange(r,r+1); //矩阵存储原始图片每一行样本

            //定义高斯二阶导数的卷积核
            cv::Mat kernel = cv::Mat::zeros(1, 3, CV_32F);
            kernel.at<float>(0, 0) = 1;
            kernel.at<float>(0, 1) = -2;
            kernel.at<float>(0, 2) = 1;
            //对图像进行高斯二阶导数卷积
            cv::Mat smoothed;
            cv::filter2D(M, smoothed, -1, kernel);

            //将边缘响应小于零的值设为0
            //threshold(smoothed, smoothed, 0, 255, THRESH_TOZERO);

            //计算每一行的二进制描述符
            // cv::GaussianBlur(M, conv_gray, cv::Size(3, 1), 0.8, 0, cv::BORDER_DEFAULT); //1*3的高斯卷积核
            // //拉普拉斯变换 自编函数
            // cv::Mat dst_gray, abs_dst_gray;
            // myLaplacianfilter(conv_gray, dst_gray); //一维
            // convertScaleAbs(dst_gray, abs_dst_gray);

            //根据阈值生成行二进制描述符
            //TODO:重新定义图像行的二进制描述符
            cv::MatIterator_<uchar> it = smoothed.row(0).begin<uchar>(), it_end  = smoothed.row(0).end<uchar>();
            for(int block_col = 0; block_col < bin_cols; block_col++)
            {
                for(int c = 0; c < BLOCK_SIZE; it++,c++)
                {
                    if(*it > BINARY_THRESHOLD)
                    {
                        bin_desc.at<uchar>(r, block_col) = 1;

                        break;
                    }
                }
            }
        }
        
    }

    //自定义滤波函数
    // void HRSmatcher::myLaplacianfilter(cv::Mat& image_input, cv::Mat& image_output)
    // {
    //     image_output = image_input.clone();
    //     int la;
    //     for (int i = 0; i < image_input.rows; i++)
    //     {
    //         for (int j = 0; j < image_input.cols; j++)
    //         {
    //             la = -2 * image_input.at<uchar>(i, j) + image_input.at<uchar>(i, j + 1) + image_input.at<uchar>(i, j - 1);

    //             image_output.at<uchar>(i, j) = cv::saturate_cast<uchar>(image_output.at<uchar>(i, j) + la);

    //         }
    //     }
    // }

    pair<int, int> HRSmatcher::RowMatching(const cv::Mat& src_row_binary, const cv::Mat& candidate_binary, const pair<int,int>& warped_range)
    {
        //计算当前图像行和重构行之间的二进制描述符的汉明距离并寻找最小值
        int dist, min, dst_row;
        dst_row = warped_range.first;
        min = DescriptorDistance(src_row_binary, candidate_binary.row(dst_row));
        
        for (int r2 = warped_range.first+1; r2 <= warped_range.second; r2++)
        {
            dist = DescriptorDistance(src_row_binary, candidate_binary.row(r2));
            if(dist < min)
            {
                min = dist;
                dst_row = r2;
            }
        }
        int matched_row = dst_row;
        return make_pair(matched_row, min);
    }


    // 检测一个点是否在图像边框内
    inline bool HRSmatcher::inside(const Eigen::Vector2d& px) 
    {
        return px(0, 0) >= BOARDER && px(1, 0) >= BOARDER
            && px(0, 0) + BOARDER < IMAGE_WIDTH && px(1, 0) + BOARDER <= IMAGE_HEIGHT;
    }

   tuple<Eigen::Vector2d, double, int> HRSmatcher::PointMatching(const cv::Mat& src_row, const cv::Mat& dst_row, 
                                    const int & matched_row, const int min_range, const int max_range,
                                    const Eigen::Vector2d& px_src, const cv::Mat& index_img)
    {
        //寻找最好匹配点,计算匹配得分沿距离分布曲线的置信度量峰值比（PKR）
        double best_score = INFINITY;
        Eigen::Vector2d px, best_match;
        vector<pair<double, double>> distances;
        //限制范围，前后帧[-20,20],左右图像[10,70]
        for (int x = min_range; x <= max_range; x++) 
        {
            px[0] = px_src[0] + x;  //匹配点的像素坐标u
            px[1] = matched_row;  //匹配点所在的行索引即为像素坐标v, TODO: 行匹配也不一定准确,可以继续优化

            if(!inside(px)) continue;

            double dist = distance(px_src, px);
            double score = SAD(src_row, dst_row, px_src, px);
            distances.push_back({dist, score});
            
            if (score < best_score) 
            {
                best_score = score;
                best_match = px;
            }
        }
        double pkr_score = PKR(distances);
        int index = index_img.at<uchar>(best_match[1], best_match[0]);  //匹配点的原始行的行索引
        return make_tuple(best_match, pkr_score, index);

    }


    double HRSmatcher::NCC(const cv::Mat& src_row, const cv::Mat& dst_row, const Eigen::Vector2d& px_src, const Eigen::Vector2d& px_dst)
    {
        const int ncc_window_size = 3;  //NCC取的窗口半宽度
        const int ncc_area = (2 * ncc_window_size + 1) * 1;   //NCC窗口面积
        
        // 零均值-归一化互相关
        // 先算均值
        double mean_src_row = 0, mean_dst_row = 0;
        vector<double> values_src_row, values_dst_row;  //当前行和匹配行的均值
        for(int x = -ncc_window_size; x <= ncc_window_size; x++)
        {
            double value_src_row = double(src_row.ptr<uchar>(int(px_src(1,0)))[int(x + px_src(0,0))]) / 255.0;
            mean_src_row += value_src_row;

            double value_dst_row = double(dst_row.ptr<uchar>(int(px_dst(1,0)))[int(x + px_dst(0,0))]) / 255.0;
            mean_dst_row += value_dst_row;

            values_src_row.push_back(value_src_row);
            values_dst_row.push_back(value_dst_row);
        }

        mean_src_row /= ncc_area;
        mean_dst_row /= ncc_area;

        //计算Zero mean NCC
        double numerator = 0, demoniator1 = 0, demoniator2 = 0;
        for(int i = 0; i < values_src_row.size(); i++)
        {
            double n = (values_src_row[i] - mean_src_row) * (values_dst_row[i] - mean_dst_row);
            numerator += n;
            demoniator1 += (values_src_row[i] - mean_src_row) * (values_src_row[i] - mean_src_row);
            demoniator2 += (values_dst_row[i] - mean_dst_row) * (values_dst_row[i] - mean_dst_row);
        }
        return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   //防止分母出现零
        
    }

    // 计算两个像素点之间的距离
    // TODO: 1. 在双目的情况下用双目视差，2. 在前后帧的情况下用三角化(opencv)
    // 暂时用欧式距离也不影响，但是到深度滤波器阶段还是需要修改的
    double HRSmatcher::distance(const Eigen::Vector2d& px_src, const Eigen::Vector2d& px_dst) 
    {
        return sqrt(pow(px_src[0] - px_dst[0], 2) + pow(px_src[1] - px_dst[1], 2));
    }

    // //从单应矩阵恢复旋转矩阵
    // void recoverRotationFromHomography(const cv::Mat& H, cv::Mat& R)
    // {
    //     cv::Mat U, S, Vt;
    //     cv::SVDecomp(H, S, U, Vt);

    //     //纯旋转情况下，奇异值矩阵中只有一个非零元素
    //     cv::Mat Sigma = cv::Mat::zeros(3, 3, CV_64F);
    //     Sigma.at<double>(0,0) = 1.0;
    //     Sigma.at<double>(1,1) = 1.0;
    //     Sigma.at<double>(2,2) = 0.0;
    //     cv::Mat Rtmp = U * Sigma * Vt;
    //     R = Rtmp.clone();
    // }

    double HRSmatcher::SAD(const cv::Mat& src_row, const cv::Mat& dst_row, const Eigen::Vector2d& px_src, const Eigen::Vector2d& px_dst)
    {
        const int window_size = 5;  //SAD取的窗口半宽度
        const int window_area = (2 * window_size + 1) * 1;   //SAD窗口面积

        // 零均值-归一化互相关
        // 先算均值
        double mean_src_row = 0.0, mean_dst_row = 0.0;
        vector<double> values_src_row, values_dst_row;  //当前行和匹配行的均值
        for (int x = -window_size; x <= window_size; x++)
        {
            double value_src_row = double(src_row.ptr<uchar>(int(px_src(1,0)))[int(x + px_src(0,0))]) / 255.0;  
            mean_src_row += value_src_row;

            double value_dst_row = double(dst_row.ptr<uchar>(int(px_dst(1,0)))[int(x + px_dst(0,0))]) / 255.0;
            mean_dst_row += value_dst_row;

            values_src_row.push_back(value_src_row);
            values_dst_row.push_back(value_dst_row);
        }
        mean_src_row /= window_area;
        mean_dst_row /= window_area;

        //计算Zero mean SAD
        double score = 0.0;
        for(int i = 0; i < values_src_row.size(); i++)
        {
            double n = (values_src_row[i] - mean_src_row) - (values_dst_row[i] - mean_dst_row);
            score += abs(n);  //取绝对值
        }
        return score;  //得分越接近于0,匹配越好
    }

    // 计算距离分布曲线的置信度量峰值比（PKR）
    double HRSmatcher::PKR(const vector<pair<double, double>>& distances) 
    {
        double max_peak = INFINITY;
        double second_peak = INFINITY;
        for (size_t i = 1; i < distances.size() - 1; i++) 
        {
            if (distances[i].second < distances[i-1].second && distances[i].second < distances[i+1].second) 
            {
                if (distances[i].second < max_peak) 
                {
                    second_peak = max_peak;
                    max_peak = distances[i].second;
                } 
                else if (distances[i].second < second_peak) 
                {
                    second_peak = distances[i].second;
                }
            }
        }
        return second_peak / max_peak;
    }

}