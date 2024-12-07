/**
 * @file Initializer.hpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 相机初始化
 * @version 0.1
 * @date 2023-04-14
 *  
 */

#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
//#include "Frame.h"

namespace HRS_VO
{

class Initializer
{
public:
    /**************************************************************
     * @brief  构造
     *************************************************************/
    Initializer();

    /**************************************************************
     * @brief  利用外部灯光进行相机行同步
     *************************************************************/
    void Initializer();

private:


}

} //namespace HRS_VO

#endif