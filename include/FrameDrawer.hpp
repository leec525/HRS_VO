#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H


#include<chrono>

#include<Python.h>
#include "numpy/arrayobject.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Group.hpp"

#include<mutex>

namespace HRS_VO
{

class Group;

class FrameDrawer
{
public:
    FrameDrawer(int group_size);


    /** @brief  将Node线程的数据拷贝到绘图线程 */
    void Update(Group *pGroup);


    /** @brief  发送frame给python*/
    void SendFrame(int node_id, cv::Mat frame, int type);


    /** @brief  释放空间 */
    bool Finish();


public:
    int mState;
    int mnNodesCount;

    PyObject* mpFunSendFrame;
    PyObject* mpFunQuit;

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
    bool Image2Numpy_UBYTE(cv::Mat& srcImage,
                            PyObject*& pPyArray);

    bool ImportNumPySupport(void) const
    {
        // 这是一个宏，其中包含了返回的语句
        import_array();
        return true;
    }

protected:

    std::mutex mMutex;
    std::mutex mPyMutex;

    int mnGroupSize; //相机组的个数

    //存储当前相机的帧数
    std::vector<int> mvnLeftFrameIds;
    std::vector<int> mvnRightFrameIds;
};

}// namespace

#endif