#include <vector>
#include <iostream>
// 假设有一个Pose类来表示位姿，包含旋转（四元数）和平移（向量）
class Pose {
public:
    Quaternion rotation; // 旋转部分，四元数
    Vector3 translation; // 平移部分，三维向量
    // 构造函数、析构函数和其他成员函数...
};

bool isKeyframe(const Pose& lastKeyframePose, const Pose& currentPose, float rotationThreshold, float translationThreshold) {
    // 计算旋转差异
    float rotDiff = (lastKeyframePose.rotation.inverse() * currentPose.rotation).norm();
    // 计算平移差异
    float transDiff = (lastKeyframePose.translation - currentPose.translation).norm();

    // 如果旋转或平移超过阈值，则视为关键帧
    return rotDiff > rotationThreshold || transDiff > translationThreshold;
}

int main() {
    // 示例：加载位姿信息、设置阈值、循环检查每一帧是否为关键帧
    // 假设poses是已加载的所有位姿信息
    std::vector<Pose> poses; 
    float rotationThreshold = 0.1; // 旋转阈值，根据实际情况调整
    float translationThreshold = 0.1; // 平移阈值，根据实际情况调整

    std::vector<Pose> keyframes;
    for (size_t i = 1; i < poses.size(); ++i) {
        if (isKeyframe(keyframes.back(), poses[i], rotationThreshold, translationThreshold)) {
            keyframes.push_back(poses[i]);
            std::cout << "Frame " << i << " is selected as a keyframe." << std::endl;
        }
    }

    return 0;
}
