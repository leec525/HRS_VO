/**
 * @file DepthFilter.hpp
 * @author lichen (xiamin_1997@163.com)
 * @brief 深度滤波
 * @version 0.1
 * @date 2023-06-01
 *  
 */

#ifndef HRS_VO_DEPTH_FILTER_H_
#define HRS_VO_DEPTH_FILTER_H_

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <vikit/performance_monitor.h>
#include "Group.hpp"
#include "HRSmatcher.hpp"
#include "FrameDrawer.hpp"

namespace HRS_VO 
{

class Group;
class Frame;

/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int batch_counter;
  static int seed_counter;
  int batch_id;                // Batch id is the id of the keyframe for which the seed was created.Batch id是为其创建种子的关键帧的id
  int id;                      // Seed ID, only used for visualization.种子ID，仅用于可视化
  Group* grp;                  // Feature in the keyframe for which the depth should be computed.应为其计算深度的关键帧中的特征
                               // TODO:-->在卷帘相机跟踪中我们取特征为每行跟踪的几个点
  float a;                     // a of Beta distribution: When high, probability of inlier is large.β分布的a：当较高时，内围的概率较大
  float b;                     // b of Beta distribution: When high, probability of outlier is large.β分布的b：当高时，异常值的概率大
  float mu;                    // Mean of normal distribution.正态分布均值
  float z_range;               // Max range of the possible depth.
  float sigma2;                // Variance of normal distribution.正态分布方差
  Eigen::Matrix2d patch_cov;   // Patch covariance in reference image.参考图像中的斑块协方差
  Seed(Group* grp, float depth_mean, float depth_min);
};

/// Depth filter implements the Bayesian Update proposed in:
/// "Video-based, Real-Time Multi View Stereo" by G. Vogiatzis and C. Hernández.
/// In Image and Vision Computing, 29(7):434-441, 2011.
///
/// The class uses a callback mechanism such that it can be used also by other
/// algorithms than nslam and for simplified testing.
class DepthFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef boost::unique_lock<boost::mutex> lock_t;
  typedef boost::function<void ( Point*, double )> callback_t;

  /// Depth-filter config parameters
  struct Options
  {
    bool check_ftr_angle;                       // gradient features are only updated if the epipolar line is orthogonal to the gradient.
    bool epi_search_1d;                         // restrict Gauss Newton in the epipolar search to the epipolar line.
    bool verbose;                               // display output.显示输出
    bool use_photometric_disparity_error;       // use photometric disparity error instead of 1px error in tau computation.使用光度误差
    int max_n_kfs;                              // maximum number of keyframes for which we maintain seeds.
    double sigma_i_sq;                          // image noise.
    double seed_convergence_sigma2_thresh;      // threshold on depth uncertainty for convergence.
    Options()
    : check_ftr_angle(false),
      epi_search_1d(false),
      verbose(false),
      use_photometric_disparity_error(false),
      max_n_kfs(3),
      sigma_i_sq(5e-4),
      seed_convergence_sigma2_thresh(200.0)
    {}
  } options_;

  DepthFilter(
      feature_detection::DetectorPtr feature_detector,
      callback_t seed_converged_cb);

  virtual ~DepthFilter();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  /// Add frame to the queue to be processed.
  void addFrame(FramePtr frame);

  /// Add new keyframe to the queue
  void addKeyframe(FramePtr frame, double depth_mean, double depth_min);

  /// Remove all seeds which are initialized from the specified keyframe. This
  /// function is used to make sure that no seeds points to a non-existent frame
  /// when a frame is removed from the map.
  void removeKeyframe(FramePtr frame);

  /// If the map is reset, call this function such that we don't have pointers
  /// to old frames.
  void reset();

  /// Returns a copy of the seeds belonging to frame. Thread-safe.
  /// Can be used to compute the Next-Best-View in parallel.
  /// IMPORTANT! Make sure you hold a valid reference counting pointer to frame
  /// so it is not being deleted while you use it.
  void getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds);

  /// Return a reference to the seeds. This is NOT THREAD SAFE!
  std::list<Seed>& getSeeds() { return seeds_; }

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updateSeed(
      const float x,
      const float tau2,
      Seed* seed);

  /// Compute the uncertainty of the measurement.
  static double computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle);

protected:
  feature_detection::DetectorPtr feature_detector_;
  callback_t seed_converged_cb_;
  std::list<Seed> seeds_;
  boost::mutex seeds_mut_;
  bool seeds_updating_halt_;            // Set this value to true when seeds updating should be interrupted.
  boost::thread* thread_;
  std::queue<FramePtr> frame_queue_;
  boost::mutex frame_queue_mut_;
  boost::condition_variable frame_queue_cond_;
  FramePtr new_keyframe_;               // Next keyframe to extract new seeds.
  bool new_keyframe_set_;               // Do we have a new keyframe to process?.
  double new_keyframe_min_depth_;       // Minimum depth in the new keyframe. Used for range in new seeds.
  double new_keyframe_mean_depth_;      // Maximum depth in the new keyframe. Used for range in new seeds.
  vk::PerformanceMonitor permon_;       // Separate performance monitor since the DepthFilter runs in a parallel thread.
  Matcher matcher_;

  /// Initialize new seeds from a frame.
  void initializeSeeds(FramePtr frame);

  /// Update all seeds with a new measurement frame.
  virtual void updateSeeds(FramePtr frame);

  /// When a new keyframe arrives, the frame queue should be cleared.
  void clearFrameQueue();

  /// A thread that is continuously updating the seeds.
  void updateSeedsLoop();
};

} // namespace svo

#endif // SVO_DEPTH_FILTER_H_
