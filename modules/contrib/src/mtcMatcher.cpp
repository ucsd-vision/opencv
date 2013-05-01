#include "precomp.hpp"
#include <algorithm>
#include <vector>

#include <iostream>
#include <iomanip>

#include "boost/tuple/tuple.hpp"
#include "boost/optional/optional.hpp"
#include <boost/foreach.hpp>
#include <cmath>
#include <map>
#include <tuple>
#include "kiss_fft/kiss_fftndr.h"

#include "opencv2/contrib/mtc.hpp"
#include "opencv2/contrib/mtcSamplePattern.hpp"
#include "opencv2/contrib/mtcUtil.hpp"
#include "opencv2/contrib/mtcMatcher.hpp"


///////////////////////////////////////////////////////////

namespace cv {

using namespace std;
using boost::optional;

/**
 * Assuming the dot product is between two unit length vectors, find
 * their l2 distance.
 * Divides by sqrt(2) to undo a previous normalization.
 */
template<typename T>
T dotProductToL2Distance(const T dotProduct) {
  return sqrt(1 - dotProduct);
}

/**
 * Determine what the dot product would have been had the vectors been
 * normalized first.
 */
template<typename T>
T nccFromUnnormalized(const NormalizationData& leftData,
                           const NormalizationData& rightData,
                           const T unnormalizedInnerProduct) {
  CV_DbgAssert(leftData.size == rightData.size);

  // Suppose we observe the inner product between two vectors
  // (a_x * x + b_x) and (a_y * y + b_y), where x and y are normalized.
  // Note (a_x * x + b_x)^T (a_y * y + b_y) is
  // (a_x * x)^T (a_y * y) + a_y * b_x^T y + a_x * b_y^T x + b_x^T b_y.
  // Thus we can solve for the normalized dot product:
  // x^T y = ((a_x * x)^T (a_y * y) - a_y * b_x^T y - a_x * b_y^T x - b_x^T b_y) / (a_x * a_y).
  const T aybxy = rightData.affinePair.scale * leftData.affinePair.offset
      * rightData.elementSum;

  const T axbyx = leftData.affinePair.scale * rightData.affinePair.offset
      * leftData.elementSum;

  const T bxby = leftData.size * leftData.affinePair.offset
      * rightData.affinePair.offset;

  const T numerator = unnormalizedInnerProduct - aybxy - axbyx - bxby;
  const T denominator = leftData.affinePair.scale
      * rightData.affinePair.scale;
  CV_DbgAssert(denominator != 0);

  const T correlation = numerator / denominator;
//  cout << correlation << endl;
  CV_DbgAssert(correlation <= 1 + epsilon());
  CV_DbgAssert(correlation >= -1 - epsilon());
  return correlation;
}

/**
 * Performs correlation (not convolution) between two signals, assuming
 * they were originally purely real and the have already been mapped
 * into Fourier space.
 */
template<typename T>
void correlationFromPreprocessed(const Mat& left, const Mat& right, Mat_<T>& correlation) {
  CV_DbgAssert(left.type() == CV_64FC2);

  CV_DbgAssert(left.size() == right.size());
  CV_DbgAssert(left.type() == right.type());

  Mat_<cv::Vec<T,2> > dotTimes(left.size());
 
  const double* p_left = left.ptr<double>(0), *p_right = right.ptr<double>(0);
  T* p_res = dotTimes.template ptr<T>(0), *p_end = p_res + 2*left.cols*left.rows;
  for(; p_res != p_end;) {
    const double & a = *p_left++;
    const double & b = *p_left++;
    const double & c = *p_right++;
    const double & d = *p_right++;
    *p_res++ = a*c + b*d;
    *p_res++ = a*d - b*c;
  }

  ifft2D(dotTimes, correlation);
}

/**
 * The map of normalized correlations.
 */
template<typename T>
T getResponseMapMax(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock) {
  CV_DbgAssert(leftBlock.fourierData.rows == rightBlock.fourierData.rows);
  CV_DbgAssert(leftBlock.fourierData.cols == rightBlock.fourierData.cols);
  // The data has been zero padded in the vertical direction, which is
  // why we're dividing by 2 here.
  CV_DbgAssert(scaleSearchRadius < leftBlock.fourierData.rows / 2);

//  cout << leftBlock.fourierData.rows << endl;
//  cout << leftBlock.fourierData.cols << endl;
//  cout << leftBlock.fourierData.channels() << endl;

  // This is real valued.
  Mat_<T> correlation;
  correlationFromPreprocessed(rightBlock.fourierData, leftBlock.fourierData, correlation);
  CV_DbgAssert(correlation.type() == CV_64FC1);

//  cout << correlation << endl;

//  Mat normalized = correlation.clone();
//  for (int scaleOffset = -scaleSearchRadius; scaleOffset <= scaleSearchRadius;
//      ++scaleOffset) {
//    const int rowIndex = mod(scaleOffset, leftBlock.fourierData.rows);
//    for (int col = 0; col < correlation.cols; ++col) {
//      const double dotProduct = correlation.at<double>(rowIndex, col);
//      const double normalizedValue = nccFromUnnormalized(
//          leftBlock.scaleMap.get(scaleOffset),
//          rightBlock.scaleMap.get(-scaleOffset), dotProduct);
//      normalized.at<double>(rowIndex, col) = normalizedValue;
//    }
//  }

  T maxi = std::numeric_limits<T>::min();

  for (int scaleOffset = -scaleSearchRadius; scaleOffset <= scaleSearchRadius;
      ++scaleOffset) {
    const int correlationRowIndex = mod(scaleOffset, leftBlock.fourierData.rows);
    const T * p_correlation = correlation.template ptr<T>(correlationRowIndex), *p_correlation_end = p_correlation+correlation.cols;
    for (; p_correlation != p_correlation_end;) {
      maxi = std::max(maxi, nccFromUnnormalized<T>(
          leftBlock.scaleMap.get(scaleOffset),
          rightBlock.scaleMap.get(-scaleOffset), *p_correlation++));
    }
  }

  return maxi;
}

/**
 * The distance between two descriptors.
 */
template<typename T>
T distance(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right) {
  return dotProductToL2Distance<T>(getResponseMapMax<T>(self.scaleSearchRadius, left, right));
}

//double distance(const int scaleSearchRadius, const Mat& leftBlock,
//                const Mat& rightBlock) {
//  return distanceInternal(NCCLogPolarMatcher(scaleSearchRadius),
//                          matToNCCBlock(leftBlock).get(),
//                          matToNCCBlock(rightBlock).get());
//}

VectorNCCBlock flatten(const VectorOptionNCCBlock& options) {
  vector<NCCBlock> data;
  for (int index = 0; index < options.getSize(); ++index) {
    if (options.isDefinedAtIndex(index)) {
      data.push_back(options.getAtIndex(index));
    }
  }
  return VectorNCCBlock(data);
}

/**
 * Match all descriptors on the left to all descriptors on the right,
 * return distances in a Mat where row indexes left and col indexes right.
 * Distances are -1 where invalid.
 */
Mat matchAllPairs(const NCCLogPolarMatcher& matcher, const VectorNCCBlock& lefts_,
                  const VectorNCCBlock& rights_) {
  const auto lefts = lefts_.data;
  const auto rights = rights_.data;

  Mat_<float> distances(lefts.size(), rights.size());
  float* p_distances = distances.ptr<float>(0);

  //#pragma omp parallel for
  for (int row = 0; row < distances.rows; ++row) {

    for (int col = 0; col < distances.cols; ++col, ++p_distances) {
      const NCCBlock& left = lefts.at(row);
      const NCCBlock& right = rights.at(col);

      *p_distances = distance<float>(matcher, left, right);
    }
  }
  return distances;
}

///**
// * For debugging.
// */
//Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
//                                const int numScales, const int numAngles,
//                                const double blurWidth,
//                                const int scaleSearchRadius, const Mat& image,
//                                const KeyPoint& left, const KeyPoint& right) {
//  const NCCLogPolarExtractor extractor(minRadius, maxRadius, numScales,
//                                       numAngles, blurWidth);
//
//  const optional<NCCBlock> leftDescriptorOption = extractSingle(extractor,
//                                                                image, left);
//  const optional<NCCBlock> rightDescriptorOption = extractSingle(extractor,
//                                                                 image, right);
//
//  if (!leftDescriptorOption.is_initialized()
//      || !rightDescriptorOption.is_initialized()) {
//    return Mat();
//  } else {
//    const NCCBlock leftDescriptor = leftDescriptorOption.get();
//    const NCCBlock rightDescriptor = rightDescriptorOption.get();
//
//    const NCCLogPolarMatcher matcher(scaleSearchRadius);
//
//    return getDistanceMap(matcher, leftDescriptor, rightDescriptor);
//  }
//}


}
