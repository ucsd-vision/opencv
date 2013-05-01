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
 * Determine what the dot product would have been had the vectors been
 * normalized first.
 */
double nccFromUnnormalized(const NormalizationData& leftData,
                           const NormalizationData& rightData,
                           const double unnormalizedInnerProduct) {
  CV_DbgAssert(leftData.size == rightData.size);

  // Suppose we observe the inner product between two vectors
  // (a_x * x + b_x) and (a_y * y + b_y), where x and y are normalized.
  // Note (a_x * x + b_x)^T (a_y * y + b_y) is
  // (a_x * x)^T (a_y * y) + a_y * b_x^T y + a_x * b_y^T x + b_x^T b_y.
  // Thus we can solve for the normalized dot product:
  // x^T y = ((a_x * x)^T (a_y * y) - a_y * b_x^T y - a_x * b_y^T x - b_x^T b_y) / (a_x * a_y).
  const double aybxy = rightData.affinePair.scale * leftData.affinePair.offset
      * rightData.elementSum;

  const double axbyx = leftData.affinePair.scale * rightData.affinePair.offset
      * leftData.elementSum;

  const double bxby = leftData.size * leftData.affinePair.offset
      * rightData.affinePair.offset;

  const double numerator = unnormalizedInnerProduct - aybxy - axbyx - bxby;
  const double denominator = leftData.affinePair.scale
      * rightData.affinePair.scale;
  CV_DbgAssert(denominator != 0);

  const double correlation = numerator / denominator;
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
Mat correlationFromPreprocessed(const Mat& left, const Mat& right) {
  CV_DbgAssert(left.type() == CV_64FC2);

  CV_DbgAssert(left.size() == right.size());
  CV_DbgAssert(left.type() == right.type());

  Mat_<cv::Vec2d> dotTimes(left.size());
 
  double* p_left = reinterpret_cast<double*>(left.data), *p_right = reinterpret_cast<double*>(right.data);
  double* p_res = reinterpret_cast<double*>(dotTimes.data), *p_end = p_res + 2*left.cols*left.rows;
  for(; p_res != p_end;) {
    const double & a = *p_left++;
    const double & b = *p_left++;
    const double & c = *p_right++;
    const double & d = *p_right++;
    *p_res++ = a*c + b*d;
    *p_res++ = a*d - b*c;
  }
    
  return ifft2DDouble(dotTimes);
}

/**
 * The map of normalized correlations.
 */
Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
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
  const Mat correlation = correlationFromPreprocessed(rightBlock.fourierData,
                                                      leftBlock.fourierData);
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

  Mat_<double> normalized(2 * scaleSearchRadius + 1, correlation.cols);
  double * p_normalized = normalized.ptr<double>(0);
  for (int scaleOffset = -scaleSearchRadius; scaleOffset <= scaleSearchRadius;
      ++scaleOffset) {
    const int correlationRowIndex = mod(scaleOffset, leftBlock.fourierData.rows);
    const double * p_correlation = correlation.ptr<double>(correlationRowIndex), *p_correlation_end = p_correlation+correlation.cols;
    for (; p_correlation != p_correlation_end;) {
      const double normalizedValue = nccFromUnnormalized(
          leftBlock.scaleMap.get(scaleOffset),
          rightBlock.scaleMap.get(-scaleOffset), *p_correlation++);
      *p_normalized++ = normalizedValue;
    }
  }

  return normalized;
}

/**
 * The map of distances.
 */
Mat responseMapToDistanceMap(const Mat& responseMap) {
  CV_DbgAssert(responseMap.type() == CV_64FC1);

  Mat_<double> distances(responseMap.size());

  const double * response = responseMap.ptr<double>(0), *response_end = response + responseMap.total();
  double * distance = distances.ptr<double>(0);
  for (; response != response_end; ++response, ++distance) {
    *distance = dotProductToL2Distance(*response);
  }
  return distances;
}

Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
                   const NCCBlock& right) {
  const Mat responseMap = getResponseMap(self.scaleSearchRadius, left, right);
  return responseMapToDistanceMap(responseMap);
}

/**
 * The distance between two descriptors.
 */
double distance(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right) {
  const Mat distanceMap = getDistanceMap(self, left, right);
  return *min_element(distanceMap.begin<double>(), distanceMap.end<double>());
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

  Mat distances(lefts.size(), rights.size(), CV_64FC1, Scalar(-1));

  //#pragma omp parallel for
  for (int row = 0; row < distances.rows; ++row) {

    for (int col = 0; col < distances.cols; ++col) {
      const NCCBlock& left = lefts.at(row);
      const NCCBlock& right = rights.at(col);

      distances.at<double>(row, col) = distance(matcher, left,
                                                          right);
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
