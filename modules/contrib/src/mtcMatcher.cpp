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
  CV_Assert(leftData.size == rightData.size);

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
  CV_Assert(denominator != 0);

  const double correlation = numerator / denominator;
  cout << correlation << endl;
  CV_Assert(correlation <= 1 + epsilon());
  CV_Assert(correlation >= -1 - epsilon());
  return correlation;
}

/**
 * Performs correlation (not convolution) between two signals, assuming
 * they were originally purely real and the have already been mapped
 * into Fourier space.
 */
Mat correlationFromPreprocessed(const Mat& left, const Mat& right) {
  CV_Assert(left.type() == CV_64FC2);
  CV_Assert(left.channels() == 2);

  CV_Assert(left.rows == right.rows);
  CV_Assert(left.cols == right.cols);
  CV_Assert(left.channels() == right.channels());
  CV_Assert(left.type() == right.type());

  vector<Mat> leftLayers;
  split(left, leftLayers);
  CV_Assert(leftLayers.size() == 2);
  const auto& leftReal = leftLayers.at(0);
  const auto& leftImaginary = leftLayers.at(1);

  vector<Mat> rightLayers;
  split(right, rightLayers);
  CV_Assert(rightLayers.size() == 2);
  const auto& rightReal = rightLayers.at(0);
  const auto& rightImaginary = rightLayers.at(1);

  // Now we do pairwise multiplication of the _conjugate_ of the left
  // matrix and the right matrix.
  const auto realPart =
      leftReal.mul(rightReal) + leftImaginary.mul(rightImaginary);
  const auto imaginaryPart =
      leftReal.mul(rightImaginary) - leftImaginary.mul(rightReal);

  vector<Mat> dotTimesLayers = {realPart, imaginaryPart};
  Mat dotTimes;



  merge(dotTimesLayers, dotTimes);

//  // The complex conjugate of the left Mat.
//  Mat leftConjugate(left.rows, left.cols, left.type());
//  for (int row = 0; row < left.rows; ++row) {
//    // The input matrices store complex values, with the odd number indices
//    // storing the imaginary parts. So we start at 1 and stride through by 2.
//    for (int col = 1; col < left.cols; col += 2) {
//      leftConjugate.at<double>(row, col) = -left.at<double>(row, col);
//    }
//  }
//
//  // Now we do a pairwise multiplication of complex values.
//  Mat dotTimes;
//  for (int row = 0; row < left.rows; ++row) {
//    for (int col = 0; col < left.cols; col += 2) {
//      const double leftReal = left.at<double>(row, col);
//      const double leftImaginary = left.at<double>(row, col + 1);
//      const double rightReal = right.at<double>(row, col);
//      const double rightImaginary = right.at<double>(row, col + 1);
//
//      const double productReal = leftReal * rightReal
//          - leftImaginary * rightImaginary;
//      const double productImaginary = leftReal * rightImaginary
//          + leftImaginary * rightReal;
//      dotTimes.at<double>(row, col) = productReal;
//      dotTimes.at<double>(row, col + 1) = productImaginary;
//    }
//  }

  return ifft2DDouble(dotTimes);
}

/**
 * The map of normalized correlations.
 */
Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock) {
  CV_Assert(leftBlock.fourierData.rows == rightBlock.fourierData.rows);
  CV_Assert(leftBlock.fourierData.cols == rightBlock.fourierData.cols);
  // The data has been zero padded in the vertical direction, which is
  // why we're dividing by 2 here.
  CV_Assert(scaleSearchRadius < leftBlock.fourierData.rows / 2);

  cout << leftBlock.fourierData.rows << endl;
  cout << leftBlock.fourierData.cols << endl;
  cout << leftBlock.fourierData.channels() << endl;

  // This is real valued.
  const Mat correlation = correlationFromPreprocessed(rightBlock.fourierData,
                                                      leftBlock.fourierData);
  CV_Assert(correlation.type() == CV_64FC1);

  cout << correlation << endl;

  Mat normalized(correlation);
  for (int scaleOffset = -scaleSearchRadius; scaleOffset <= scaleSearchRadius;
      ++scaleOffset) {
    const int rowIndex = mod(scaleOffset, leftBlock.fourierData.rows);
    for (int col = 0; col < correlation.cols; ++col) {
      cout << scaleOffset << endl;
      const double dotProduct = correlation.at<double>(rowIndex, col);
      cout << dotProduct << endl;
      const double normalized = nccFromUnnormalized(
          leftBlock.scaleMap.get(scaleOffset),
          rightBlock.scaleMap.get(-scaleOffset), dotProduct);
    }
  }

  cout << normalized << endl;

  return normalized;
}

/**
 * The map of distances.
 */
Mat responseMapToDistanceMap(const Mat& responseMap) {
  CV_Assert(responseMap.type() == CV_64FC1);

  Mat distances(responseMap.size(), responseMap.type());

  MatConstIterator_<double> response = responseMap.begin<double>();
  MatIterator_<double> distance = distances.begin<double>();
  for (; response != responseMap.end<double>(); ++response, ++distance) {
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
double distanceInternal(const NCCLogPolarMatcher& self, const NCCBlock& left,
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

/**
 * Match all descriptors on the left to all descriptors on the right,
 * return distances in a Mat where row indexes left and col indexes right.
 * Distances are -1 where invalid.
 * If the distance is symmetric, there is redundancy across the diagonal.
 */
Mat matchAllPairs(const int scaleSearchRadius, const vector<Option<NCCBlock> >& lefts,
                  const vector<Option<NCCBlock> >& rights) {
  const NCCLogPolarMatcher matcher(scaleSearchRadius);

  Mat distances(lefts.size(), rights.size(), CV_64FC1, Scalar(-1));
  for (int row = 0; row < distances.rows; ++row) {
    for (int col = 0; col < distances.cols; ++col) {
      const Option<NCCBlock>& left = lefts.at(row);
      const Option<NCCBlock>& right = rights.at(col);

      if (isDefined(left) && isDefined(right)) {
        distances.at<double>(row, col) = distanceInternal(matcher, get(left),
                                                          get(right));
      }
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
