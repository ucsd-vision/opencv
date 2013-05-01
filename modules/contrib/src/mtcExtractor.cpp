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
#include "opencv2/contrib/mtcExtractor.hpp"


///////////////////////////////////////////////////////////

namespace cv {

using namespace std;
using boost::optional;

/**
 * Normalize descriptor to have zero mean and unit norm.
 */
Mat normalizeL2(const Mat& descriptor) {
  Mat doubleDescriptor;
  descriptor.convertTo(doubleDescriptor, CV_64F);

  const double offset = mean(doubleDescriptor).val[0];
  const Mat centered = doubleDescriptor - offset;
  const double scale = norm(centered);
  // If this fails, it probably means you're on a uniform patch.
  // We address this now by returning random noise; basically a guaranteed
  // failure, so hopefully this doesn't happen too often.
  // TODO: Address this degenerate case.
//  if (scale == 0) {
//    const Mat noise(descriptor.rows, descriptor.cols, descriptor.type());
//    randn(noise, 0, 1);
//    return normalizeL2(noise);
//  }
  CV_DbgAssert(scale > 0);
  return centered / scale;
//
//  const AffinePair affinePair = getAffinePair(descriptor);
//  return (descriptor - affinePair.offset) / affinePair.scale;
}



/**
 * Find the affine pair that normalizes this matrix.
 */
AffinePair getAffinePair(const Mat& descriptor) {
  CV_DbgAssert(descriptor.type() == CV_8UC1);
  CV_DbgAssert(descriptor.total() > 1);

  Mat doubleDescriptor;
  descriptor.convertTo(doubleDescriptor, CV_64F);

  const double offset = mean(doubleDescriptor).val[0];
  const double scale = norm(doubleDescriptor - offset);
//  CV_DbgAssert(scale > 0);
  return AffinePair(scale, offset);
}

/**
 * Get the normalization data for a matrix.
 */
NormalizationData getNormalizationData(const Mat& descriptor) {
  CV_DbgAssert(descriptor.type() == CV_8UC1);
  const AffinePair affinePair = getAffinePair(descriptor);

  // Check if we're dealing with a uniform patch.
  // If so, we give it an artificially high scale so it will fail to match
  // everything.
  if (affinePair.scale == 0) {
    return NormalizationData(AffinePair(1, affinePair.offset),
                             0,
                             descriptor.total());
  }

  return NormalizationData(getAffinePair(descriptor),
                           sum(normalizeL2(descriptor)).val[0],
                           descriptor.total());
}

/**
 * Get the scale map for an entire log-polar pattern.
 */
ScaleMapNormalizationData getScaleMap(const Mat& descriptor) {
  CV_DbgAssert(descriptor.type() == CV_8UC1);

  CV_DbgAssert(descriptor.rows > 0);
  CV_DbgAssert(descriptor.cols > 1);

  const int numScales = descriptor.rows;

  map<int, NormalizationData> data;
  for (int scaleOffset = -numScales + 1; scaleOffset <= numScales - 1;
      ++scaleOffset) {
    const int start = max(scaleOffset, 0);
    const int stop = min(numScales, scaleOffset + numScales);

    const Mat roi = descriptor(Range(start, stop), Range::all());
    CV_DbgAssert(roi.rows == stop - start);
    CV_DbgAssert(roi.cols == descriptor.cols);

    getNormalizationData(roi);

    data[scaleOffset] = getNormalizationData(roi);;
  }

  return ScaleMapNormalizationData(data);
}



/**
 * Get a descriptor from an entire log-polar pattern.
 */
NCCBlock getNCCBlock(const Mat& samples) {
  CV_DbgAssert(samples.type() == CV_8UC1);

  // We require the descriptor width and height each be a power of two.
  CV_DbgAssert(isPowerOfTwo(samples.rows));
  CV_DbgAssert(samples.cols > 1 && isPowerOfTwo(samples.cols));

  const ScaleMapNormalizationData scaleMap = getScaleMap(samples);

  const Mat zeroPadding = Mat::zeros(samples.rows, samples.cols,
                                     samples.type());
  Mat padded;
  vconcat(samples, zeroPadding, padded);
  // For now, we're working with floating point values.
  Mat converted;
  padded.convertTo(converted, CV_64FC1);
  Mat fourierData;
  fft2D(converted, fourierData);

  return NCCBlock(fourierData, scaleMap);
}

/**
 * Extract descriptors from the given keypoints.
 */
VectorOptionNCCBlock extract(const NCCLogPolarExtractor& self, const Mat& image,
                             const vector<KeyPoint>& keyPoints) {
  const vector<Option<Mat> > sampleOptions = rawLogPolarSeq(self.minRadius,
                                                            self.maxRadius,
                                                            self.numScales,
                                                            self.numAngles,
                                                            self.blurWidth,
                                                            image, keyPoints);
  CV_DbgAssert(sampleOptions.size() == keyPoints.size());

  vector<Option<NCCBlock> > out;
  for (const auto& sampleOption : sampleOptions) {
    if (isDefined(sampleOption)) {
      const Mat sample = get(sampleOption);
      CV_DbgAssert(sample.rows == self.numScales);
      CV_DbgAssert(sample.cols == self.numAngles);
      out.push_back(Some<NCCBlock>(getNCCBlock(sample)));
    } else {
      out.push_back(None<NCCBlock>());
    }
  }

  CV_DbgAssert(out.size() == keyPoints.size());
  return VectorOptionNCCBlock(out);
}

}  // namespace cv

