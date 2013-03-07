#ifndef __OPENCV_MTC_HPP__
#define __OPENCV_MTC_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui_c.h"

#include "opencv2/features2d/features2d.hpp"
#include <algorithm>
#include <vector>

#include <iostream>
#include <iomanip>

#include "boost/tuple/tuple.hpp"
#include "boost/optional/optional.hpp"
#include <boost/foreach.hpp>
#include <cmath>

#include "mtcSamplePattern.hpp"
#include "mtcUtil.hpp"
#include "mtcWrapper.hpp"

/////////////////////////////////////////////

namespace cv {

/**
 * The two values that characterize a 1D affine function.
 */
struct CV_EXPORTS_W AffinePair {
  // Stupid fucking assignment operator. How do I make this const?
  CV_WRAP
  double scale;CV_WRAP
  double offset;

  AffinePair() {
  }

  AffinePair(const double scale_, const double offset_)
      : scale(scale_),
        offset(offset_) {
  }
};

CV_EXPORTS_W AffinePair getAffinePair(const Mat& descriptor);

/**
 * Data needed to determine normalized dot product from dot product
 * of unnormalized vectors.
 */
struct CV_EXPORTS_W NormalizationData {
  CV_WRAP
  AffinePair affinePair;
  // This is the sum of the elements of the normalized vector.
  CV_WRAP
  double elementSum;CV_WRAP
  int size;

  NormalizationData() {
  }

  NormalizationData(const AffinePair& affinePair_, const double elementSum_,
                    const int size_)
      : affinePair(affinePair_),
        elementSum(elementSum_),
        size(size_) {
  }
};

/**
 * A mapping from scale levels. Scale levels must be a sequential and
 * symmetric about zero.
 */
template<class A>
struct ScaleMap {
  std::map<int, A> data;

  ScaleMap() {
  }

  ScaleMap(const std::map<int, A>& data_)
      : data(data_) {
    vector<int> keys;

    for (typename std::map<int, A>::const_iterator keyValue = data.begin();
        keyValue != data.end(); ++keyValue) {
      keys.push_back(keyValue->first);
    }

    // Now keys is sorted.
    sort(keys.begin(), keys.end());
    const int minKey = keys.at(0);
    const int maxKey = keys.at(keys.size() - 1);

    CV_Assert(-minKey == maxKey);
    for (int index = 0; index < keys.size(); ++index) {
      CV_Assert(keys.at(index) == index + minKey);
    }
  }
};

/**
 * The descriptor. Contains a Fourier-space version of the log polar
 * data as well as normalization data for each scale.
 */
struct CV_EXPORTS_W NCCBlock {
  CV_WRAP
  Mat fourierData;CV_WRAP
  ScaleMap<NormalizationData> scaleMap;

  NCCBlock() {
  }

  NCCBlock(const Mat& fourierData_,
           const ScaleMap<NormalizationData>& scaleMap_)
      : fourierData(fourierData_),
        scaleMap(scaleMap_) {
    CV_Assert(fourierData.rows - 1 == scaleMap.data.size());
  }
};

/**
 * The extractor.
 * numScales and numAngles must be powers of 2.
 * numAngles must be >= 2.
 */
struct CV_EXPORTS_W NCCLogPolarExtractor {
  CV_WRAP
  double minRadius;CV_WRAP
  double maxRadius;CV_WRAP
  int numScales;CV_WRAP
  int numAngles;CV_WRAP
  double blurWidth;

  NCCLogPolarExtractor() {
  }

  NCCLogPolarExtractor(const double minRadius_, const double maxRadius_,
                       const int numScales_, const int numAngles_,
                       const double blurWidth_)
      : minRadius(minRadius_),
        maxRadius(maxRadius_),
        numScales(numScales_),
        numAngles(numAngles_),
        blurWidth(blurWidth_) {
    CV_Assert(isPowerOfTwo(numScales));
    CV_Assert(numAngles > 1 && isPowerOfTwo(numAngles));
  }
};

struct CV_EXPORTS_W NCCLogPolarMatcher {
  CV_WRAP
  int scaleSearchRadius;

  NCCLogPolarMatcher() {
  }

  NCCLogPolarMatcher(const int scaleSearchRadius_)
      : scaleSearchRadius(scaleSearchRadius_) {
    CV_Assert(scaleSearchRadius >= 0);
  }
};



NormalizationData getNormalizationData(const Mat& descriptor);

ScaleMap<NormalizationData> getScaleMap(const Mat& descriptor);

NCCBlock getNCCBlock(const Mat& samples);

vector<Option<NCCBlock> > extractInternal(const NCCLogPolarExtractor& self,
                                            const Mat& image,
                                            const vector<KeyPoint>& keyPoints);

//CV_EXPORTS_W Mat fft2DInteger(const Mat& spatialData);
//
//CV_EXPORTS_W Mat ifft2DInteger(const Mat& fourierData);

CV_EXPORTS_W Mat extract(const double minRadius, const double maxRadius,
                         const int numScales, const int numAngles,
                         const double blurWidth, const Mat& image,
                         const vector<KeyPoint>& keyPoints);

Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock);

Mat responseMapToDistanceMap(const Mat& responseMap);

Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
                   const NCCBlock& right);

CV_EXPORTS_W Mat matchAllPairs(const int scaleSearchRadius,
                               const Mat& leftBlocks, const Mat& rightBlocks);

double distanceInternal(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right);

//CV_EXPORTS_W Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
//                                const int numScales, const int numAngles,
//                                const double blurWidth,
//                                const int scaleSearchRadius, const Mat& image,
//                                const KeyPoint& left, const KeyPoint& right);

}
#endif
