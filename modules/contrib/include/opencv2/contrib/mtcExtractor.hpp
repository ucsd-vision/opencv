#ifndef __OPENCV_MTCEXTRACTOR_HPP__
#define __OPENCV_MTCEXTRACTOR_HPP__

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

using namespace std;

/**
 * The two values that characterize a 1D affine function.
 */
struct CV_EXPORTS_W AffinePair {
  CV_WRAP double scale;
  CV_WRAP double offset;

  CV_WRAP double getScale() const { return scale; }
  CV_WRAP double getOffset() const { return offset; }

  AffinePair() {
  }

  CV_WRAP AffinePair(const double scale_, const double offset_)
      : scale(scale_),
        offset(offset_) {
  }
};

/**
 * Data needed to determine normalized dot product from dot product
 * of unnormalized vectors.
 */
struct CV_EXPORTS_W NormalizationData {
  CV_WRAP AffinePair affinePair;
  // This is the sum of the elements of the normalized vector.
  CV_WRAP double elementSum;
  CV_WRAP int size;

  CV_WRAP AffinePair getAffinePair() const { return affinePair; }
  CV_WRAP double getElementSum() const { return elementSum; }
  CV_WRAP int getSize() const { return size; }

  NormalizationData() {
  }

  CV_WRAP NormalizationData(
      const AffinePair& affinePair_,
      const double elementSum_,
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
//template<class A>
//struct ScaleMap {
//  int radius;
//  vector<A> privateData;
//
//  const A& get(int index) const {
//    CV_Assert(index >= -radius);
//    CV_Assert(index <= radius);
//
//    return privateData.at(index + radius);
//  }
//
//  ScaleMap() {
//  }
//
//  ScaleMap(const std::map<int, A>& data_) {
//    vector<int> keys;
//
//    privateData.resize(keys.size());
//    for (typename std::map<int, A>::const_iterator keyValue = data_.begin();
//        keyValue != data_.end(); ++keyValue) {
//      keys.push_back(keyValue->first);
//      privateData.at(keyValue->first + data_.size() / 2) = keyValue->second;
//    }
//
//    // Now keys is sorted.
//    sort(keys.begin(), keys.end());
//    const int minKey = keys.at(0);
//    const int maxKey = keys.at(keys.size() - 1);
//
//    CV_Assert(-minKey == maxKey);
//    for (int index = 0; index < keys.size(); ++index) {
//      CV_Assert(keys.at(index) == index + minKey);
//    }
//  }
//};

///**
// * Duplicated so the wrapper generator will find it.
// *
// * A mapping from scale levels. Scale levels must be a sequential and
// * symmetric about zero.
// */
struct CV_EXPORTS_W ScaleMapNormalizationData {
  CV_WRAP int radius;
  vector<NormalizationData> privateData;

  CV_WRAP int getRadius() const { return radius; }

  CV_WRAP const NormalizationData& get(int index) const {
    CV_Assert(index >= -radius);
    CV_Assert(index <= radius);

    return privateData.at(index + radius);
  }

  NormalizationData& get(int index) {
    CV_Assert(index >= -radius);
    CV_Assert(index <= radius);

    return privateData.at(index + radius);
  }

  ScaleMapNormalizationData() {
  }

  ScaleMapNormalizationData(const std::map<int, NormalizationData>& data_) {
    privateData.resize(data_.size());
    radius = data_.size() / 2;

    vector<int> keys;
    for (typename std::map<int, NormalizationData>::const_iterator keyValue = data_.begin();
        keyValue != data_.end(); ++keyValue) {
      keys.push_back(keyValue->first);
      get(keyValue->first) = keyValue->second;
//      privateData.at(keyValue->first + data_.size() / 2) = keyValue->second;
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
  CV_WRAP Mat fourierData;
  CV_WRAP ScaleMapNormalizationData scaleMap;

  CV_WRAP Mat getFourierData() const { return fourierData; }
  CV_WRAP ScaleMapNormalizationData getScaleMap() const { return scaleMap; }

  NCCBlock() {
  }

  CV_WRAP NCCBlock(const Mat& fourierData_,
           const ScaleMapNormalizationData& scaleMap_)
      : fourierData(fourierData_),
        scaleMap(scaleMap_) {
    CV_Assert(fourierData.rows - 1 == 2 * scaleMap.radius + 1);
  }
};

/**
 * The extractor.
 * numScales and numAngles must be powers of 2.
 * numAngles must be >= 2.
 */
struct CV_EXPORTS_W NCCLogPolarExtractor {
  double minRadius;
  double maxRadius;
  int numScales;
  int numAngles;
  double blurWidth;

  CV_WRAP double getMinRadius() { return minRadius; }

  CV_WRAP double getMaxRadius() { return maxRadius; }

  CV_WRAP int getNumScales() const { return numScales; }

  CV_WRAP int getNumAngles() const { return numAngles; }

  CV_WRAP double getBlurWidth() const { return blurWidth; }

  NCCLogPolarExtractor() {
  }

  CV_WRAP NCCLogPolarExtractor(const double minRadius_, const double maxRadius_,
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

CV_EXPORTS_W AffinePair getAffinePair(const Mat& descriptor);

CV_EXPORTS_W NormalizationData getNormalizationData(const Mat& descriptor);

CV_EXPORTS_W ScaleMapNormalizationData getScaleMap(const Mat& descriptor);

CV_EXPORTS_W NCCBlock getNCCBlock(const Mat& samples);

/**
 * Wrapper generator workaround.
 */
struct CV_EXPORTS_W VectorOptionNCCBlock {
  vector<Option<NCCBlock>> data;

  CV_WRAP int getSize() const { return data.size(); }

  CV_WRAP bool isDefinedAtIndex(const int index) const {
    return isDefined(data.at(index));
  }

  CV_WRAP const NCCBlock& getAtIndex(const int index) const {
    return get(data.at(index));
  }

  VectorOptionNCCBlock() {}

  VectorOptionNCCBlock(const vector<Option<NCCBlock>>& data_) : data(data_) {}
};

CV_WRAP VectorOptionNCCBlock extract(const NCCLogPolarExtractor& self,
                                  const Mat& image,
                                  const vector<KeyPoint>& keyPoints);

}  // namespace cv

#endif
