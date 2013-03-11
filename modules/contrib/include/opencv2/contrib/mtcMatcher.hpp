#ifndef __OPENCV_MTCMATCHER_HPP__
#define __OPENCV_MTCMATCHER_HPP__

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
#include "mtcExtractor.hpp"

/////////////////////////////////////////////

namespace cv {

using namespace std;

struct CV_EXPORTS_W NCCLogPolarMatcher {
  CV_WRAP int scaleSearchRadius;

  NCCLogPolarMatcher() {
  }

  CV_WRAP NCCLogPolarMatcher(const int scaleSearchRadius_)
      : scaleSearchRadius(scaleSearchRadius_) {
    CV_Assert(scaleSearchRadius >= 0);
  }
};

CV_WRAP double nccFromUnnormalized(const NormalizationData& leftData,
                           const NormalizationData& rightData,
                           const double unnormalizedInnerProduct);

CV_WRAP Mat correlationFromPreprocessed(const Mat& left, const Mat& right);

CV_WRAP Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock);

CV_WRAP Mat responseMapToDistanceMap(const Mat& responseMap);

CV_WRAP Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
                   const NCCBlock& right);

CV_WRAP double distance(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right);

/**
 * Wrapper generator workaround.
 */
struct CV_EXPORTS_W VectorNCCBlock {
  vector<NCCBlock> data;

  CV_WRAP int getSize() { return data.size(); }

  CV_WRAP const NCCBlock& getAtIndex(const int index) {
    return data.at(index);
  }

  CV_WRAP void pushBack(const NCCBlock& element) { data.push_back(element); }

  CV_WRAP VectorNCCBlock() {}

  VectorNCCBlock(const vector<NCCBlock>& data_) : data(data_) {}
};

CV_WRAP VectorNCCBlock flatten(const VectorOptionNCCBlock& options);

CV_WRAP Mat matchAllPairs(const NCCLogPolarMatcher& matcher, const VectorNCCBlock& lefts,
                  const VectorNCCBlock& rights);

//CV_EXPORTS_W Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
//                                const int numScales, const int numAngles,
//                                const double blurWidth,
//                                const int scaleSearchRadius, const Mat& image,
//                                const KeyPoint& left, const KeyPoint& right);

}  // namespace cv

#endif
