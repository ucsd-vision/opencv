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

Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock);

Mat responseMapToDistanceMap(const Mat& responseMap);

Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
                   const NCCBlock& right);

Mat matchAllPairs(const int scaleSearchRadius, const vector<Option<NCCBlock> >& lefts,
                  const vector<Option<NCCBlock> >& rights);

double distanceInternal(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right);

//CV_EXPORTS_W Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
//                                const int numScales, const int numAngles,
//                                const double blurWidth,
//                                const int scaleSearchRadius, const Mat& image,
//                                const KeyPoint& left, const KeyPoint& right);

}  // namespace cv

#endif
