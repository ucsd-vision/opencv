#ifndef __OPENCV_MTCSAMPLEPATTERN_HPP__
#define __OPENCV_MTCSAMPLEPATTERN_HPP__

#include "opencv2/core/core.hpp"

#include "opencv2/features2d/features2d.hpp"
#include <algorithm>
#include <vector>

#include <iostream>
#include <iomanip>

#include "boost/tuple/tuple.hpp"
#include "boost/optional/optional.hpp"
#include <boost/foreach.hpp>
#include <cmath>

#include "mtcUtil.hpp"

/////////////////////////////////////////////////////

namespace cv {

using namespace std;

///**
// * Like boost::optional (really, Scala's Option), but with CV_WRAP.
// * The wrapper doesn't like templates, so you need to call this macro to create
// * each option type you want to use.
// */
//#define CREATE_OPTION_TYPE(type) \
//  struct CV_EXPORTS_W type ## Option { \
//    CV_WRAP type value; \
//    CV_WRAP bool isDefined; \
//    type ## Option() : isDefined(false) {} \
//    CV_WRAP type ## Option(const type& value_) : value(value_), isDefined(true) {} \
//    CV_WRAP const type& get() { CV_Assert(isDefined); return value; } \
//  }
//


CV_EXPORTS_W vector<double> getScaleFactors(const double samplingRadius,
                                            const double minRadius,
                                            const double maxRadius,
                                            const int numScales);

CV_EXPORTS_W Mat getRealScaleTargetsMat(
    const vector<double>& idealScalingFactors, const int imageWidth,
    const int imageHeight);

CV_EXPORTS_W vector<Mat> scaleImagesOnly(const double samplingRadius,
                                         const double minRadius,
                                         const double maxRadius,
                                         const double numScales,
                                         const double blurWidth,
                                         const Mat& image);

CV_EXPORTS_W int sampleSubPixelGray(const Mat& image, double x, double y);

CV_EXPORTS_W Point2f samplePoint(const double samplingRadius,
                                 const int numAngles,
                                 const double realScaleFactorX,
                                 const double realScaleFactorY,
                                 const int angleIndex, const Point2f& keyPoint);


ScaleMap<NormalizationData> getScaleMap(const Mat& descriptor);

}

#endif
