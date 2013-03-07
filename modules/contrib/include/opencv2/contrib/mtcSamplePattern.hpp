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

CV_EXPORTS_W vector<double> getScaleFactors(const double samplingRadius,
                                            const double minRadius,
                                            const double maxRadius,
                                            const int numScales);

vector<tuple<tuple<int, int>, tuple<double, double>>> getRealScaleTargets(
    const vector<double>& idealScalingFactors, const int imageWidth,
    const int imageHeight);

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

/**
 * vector<Option<Mat>>
 */
CV_EXPORTS_W vector<vector<Mat> > rawLogPolarSeq(
    const double minRadius, const double maxRadius, const int numScales,
    const int numAngles, const double blurWidth, const Mat& image,
    const vector<KeyPoint>& keyPoints);

}  // namespace cv

#endif
