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

#include "opencv2/contrib/mtcSamplePattern.hpp"

////////////////////////////////////////////

namespace cv {

using namespace std;
using boost::optional;

/**
 * Get image scaling factors. These are used to create the image pyramid from
 * which descriptors are extracted.
 */
vector<double> getScaleFactors(const double samplingRadius,
                               const double minRadius, const double maxRadius,
                               const int numScales) {
  const double maxScalingFactor = samplingRadius / minRadius;
  const double minScalingFactor = samplingRadius / maxRadius;
  CV_Assert(maxScalingFactor > minScalingFactor);
  const double base = exp(
      log(minScalingFactor / maxScalingFactor) / (numScales - 1));
  CV_Assert(base < 1);

  vector<double> scaleFactors;
  for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
    const double scaleFactor = maxScalingFactor * pow(base, scaleIndex);
    CV_Assert(scaleFactor >= minScalingFactor - epsilon());
    CV_Assert(scaleFactor <= maxScalingFactor + epsilon());
    if (scaleIndex == 0)
      assertNear(scaleFactor, maxScalingFactor);
    if (scaleIndex == numScales - 1)
      assertNear(scaleFactor, minScalingFactor);
    scaleFactors.push_back(scaleFactor);
  }

  return scaleFactors;
}

/**
 * Not all scaling factors can be perfectly realized, as images must have
 * integer width and height.
 * Returns the realizable (width, height) pairs that most closely match
 * the desired scaling factors.
 */
vector<tuple<tuple<int, int>, tuple<double, double>>> getRealScaleTargets(
    const vector<double>& idealScalingFactors, const int imageWidth,
    const int imageHeight) {
  vector<tuple<tuple<int, int>, tuple<double, double> > > scaleTargets;
  BOOST_FOREACH(const double scaleFactor, idealScalingFactors){
  const int scaledWidth = round(scaleFactor * imageWidth);
  const int scaledHeight = round(scaleFactor * imageHeight);

  const double realFactorX = static_cast<double>(scaledWidth) / imageWidth;
  const double realFactorY = static_cast<double>(scaledHeight) / imageHeight;

  scaleTargets.push_back(make_tuple(
          make_tuple(scaledWidth, scaledHeight),
          make_tuple(realFactorX, realFactorY)));
}
  return scaleTargets;
}

Mat getRealScaleTargetsMat(const vector<double>& idealScalingFactors,
                           const int imageWidth, const int imageHeight) {
  const vector<tuple<tuple<int, int>, tuple<double, double> > > tuples =
      getRealScaleTargets(idealScalingFactors, imageWidth, imageHeight);
  Mat out(tuples.size(), 4, CV_64FC1);
  for (int row = 0; row < tuples.size(); ++row) {
    const tuple<tuple<int, int>, tuple<double, double> > pair = tuples.at(row);
    out.at<double>(row, 0) = get<0>(get<0>(pair));
    out.at<double>(row, 1) = get<1>(get<0>(pair));
    out.at<double>(row, 2) = get<0>(get<1>(pair));
    out.at<double>(row, 3) = get<1>(get<1>(pair));
  }
  return out;
}

/**
 * Get the image pyramid, as well as the scaling factors that
 * were used to create it. Since the ideal scaling factors cannot be
 * used in all cases (integer rounding), we return the ideal factors
 * plus the actual factors used.
 */
tuple<vector<double>, vector<tuple<double, double> >, vector<Mat> > scaleImage(
    const double samplingRadius, const double minRadius, const double maxRadius,
    const double numScales, const double blurWidth, const Mat& image) {
  const vector<double> idealScaleFactors = getScaleFactors(samplingRadius,
                                                           minRadius, maxRadius,
                                                           numScales);
  Mat blurred;
//  blur(image, blurred, Size(blurWidth, blurWidth));
  GaussianBlur(image, blurred, Size(0, 0), blurWidth, 0);

  const vector<tuple<tuple<int, int>, tuple<double, double> > > scaleSizesAndFactors =
      getRealScaleTargets(idealScaleFactors, blurred.cols, blurred.rows);
  CV_Assert(idealScaleFactors.size() == scaleSizesAndFactors.size());

  vector<tuple<double, double> > realScaleFactors;
  vector<Mat> scaledImages;
  for (int index = 0; index < scaleSizesAndFactors.size(); ++index) {
    const tuple<double, double> realScaleFactor = get<1>(scaleSizesAndFactors.at(index));
    realScaleFactors.push_back(realScaleFactor);

    const tuple<int, int> realScaleSize =
        get<0>(scaleSizesAndFactors.at(index));
    Mat resized;
    resize(blurred, resized,
           Size(get<0>(realScaleSize), get<1>(realScaleSize)), 0, 0,
           INTER_CUBIC);
    scaledImages.push_back(resized);
  }
  return make_tuple(idealScaleFactors, realScaleFactors, scaledImages);
}

vector<Mat> scaleImagesOnly(const double samplingRadius, const double minRadius,
                            const double maxRadius, const double numScales,
                            const double blurWidth, const Mat& image) {
  return get<2>(scaleImage(samplingRadius, minRadius, maxRadius, numScales, blurWidth,
                    image));
}

/**
 * Sample a gray pixel from a color image with sub-pixel resolution.
 */
int sampleSubPixelGray(const Mat& image, double x, double y) {
  CV_Assert(image.channels() == 3);

  Mat pixelPatch;
  // This adjustement is necessary to match the behavior of my Scala reference
  // implementation.
  const float adjustedX = x - 0.5;
  const float adjustedY = y - 0.5;
  getRectSubPix(image, Size(1, 1), Point2f(adjustedX, adjustedY), pixelPatch);
  Mat cloned = pixelPatch.clone();
  CV_Assert(cloned.type() == CV_8UC3);
  CV_Assert(cloned.rows == 1);
  CV_Assert(cloned.cols == 1);
  CV_Assert(cloned.channels() == 3);
  const int red = cloned.data[0];
  const int green = cloned.data[1];
  const int blue = cloned.data[2];

//  Mat grayPixelPatch;
//  cvtColor(pixelPatch, grayPixelPatch, CV_BGR2GRAY);
  return (red + green + blue) / 3;
}

/**
 * Get the pixel sampling point for a given KeyPoint at a given scale
 * and angle.
 */
Point2f samplePoint(const double samplingRadius, const int numAngles,
                    const double realScaleFactorX,
                    const double realScaleFactorY, const int angleIndex,
                    const Point2f& keyPoint) {
  // Determines the place of the keypoint in the scaled image.
  const double scaledX = realScaleFactorX * keyPoint.x;
  const double scaledY = realScaleFactorY * keyPoint.y;

  const double angle = (2 * M_PI * angleIndex) / numAngles;
  const double pixelOffsetX = samplingRadius * cos(angle);
  const double pixelOffsetY = samplingRadius * sin(angle);

  const double x = scaledX + pixelOffsetX;
  const double y = scaledY + pixelOffsetY;
  return Point2f(x, y);
}

/**
 * Samples a log polar pattern at each keypoint in the provided image.
 * The sampled pattern is represented as a 2D array of ints, of size numScales
 * by numAngles.
 * It may fail to extract keypoints near borders.
 */
vector<optional<Mat> > rawLogPolarSeqInternal(
    const double minRadius, const double maxRadius, const int numScales,
    const int numAngles, const double blurWidth, const Mat& image,
    const vector<KeyPoint>& keyPoints) {
  // The larger this number, the more accurate the sampling
  // but the larger the largest resized image.
  const double samplingRadius = 4.0;

  // Build the pyramid of scaled images.
  const tuple<vector<double>, vector<tuple<double, double> >, vector<Mat> > scaleData =
      scaleImage(samplingRadius, minRadius, maxRadius, numScales, blurWidth,
                 image);
  const vector<tuple<double, double> >& realScaleFactors = get<1>(scaleData);
  const vector<Mat>& scaledImages = get<2>(scaleData);
  CV_Assert(realScaleFactors.size() == numScales);

  vector<optional<Mat> > descriptors;
  for (vector<KeyPoint>::const_iterator keyPoint = keyPoints.begin();
      keyPoint != keyPoints.end(); ++keyPoint) {
//  BOOST_FOREACH(const KeyPoint keyPoint, keyPoints){

  const double x = keyPoint->pt.x * get<0>(realScaleFactors.at(numScales - 1));
  const double y = keyPoint->pt.y * get<1>(realScaleFactors.at(numScales - 1));
  const int width = scaledImages.at(numScales - 1).cols;
  const int height = scaledImages.at(numScales - 1).rows;

  const bool isInsideBounds = x - epsilon() > samplingRadius &&
  x + epsilon() + samplingRadius < width &&
  y - epsilon() > samplingRadius &&
  y + epsilon() + samplingRadius < height;

  if (!isInsideBounds) {
    descriptors.push_back(optional<Mat>());
  } else {
    Mat matrix = Mat::zeros(numScales, numAngles, CV_8UC1);
    for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
      for (int angleIndex = 0; angleIndex < numAngles; ++angleIndex) {
        const Mat& scaledImage = scaledImages.at(scaleIndex);
        const Point2f point = samplePoint(
            samplingRadius,
            numAngles,
            get<0>(realScaleFactors.at(scaleIndex)),
            get<1>(realScaleFactors.at(scaleIndex)),
            angleIndex,
            keyPoint->pt);

        const int pixel = sampleSubPixelGray(scaledImage, point.x, point.y);

        matrix.at<uint8_t>(scaleIndex, angleIndex) = pixel;
      }
    }
    descriptors.push_back(optional<Mat>(matrix));
  }
}
  CV_Assert(descriptors.size() == keyPoints.size());

  return descriptors;
}

vector<Mat> rawLogPolarSeq(const double minRadius, const double maxRadius,
                           const int numScales, const int numAngles,
                           const double blurWidth, const Mat& image,
                           const vector<KeyPoint>& keyPoints) {
  const vector<optional<Mat> > matOptions = rawLogPolarSeqInternal(minRadius,
                                                                   maxRadius,
                                                                   numScales,
                                                                   numAngles,
                                                                   blurWidth,
                                                                   image,
                                                                   keyPoints);

  vector<Mat> out;
  for (vector<optional<Mat> >::const_iterator matOption = matOptions.begin();
      matOption != matOptions.end(); ++matOption) {
//  BOOST_FOREACH(const optional<Mat> matOption, matOptions){
  const Mat mat = matOption->is_initialized() ? matOption->get() : Mat();
  out.push_back(mat);
}
  return out;
}

}
