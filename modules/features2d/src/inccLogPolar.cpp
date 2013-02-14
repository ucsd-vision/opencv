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
#include "kiss_fftndr.h"

using namespace std;
using namespace boost;

///////////////////////////////////////////////////////////

namespace cv {

const double epsilon = 0.00001;

void assertNear(const double left, const double right) {
  assert(std::abs(left - right) < epsilon);
}

/**
 * Get image scaling factors. These are used to create the image pyramid from
 * which descriptors are extracted.
 */
vector<double> getScaleFactors(const double samplingRadius,
                               const double minRadius, const double maxRadius,
                               const int numScales) {
  const double maxScalingFactor = samplingRadius / minRadius;
  const double minScalingFactor = samplingRadius / maxRadius;
  assert(maxScalingFactor > minScalingFactor);
  const double base = exp(
      log(minScalingFactor / maxScalingFactor) / (numScales - 1));
  assert(base < 1);

  vector<double> scaleFactors;
  for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
    const double scaleFactor = maxScalingFactor * pow(base, scaleIndex);
    assert(scaleFactor >= minScalingFactor - epsilon);
    assert(scaleFactor <= maxScalingFactor + epsilon);
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
vector<tuple<tuple<int, int>, tuple<double, double> > > getRealScaleTargets(
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
    out.at<double>(row, 0) = pair.get<0>().get<0>();
    out.at<double>(row, 1) = pair.get<0>().get<1>();
    out.at<double>(row, 2) = pair.get<1>().get<0>();
    out.at<double>(row, 3) = pair.get<1>().get<1>();
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
  assert(idealScaleFactors.size() == scaleSizesAndFactors.size());

  vector<tuple<double, double> > realScaleFactors;
  vector<Mat> scaledImages;
  for (int index = 0; index < scaleSizesAndFactors.size(); ++index) {
    const tuple<double, double> realScaleFactor = scaleSizesAndFactors.at(index)
        .get<1>();
    realScaleFactors.push_back(realScaleFactor);

    const tuple<int, int> realScaleSize =
        scaleSizesAndFactors.at(index).get<0>();
    Mat resized;
    resize(blurred, resized,
           Size(realScaleSize.get<0>(), realScaleSize.get<1>()), 0, 0,
           INTER_CUBIC);
    scaledImages.push_back(resized);
  }
  return make_tuple(idealScaleFactors, realScaleFactors, scaledImages);
}

vector<Mat> scaleImagesOnly(const double samplingRadius, const double minRadius,
                            const double maxRadius, const double numScales,
                            const double blurWidth, const Mat& image) {
  return scaleImage(samplingRadius, minRadius, maxRadius, numScales, blurWidth,
                    image).get<2>();
}

/**
 * Sample a gray pixel from a color image with sub-pixel resolution.
 */
int sampleSubPixelGray(const Mat& image, double x, double y) {
  assert(image.channels() == 3);

  Mat pixelPatch;
  // This adjustement is necessary to match the behavior of my Scala reference
  // implementation.
  const float adjustedX = x - 0.5;
  const float adjustedY = y - 0.5;
  getRectSubPix(image, Size(1, 1), Point2f(adjustedX, adjustedY), pixelPatch);
  Mat cloned = pixelPatch.clone();
  assert(cloned.type() == CV_8UC3);
  const int numElements = cloned.total();
  assert(numElements == 3);
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
  const vector<tuple<double, double> >& realScaleFactors = scaleData.get<1>();
  const vector<Mat>& scaledImages = scaleData.get<2>();
  assert(realScaleFactors.size() == numScales);

  vector<optional<Mat> > descriptors;
  BOOST_FOREACH(const KeyPoint keyPoint, keyPoints){

  const double x = keyPoint.pt.x * realScaleFactors.at(numScales - 1).get<0>();
  const double y = keyPoint.pt.y * realScaleFactors.at(numScales - 1).get<1>();
  const int width = scaledImages.at(numScales - 1).cols;
  const int height = scaledImages.at(numScales - 1).rows;

  const bool isInsideBounds = x - epsilon > samplingRadius &&
  x + epsilon + samplingRadius < width &&
  y - epsilon > samplingRadius &&
  y + epsilon + samplingRadius < height;

  if (!isInsideBounds)
  descriptors.push_back(optional<Mat>());
  else {
    Mat matrix = Mat::zeros(numScales, numAngles, CV_8UC1);
    for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
      for (int angleIndex = 0; angleIndex < numAngles; ++angleIndex) {
        const Mat& scaledImage = scaledImages.at(scaleIndex);
        const Point2f point = samplePoint(
            samplingRadius,
            numAngles,
            realScaleFactors.at(scaleIndex).get<0>(),
            realScaleFactors.at(scaleIndex).get<1>(),
            angleIndex,
            keyPoint.pt);

        const int pixel = sampleSubPixelGray(scaledImage, point.x, point.y);

        matrix.at<uint8_t>(scaleIndex, angleIndex) = pixel;
      }
    }
    descriptors.push_back(optional<Mat>(matrix));
  }
}

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
  BOOST_FOREACH(const optional<Mat> matOption, matOptions){
  const Mat mat = matOption.is_initialized() ? matOption.get() : Mat();
  out.push_back(mat);
}
  return out;
}

/**
 * The two values that characterize a 1D affine function.
 */
struct AffinePair {
  // Stupid fucking assignment operator. How do I make this const?
  double scale;
  double offset;

  AffinePair(const double scale_, const double offset_)
      : scale(scale_),
        offset(offset_) {
  }
};

/**
 * Data needed to determine normalized dot product from dot product
 * of unnormalized vectors.
 */
struct NormalizationData {
  AffinePair affinePair;
  // This is the sum of the elements of the normalized vector.
  double elementSum;
  int size;

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
  const map<int, A> data;

  ScaleMap(const map<int, A>& data_)
      : data(data_) {
    vector<int> keys;

    // BOOST_FOREACH can't handle this.
    // "typename" added as magic at compiler's suggestion.
    for (typename map<int, A>::const_iterator keyValue = data.begin();
        keyValue != data.end(); ++keyValue) {
      keys.push_back(keyValue->first);
    }

    // Now keys is sorted.
    sort(keys.begin(), keys.end());
    const int minKey = keys.at(0);
    const int maxKey = keys.at(keys.size() - 1);

    assert(-minKey == maxKey);
    for (int index = 0; index < keys.size(); ++index) {
      assert(keys.at(index) == index + minKey);
    }
  }
};

/**
 * The descriptor. Contains a Fourier-space version of the log polar
 * data as well as normalization data for each scale.
 */
struct INCCBlock {
  const Mat fourierData;
  const ScaleMap<NormalizationData> scaleMap;

  INCCBlock(const Mat& fourierData_,
            const ScaleMap<NormalizationData>& scaleMap_)
      : fourierData(fourierData_),
        scaleMap(scaleMap_) {
    assert(fourierData.rows - 1 == scaleMap.data.size());
  }
};

/**
 * Only nonnegative powers of 2.
 */
bool isPowerOfTwo(const int x) {
  if (x < 1)
    return false;
  else if (x == 1)
    return true;
  else
    return x % 2 == 0 && isPowerOfTwo(x / 2);
}

/**
 * Find the affine pair that normalizes this matrix.
 */
AffinePair getAffinePair(const Mat& descriptor) {
  assert(descriptor.type() == CV_8UC1);
  assert(descriptor.total() > 1);

  const double offset = mean(descriptor).val[0];
  const double scale = norm(descriptor - offset);
  assert(scale > 0);
  return AffinePair(scale, offset);
}

/**
 * Normalize descriptor to have zero mean and unit norm.
 */
Mat normalizeL2(const Mat& descriptor) {
  const AffinePair affinePair = getAffinePair(descriptor);
  return (descriptor - affinePair.offset) / affinePair.scale;
}

/**
 * Get the normalization data for a matrix.
 */
NormalizationData getNormalizationData(const Mat& descriptor) {
  assert(descriptor.type() == CV_8UC1);
  return NormalizationData(getAffinePair(descriptor),
                           sum(normalizeL2(descriptor)).val[0],
                           descriptor.total());
}

/**
 * Get the scale map for an entire log-polar pattern.
 */
ScaleMap<NormalizationData> getScaleMap(const Mat& descriptor) {
  assert(descriptor.type() == CV_8UC1);

  assert(descriptor.rows > 0);
  assert(descriptor.cols > 1);

  const int numScales = descriptor.rows;

  map<int, NormalizationData> data;
  for (int scaleOffset = -numScales + 1; scaleOffset <= numScales - 1;
      ++scaleOffset) {
    const int start = max(scaleOffset, 0);
    const int stop = min(numScales, scaleOffset + numScales);

    const Mat roi = descriptor(Range(start, stop), Range::all());
    assert(roi.rows == stop - start);
    assert(roi.cols == descriptor.cols);

    getNormalizationData(roi);

    data.at(scaleOffset) = getNormalizationData(roi);
  }

  return ScaleMap<NormalizationData>(data);
}

/**
 * Convert a Mat of type uint8_t to type int16_t.
 */
Mat uint8ToInt16(const Mat& in) {
  assert(in.type() == CV_8UC1);

  Mat out(in.rows, in.cols, CV_16SC1);
  for (int row = 0; row < in.rows; ++row) {
    for (int col = 0; col < in.cols; ++col) {
      out.at<int16_t>(row, col) = in.at<uint8_t>(row, col);
    }
  }
  return out;
}

/**
 * A function used for debugging.
 *
 * This cannot be active at the same time as fft2DInteger, because
 * they depend on different settings of kiss_fft_scalar.
 */
Mat fft2DDouble(const Mat& spatialData) {
  assert(spatialData.type() == CV_64FC1);
  assert(spatialData.channels() == 1);
  // This part isn't strictly necessary for FFT, but it is for
  // INCCLogPolar.
  assert(spatialData.rows > 1 && isPowerOfTwo(spatialData.rows));
  assert(spatialData.cols > 1 && isPowerOfTwo(spatialData.cols));

  const int dims[2] = { spatialData.rows, spatialData.cols };
  const kiss_fftndr_cfg config = kiss_fftndr_alloc(dims, 2, false, NULL, NULL);

  // The Fourier representation is double wide to hold complex values.
  Mat fourierData(spatialData.rows, 2 * spatialData.cols, spatialData.type());
  kiss_fftndr(config, reinterpret_cast<kiss_fft_scalar*>(spatialData.data),
              reinterpret_cast<kiss_fft_cpx*>(fourierData.data));
  free(config);
  return fourierData;
}

///**
// * Performs 2D FFT on an integer Mat, returning an integer Mat.
// * The returned Mat has the same type as the input, the same number of rows,
// * but twice the number of columns. This is because each complex value is
// * reprsented as two adjacent ints. For example, the memory might look like:
// * [real_0, imag_0, real_1, imag_1, ...].
// */
//Mat fft2DInteger(const Mat& spatialData) {
//  assert(spatialData.type() == CV_16SC1);
//  assert(spatialData.channels() == 1);
//  // This part isn't strictly necessary for FFT, but it is for
//  // INCCLogPolar.
//  assert(spatialData.rows > 1 && isPowerOfTwo(spatialData.rows));
//  assert(spatialData.cols > 1 && isPowerOfTwo(spatialData.cols));
//
//  const int dims[2] = { spatialData.rows, spatialData.cols };
//  const kiss_fftndr_cfg config = kiss_fftndr_alloc(dims, 2, false, NULL, NULL);
//
//  // The Fourier representation is double wide to hold complex values.
//  Mat fourierData(spatialData.rows, 2 * spatialData.cols, spatialData.type());
//  kiss_fftndr(config, reinterpret_cast<kiss_fft_scalar*>(spatialData.data),
//              reinterpret_cast<kiss_fft_cpx*>(fourierData.data));
//  free(config);
//  return fourierData;
//}

//Mat ifft2DInteger(const Mat& fourierData) {
//  assert(fourierData.type() == CV_16SC1);
//  assert(fourierData.channels() == 1);
//  // This part isn't strictly necessary for FFT, but it is for
//  // INCCLogPolar.
//  assert(fourierData.rows > 1 && isPowerOfTwo(fourierData.rows));
//  assert(fourierData.cols > 1 && isPowerOfTwo(fourierData.cols));
//
//  const int dims[2] = { fourierData.rows, fourierData.cols };
//  const kiss_fftndr_cfg config = kiss_fftndr_alloc(dims, 2, false, NULL, NULL);
//
//  // The Fourier representation is double wide to hold complex values.
//  Mat fourierData(spatialData.rows, 2 * spatialData.cols, spatialData.type());
//  kiss_fftndr(config, reinterpret_cast<kiss_fft_scalar*>(spatialData.data),
//              reinterpret_cast<kiss_fft_cpx*>(fourierData.data));
//  return fourierData;
//}

///**
// * Get a descriptor from an entire log-polar pattern.
// */
//INCCBlock getNCCBlock(const Mat& samples) {
//  assert(samples.type() == CV_8UC1);
//
//  // We require the descriptor width and height each be a power of two.
//  assert(isPowerOfTwo(samples.rows));
//  assert(samples.cols > 1 && isPowerOfTwo(samples.cols));
//
//  const ScaleMap<NormalizationData> scaleMap = getScaleMap(samples);
//
//  val fourierData = {
//    val zeroPadding = DenseMatrix.zeros[Int](
//      samples.rows,
//      samples.cols)
//
//    val padded = DenseMatrix.vertcat(samples, zeroPadding)
//
//    FFT.fft2(padded mapValues (r => Complex(r, 0)))
//  }
//
//  NCCBlock(fourierData, scaleMap)
//}

/**
 * The extractor.
 * numScales and numAngles must be powers of 2.
 * numAngles must be >= 2.
 */
struct INCCLogPolarExtractor {
  const double minRadius;
  const double maxRadius;
  const int numScales;
  const int numAngles;
  const double blurWidth;

  INCCLogPolarExtractor(const double minRadius_, const double maxRadius_,
                        const int numScales_, const int numAngles_,
                        const double blurWidth_)
      : minRadius(minRadius_),
        maxRadius(maxRadius_),
        numScales(numScales_),
        numAngles(numAngles_),
        blurWidth(blurWidth_) {
    assert(isPowerOfTwo(numScales));
    assert(numAngles > 1 && isPowerOfTwo(numAngles));
  }
};

}
