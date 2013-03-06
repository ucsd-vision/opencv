//#include "precomp.hpp"
//#include <algorithm>
//#include <vector>
//
//#include <iostream>
//#include <iomanip>
//
//#include "boost/tuple/tuple.hpp"
//#include "boost/optional/optional.hpp"
//#include <boost/foreach.hpp>
//#include <cmath>
//#include <map>
//#include <tuple>
//#include "kiss_fftndr.h"
//
//#include "opencv2/highgui/highgui.hpp"
//
//// Enable asserts for this file.
//#undef NDEBUG
//#include <assert.h>
//#define NDEBUG
//
//using namespace std;
////using namespace boost;
//using boost::optional;
//
/////////////////////////////////////////////////////////////
//
//namespace cv {
//
//const double epsilon = 0.00001;
//
//void assertNear(const double left, const double right) {
//  CV_Assert(std::abs(left - right) < epsilon);
//}
//
///**
// * Get image scaling factors. These are used to create the image pyramid from
// * which descriptors are extracted.
// */
//vector<double> getScaleFactors(const double samplingRadius,
//                               const double minRadius, const double maxRadius,
//                               const int numScales) {
//  const double maxScalingFactor = samplingRadius / minRadius;
//  const double minScalingFactor = samplingRadius / maxRadius;
//  CV_Assert(maxScalingFactor > minScalingFactor);
//  const double base = exp(
//      log(minScalingFactor / maxScalingFactor) / (numScales - 1));
//  CV_Assert(base < 1);
//
//  vector<double> scaleFactors;
//  for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
//    const double scaleFactor = maxScalingFactor * pow(base, scaleIndex);
//    CV_Assert(scaleFactor >= minScalingFactor - epsilon);
//    CV_Assert(scaleFactor <= maxScalingFactor + epsilon);
//    if (scaleIndex == 0)
//      assertNear(scaleFactor, maxScalingFactor);
//    if (scaleIndex == numScales - 1)
//      assertNear(scaleFactor, minScalingFactor);
//    scaleFactors.push_back(scaleFactor);
//  }
//
//  return scaleFactors;
//}
//
///**
// * Not all scaling factors can be perfectly realized, as images must have
// * integer width and height.
// * Returns the realizable (width, height) pairs that most closely match
// * the desired scaling factors.
// */
//vector<tuple<tuple<int, int>, tuple<double, double>>> getRealScaleTargets(
//    const vector<double>& idealScalingFactors, const int imageWidth,
//    const int imageHeight) {
//  vector<tuple<tuple<int, int>, tuple<double, double> > > scaleTargets;
//  BOOST_FOREACH(const double scaleFactor, idealScalingFactors){
//  const int scaledWidth = round(scaleFactor * imageWidth);
//  const int scaledHeight = round(scaleFactor * imageHeight);
//
//  const double realFactorX = static_cast<double>(scaledWidth) / imageWidth;
//  const double realFactorY = static_cast<double>(scaledHeight) / imageHeight;
//
//  scaleTargets.push_back(make_tuple(
//          make_tuple(scaledWidth, scaledHeight),
//          make_tuple(realFactorX, realFactorY)));
//}
//  return scaleTargets;
//}
//
//Mat getRealScaleTargetsMat(const vector<double>& idealScalingFactors,
//                           const int imageWidth, const int imageHeight) {
//  const vector<tuple<tuple<int, int>, tuple<double, double> > > tuples =
//      getRealScaleTargets(idealScalingFactors, imageWidth, imageHeight);
//  Mat out(tuples.size(), 4, CV_64FC1);
//  for (int row = 0; row < tuples.size(); ++row) {
//    const tuple<tuple<int, int>, tuple<double, double> > pair = tuples.at(row);
//    out.at<double>(row, 0) = get<0>(get<0>(pair));
//    out.at<double>(row, 1) = get<1>(get<0>(pair));
//    out.at<double>(row, 2) = get<0>(get<1>(pair));
//    out.at<double>(row, 3) = get<1>(get<1>(pair));
//  }
//  return out;
//}
//
///**
// * Get the image pyramid, as well as the scaling factors that
// * were used to create it. Since the ideal scaling factors cannot be
// * used in all cases (integer rounding), we return the ideal factors
// * plus the actual factors used.
// */
//tuple<vector<double>, vector<tuple<double, double> >, vector<Mat> > scaleImage(
//    const double samplingRadius, const double minRadius, const double maxRadius,
//    const double numScales, const double blurWidth, const Mat& image) {
//  const vector<double> idealScaleFactors = getScaleFactors(samplingRadius,
//                                                           minRadius, maxRadius,
//                                                           numScales);
//  Mat blurred;
////  blur(image, blurred, Size(blurWidth, blurWidth));
//  GaussianBlur(image, blurred, Size(0, 0), blurWidth, 0);
//
//  const vector<tuple<tuple<int, int>, tuple<double, double> > > scaleSizesAndFactors =
//      getRealScaleTargets(idealScaleFactors, blurred.cols, blurred.rows);
//  CV_Assert(idealScaleFactors.size() == scaleSizesAndFactors.size());
//
//  vector<tuple<double, double> > realScaleFactors;
//  vector<Mat> scaledImages;
//  for (int index = 0; index < scaleSizesAndFactors.size(); ++index) {
//    const tuple<double, double> realScaleFactor = get<1>(scaleSizesAndFactors.at(index));
//    realScaleFactors.push_back(realScaleFactor);
//
//    const tuple<int, int> realScaleSize =
//        get<0>(scaleSizesAndFactors.at(index));
//    Mat resized;
//    resize(blurred, resized,
//           Size(get<0>(realScaleSize), get<1>(realScaleSize)), 0, 0,
//           INTER_CUBIC);
//    scaledImages.push_back(resized);
//  }
//  return make_tuple(idealScaleFactors, realScaleFactors, scaledImages);
//}
//
//vector<Mat> scaleImagesOnly(const double samplingRadius, const double minRadius,
//                            const double maxRadius, const double numScales,
//                            const double blurWidth, const Mat& image) {
//  return get<2>(scaleImage(samplingRadius, minRadius, maxRadius, numScales, blurWidth,
//                    image));
//}
//
///**
// * Sample a gray pixel from a color image with sub-pixel resolution.
// */
//int sampleSubPixelGray(const Mat& image, double x, double y) {
//  CV_Assert(image.channels() == 3);
//
//  Mat pixelPatch;
//  // This adjustement is necessary to match the behavior of my Scala reference
//  // implementation.
//  const float adjustedX = x - 0.5;
//  const float adjustedY = y - 0.5;
//  getRectSubPix(image, Size(1, 1), Point2f(adjustedX, adjustedY), pixelPatch);
//  Mat cloned = pixelPatch.clone();
//  CV_Assert(cloned.type() == CV_8UC3);
//  CV_Assert(cloned.rows == 1);
//  CV_Assert(cloned.cols == 1);
//  CV_Assert(cloned.channels() == 3);
//  const int red = cloned.data[0];
//  const int green = cloned.data[1];
//  const int blue = cloned.data[2];
//
////  Mat grayPixelPatch;
////  cvtColor(pixelPatch, grayPixelPatch, CV_BGR2GRAY);
//  return (red + green + blue) / 3;
//}
//
///**
// * Get the pixel sampling point for a given KeyPoint at a given scale
// * and angle.
// */
//Point2f samplePoint(const double samplingRadius, const int numAngles,
//                    const double realScaleFactorX,
//                    const double realScaleFactorY, const int angleIndex,
//                    const Point2f& keyPoint) {
//  // Determines the place of the keypoint in the scaled image.
//  const double scaledX = realScaleFactorX * keyPoint.x;
//  const double scaledY = realScaleFactorY * keyPoint.y;
//
//  const double angle = (2 * M_PI * angleIndex) / numAngles;
//  const double pixelOffsetX = samplingRadius * cos(angle);
//  const double pixelOffsetY = samplingRadius * sin(angle);
//
//  const double x = scaledX + pixelOffsetX;
//  const double y = scaledY + pixelOffsetY;
//  return Point2f(x, y);
//}
//
///**
// * Samples a log polar pattern at each keypoint in the provided image.
// * The sampled pattern is represented as a 2D array of ints, of size numScales
// * by numAngles.
// * It may fail to extract keypoints near borders.
// */
//vector<optional<Mat> > rawLogPolarSeqInternal(
//    const double minRadius, const double maxRadius, const int numScales,
//    const int numAngles, const double blurWidth, const Mat& image,
//    const vector<KeyPoint>& keyPoints) {
//  // The larger this number, the more accurate the sampling
//  // but the larger the largest resized image.
//  const double samplingRadius = 4.0;
//
//  // Build the pyramid of scaled images.
//  const tuple<vector<double>, vector<tuple<double, double> >, vector<Mat> > scaleData =
//      scaleImage(samplingRadius, minRadius, maxRadius, numScales, blurWidth,
//                 image);
//  const vector<tuple<double, double> >& realScaleFactors = get<1>(scaleData);
//  const vector<Mat>& scaledImages = get<2>(scaleData);
//  CV_Assert(realScaleFactors.size() == numScales);
//
//  vector<optional<Mat> > descriptors;
//  for (vector<KeyPoint>::const_iterator keyPoint = keyPoints.begin();
//      keyPoint != keyPoints.end(); ++keyPoint) {
////  BOOST_FOREACH(const KeyPoint keyPoint, keyPoints){
//
//  const double x = keyPoint->pt.x * get<0>(realScaleFactors.at(numScales - 1));
//  const double y = keyPoint->pt.y * get<1>(realScaleFactors.at(numScales - 1));
//  const int width = scaledImages.at(numScales - 1).cols;
//  const int height = scaledImages.at(numScales - 1).rows;
//
//  const bool isInsideBounds = x - epsilon > samplingRadius &&
//  x + epsilon + samplingRadius < width &&
//  y - epsilon > samplingRadius &&
//  y + epsilon + samplingRadius < height;
//
//  if (!isInsideBounds) {
//    descriptors.push_back(optional<Mat>());
//  } else {
//    Mat matrix = Mat::zeros(numScales, numAngles, CV_8UC1);
//    for (int scaleIndex = 0; scaleIndex < numScales; ++scaleIndex) {
//      for (int angleIndex = 0; angleIndex < numAngles; ++angleIndex) {
//        const Mat& scaledImage = scaledImages.at(scaleIndex);
//        const Point2f point = samplePoint(
//            samplingRadius,
//            numAngles,
//            get<0>(realScaleFactors.at(scaleIndex)),
//            get<1>(realScaleFactors.at(scaleIndex)),
//            angleIndex,
//            keyPoint->pt);
//
//        const int pixel = sampleSubPixelGray(scaledImage, point.x, point.y);
//
//        matrix.at<uint8_t>(scaleIndex, angleIndex) = pixel;
//      }
//    }
//    descriptors.push_back(optional<Mat>(matrix));
//  }
//}
//  CV_Assert(descriptors.size() == keyPoints.size());
//
//  return descriptors;
//}
//
//vector<Mat> rawLogPolarSeq(const double minRadius, const double maxRadius,
//                           const int numScales, const int numAngles,
//                           const double blurWidth, const Mat& image,
//                           const vector<KeyPoint>& keyPoints) {
//  const vector<optional<Mat> > matOptions = rawLogPolarSeqInternal(minRadius,
//                                                                   maxRadius,
//                                                                   numScales,
//                                                                   numAngles,
//                                                                   blurWidth,
//                                                                   image,
//                                                                   keyPoints);
//
//  vector<Mat> out;
//  for (vector<optional<Mat> >::const_iterator matOption = matOptions.begin();
//      matOption != matOptions.end(); ++matOption) {
////  BOOST_FOREACH(const optional<Mat> matOption, matOptions){
//  const Mat mat = matOption->is_initialized() ? matOption->get() : Mat();
//  out.push_back(mat);
//}
//  return out;
//}
//
///**
// * Only nonnegative powers of 2.
// */
//bool isPowerOfTwo(const int x) {
//  if (x < 1)
//    return false;
//  else if (x == 1)
//    return true;
//  else
//    return x % 2 == 0 && isPowerOfTwo(x / 2);
//}
//
///**
// * Find the affine pair that normalizes this matrix.
// */
//AffinePair getAffinePair(const Mat& descriptor) {
//  CV_Assert(descriptor.type() == CV_8UC1);
//  CV_Assert(descriptor.total() > 1);
//
//  const double offset = mean(descriptor).val[0];
//  const double scale = norm(descriptor - offset);
//  CV_Assert(scale > 0);
//  return AffinePair(scale, offset);
//}
//
///**
// * Normalize descriptor to have zero mean and unit norm.
// */
//Mat normalizeL2(const Mat& descriptor) {
//  const AffinePair affinePair = getAffinePair(descriptor);
//  return (descriptor - affinePair.offset) / affinePair.scale;
//}
//
///**
// * Get the normalization data for a matrix.
// */
//NormalizationData getNormalizationData(const Mat& descriptor) {
//  CV_Assert(descriptor.type() == CV_8UC1);
//  return NormalizationData(getAffinePair(descriptor),
//                           sum(normalizeL2(descriptor)).val[0],
//                           descriptor.total());
//}
//
////NormalizationData* getNormalizationDataPointer(const Mat& descriptor) {
////  return new NormalizationData(getNormalizationData(descriptor));
////}
//
//void* getNormalizationDataVoidPointer(const Mat& descriptor) {
//  return new NormalizationData(getNormalizationData(descriptor));
//}
//
///**
// * Get the scale map for an entire log-polar pattern.
// */
//ScaleMap<NormalizationData> getScaleMap(const Mat& descriptor) {
//  CV_Assert(descriptor.type() == CV_8UC1);
//
//  CV_Assert(descriptor.rows > 0);
//  CV_Assert(descriptor.cols > 1);
//
//  const int numScales = descriptor.rows;
//
//  map<int, NormalizationData> data;
//  for (int scaleOffset = -numScales + 1; scaleOffset <= numScales - 1;
//      ++scaleOffset) {
//    const int start = max(scaleOffset, 0);
//    const int stop = min(numScales, scaleOffset + numScales);
//
//    const Mat roi = descriptor(Range(start, stop), Range::all());
//    CV_Assert(roi.rows == stop - start);
//    CV_Assert(roi.cols == descriptor.cols);
//
//    getNormalizationData(roi);
//
//    data[scaleOffset] = getNormalizationData(roi);;
//  }
//
//  return ScaleMap<NormalizationData>(data);
//}
//
//int matrixIndex(const int cols, const int row, const int col) {
//  return row * cols + col;
//}
//
//kiss_fft_cpx kiss_fft_cpx_alloc(const kiss_fft_scalar r,
//                                const kiss_fft_scalar i) {
//  kiss_fft_cpx out;
//  out.r = r;
//  out.i = i;
//  return out;
//}
//
////void fft2DR(const int rows, const int cols, const kiss_fft_scalar *spatialData,
////            kiss_fft_cpx *fourierData) {
////  const int ndims = 2;
////  const int dims[2] = { rows, cols };
////
////  const kiss_fftndr_cfg config = kiss_fftndr_alloc(dims, ndims, false, NULL,
////                                                   NULL);
////
////  // The Fourier representation is conjugate symmetric, so kiss_fft only
////  // returns the non-redundant part. Here we add that redundancy back in,
////  // using a bit more memory but making the bookkeeping easier.
////  const int compressedCols = cols / 2 + 1;
////  // The data for a matrix of size rows x compressedCols.
////  vector<kiss_fft_cpx> fourierCompressed(rows * compressedCols);
////  kiss_fftndr(config, spatialData, fourierCompressed.data());
////  free(config);
////
////  // Now copy the data out of the compressed representation and into the
////  // uncompressed representation.
////  for (int row = 0; row < rows; ++row) {
////    for (int compressedCol = 0; compressedCol < compressedCols;
////        ++compressedCol) {
////      const kiss_fft_cpx element = fourierCompressed[matrixIndex(compressedCols,
////                                                                 row,
////                                                                 compressedCol)];
////      // Copy the data on the left side of the matrix.
////      fourierData[matrixIndex(cols, row, compressedCol)] = element;
////
////      // Copy the data on the right side of the matrix. We must first
////      // conjugate it.
////      const kiss_fft_cpx conjugateElement = kiss_fft_cpx_alloc(element.r,
////                                                               -element.i);
////      // This is the symmetry part.
////      const int offset = compressedCols - compressedCol - 1;
////      const int rightCol = compressedCols - 1 + offset;
////      fourierData[matrixIndex(cols, row, rightCol)] = conjugateElement;
////    }
////  }
////}
////
////void ifft2DR(const int rows, const int cols, const kiss_fft_cpx *fourierData,
////             kiss_fft_scalar *spatialData) {
////  const int ndims = 2;
////  const int dims[2] = { rows, cols };
////
////  const kiss_fftndr_cfg config = kiss_fftndr_alloc(dims, ndims, true, NULL,
////                                                   NULL);
////
////  // The Fourier representation is conjugate symmetric, so kiss_fft only
////  // returns the non-redundant part. Here we add that redundancy back in,
////  // using a bit more memory but making the bookkeeping easier.
////  const int compressedCols = cols / 2 + 1;
////  // The data for a matrix of size rows x compressedCols.
////  vector<kiss_fft_cpx> fourierCompressed(rows * compressedCols);
////  for (int row = 0; row < rows; ++row) {
////    for (int compressedCol = 0; compressedCol < compressedCols;
////        ++compressedCol) {
////      fourierCompressed[matrixIndex(compressedCols, row, compressedCol)] =
////          fourierData[matrixIndex(cols, row, compressedCol)];
////    }
////  }
////
////  kiss_fftndri(config, fourierCompressed.data(), spatialData);
////  free(config);
////}
//
/////**
//// * A function used for debugging.
//// *
//// * This cannot be active at the same time as fft2DInteger, because
//// * they depend on different settings of kiss_fft_scalar.
//// */
////Mat fft2DDouble(const Mat& spatialData) {
////  CV_Assert(spatialData.type() == CV_64FC1);
////  CV_Assert(spatialData.channels() == 1);
////  // This part isn't strictly necessary for FFT, but it is for
////  // INCCLogPolar.
////  CV_Assert(spatialData.rows > 0 && isPowerOfTwo(spatialData.rows));
////  CV_Assert(spatialData.cols > 1 && isPowerOfTwo(spatialData.cols));
////
////  Mat fourierData(spatialData.rows, 2 * spatialData.cols, spatialData.type());
////  fft2DR(spatialData.rows, spatialData.cols,
////         reinterpret_cast<kiss_fft_scalar*>(spatialData.data),
////         reinterpret_cast<kiss_fft_cpx*>(fourierData.data));
////
////  cout << fourierData << endl;
////
////  // This clone appears to be necessary to avoid getting a pointer to
////  // uninitialized data.
////  return fourierData.clone();
////}
////
////Mat ifft2DDouble(const Mat& fourierData) {
////  CV_Assert(fourierData.type() == CV_64FC1);
////  CV_Assert(fourierData.channels() == 1);
////  // This part isn't strictly necessary for FFT, but it is for
////  // INCCLogPolar.
////  CV_Assert(fourierData.rows > 0 && isPowerOfTwo(fourierData.rows));
////  CV_Assert(fourierData.cols > 1 && isPowerOfTwo(fourierData.cols));
////
////  Mat spatialData(fourierData.rows, fourierData.cols / 2, fourierData.type());
////  ifft2DR(spatialData.rows, spatialData.cols,
////          reinterpret_cast<kiss_fft_cpx*>(fourierData.data),
////          reinterpret_cast<kiss_fft_scalar*>(spatialData.data));
////
////  // It appears kiss_fft doesn't normalize the fft or ifft.
////  const Mat normalizedSpatialData = spatialData / spatialData.total();
////
////  return normalizedSpatialData.clone();
////}
//
//Mat fft2DDouble(const Mat& spatialData) {
//  CV_Assert(spatialData.type() == CV_64FC1);
//  CV_Assert(spatialData.channels() == 1);
//
//  Mat fourierData;
//  dft(spatialData, fourierData, DFT_COMPLEX_OUTPUT, 0);
//  return fourierData;
//}
//
//Mat ifft2DDouble(const Mat& fourierData) {
//  CV_Assert(fourierData.type() == CV_64FC2);
//  CV_Assert(fourierData.channels() == 2);
//
//  Mat spatialData;
//  idft(fourierData, spatialData, DFT_REAL_OUTPUT | DFT_SCALE, 0);
//  return spatialData;
//}
//
/////**
//// * Performs 2D FFT on an integer Mat, returning an integer Mat.
//// * The returned Mat has the same type as the input, the same number of rows,
//// * but twice the number of columns. This is because each complex value is
//// * reprsented as two adjacent ints. For example, the memory might look like:
//// * [real_0, imag_0, real_1, imag_1, ...].
//// */
////Mat fft2DInteger(const Mat& spatialData) {
////  CV_Assert(spatialData.type() == CV_16SC1);
////  CV_Assert(spatialData.channels() == 1);
////  // This part isn't strictly necessary for FFT, but it is for
////  // INCCLogPolar.
////  CV_Assert(spatialData.rows > 0 && isPowerOfTwo(spatialData.rows));
////  CV_Assert(spatialData.cols > 1 && isPowerOfTwo(spatialData.cols));
////
////  Mat fourierData(spatialData.rows, 2 * spatialData.cols, spatialData.type());
////  fft2DR(spatialData.rows, spatialData.cols,
////         reinterpret_cast<kiss_fft_scalar*>(spatialData.data),
////         reinterpret_cast<kiss_fft_cpx*>(fourierData.data));
////
////  return fourierData.clone();
////}
////
////Mat ifft2DInteger(const Mat& fourierData) {
////  CV_Assert(fourierData.type() == CV_16SC1);
////  CV_Assert(fourierData.channels() == 1);
////  // This part isn't strictly necessary for FFT, but it is for
////  // INCCLogPolar.
////  CV_Assert(fourierData.rows > 0 && isPowerOfTwo(fourierData.rows));
////  CV_Assert(fourierData.cols > 1 && isPowerOfTwo(fourierData.cols));
////
////  Mat spatialData(fourierData.rows, fourierData.cols / 2, fourierData.type());
////  ifft2DR(spatialData.rows, spatialData.cols,
////          reinterpret_cast<kiss_fft_cpx*>(fourierData.data),
////          reinterpret_cast<kiss_fft_scalar*>(spatialData.data));
////
////  return spatialData.clone();
////}
//
///**
// * Get a descriptor from an entire log-polar pattern.
// */
//NCCBlock getNCCBlock(const Mat& samples) {
//  CV_Assert(samples.type() == CV_8UC1);
//
//  // We require the descriptor width and height each be a power of two.
//  CV_Assert(isPowerOfTwo(samples.rows));
//  CV_Assert(samples.cols > 1 && isPowerOfTwo(samples.cols));
//
//  const ScaleMap<NormalizationData> scaleMap = getScaleMap(samples);
//
//  const Mat zeroPadding = Mat::zeros(samples.rows, samples.cols,
//                                     samples.type());
//  Mat padded;
//  vconcat(samples, zeroPadding, padded);
//  // For now, we're working with floating point values.
//  Mat converted;
//  padded.convertTo(converted, CV_64FC1);
//  const Mat fourierData = fft2DDouble(converted);
//
//  return NCCBlock(fourierData, scaleMap);
//}
//
///**
// * Extract descriptors from the given keypoints.
// */
//vector<optional<NCCBlock> > extractInternal(const NCCLogPolarExtractor& self,
//                                            const Mat& image,
//                                            const vector<KeyPoint>& keyPoints) {
//  const vector<optional<Mat> > sampleOptions = rawLogPolarSeqInternal(
//      self.minRadius, self.maxRadius, self.numScales, self.numAngles,
//      self.blurWidth, image, keyPoints);
//  CV_Assert(sampleOptions.size() == keyPoints.size());
//
////  cout << "sampleOptions.size " << sampleOptions.size() << endl;
//
//  vector<optional<NCCBlock> > out;
//  for (vector<optional<Mat> >::const_iterator sampleOption = sampleOptions.begin();
//       sampleOption != sampleOptions.end(); ++sampleOption) {
////  BOOST_FOREACH(const optional<Mat> sampleOption, sampleOptions){
//  if (sampleOption->is_initialized()) {
//    const Mat sample = sampleOption->get();
//    CV_Assert(sample.rows == self.numScales);
//    CV_Assert(sample.cols == self.numAngles);
//    out.push_back(optional<NCCBlock>(getNCCBlock(sample)));
//  } else {
//    out.push_back(optional<NCCBlock>());
//  }
//}
//
//  CV_Assert(out.size() == keyPoints.size());
////  cout << "out.size " << out.size() << endl;
//  return out;
//}
//
////void getAffinePair(AffinePair& pair) {}
//
//void getAffinePair(Ptr<int> pair) {}
//
//Ptr<int> asdf;
//
///**
// * Converts an optional<NCCBlock> to a single-row Mat.
// */
//Mat nccBlockToMat(const optional<NCCBlock>& blockOption) {
//  // The Mat of all zeros represents "not initialized".
//  Mat out(1, sizeof(NCCBlock), CV_8UC1, Scalar(0));
//
//  if (blockOption.is_initialized()) {
////    cout << "block is initialized" << endl;
//    NCCBlock* block = const_cast<NCCBlock*>(&blockOption.get());
//    Mat mat(1, sizeof(NCCBlock), CV_8UC1, reinterpret_cast<uint8_t*>(block));
//    out = mat.clone();
//  }
//
////  cout << out << endl;
//
//  return out;
//}
//
///**
// * Converts several optional<NCCBlock> to a Mat.
// */
//Mat nccBlocksToMat(const vector<optional<NCCBlock> >& blockOptions) {
//  Mat out(blockOptions.size(), sizeof(NCCBlock), CV_8UC1, Scalar(0));
//
//  for (int row = 0; row < blockOptions.size(); ++row) {
//    out.row(row) = nccBlockToMat(blockOptions.at(row));
//  }
//
//  return out;
//}
//
///**
// * Checks whether a Mat represents an initialized optional<NCCBlock>.
// */
//bool matIsInitializedNCCBlock(const Mat& mat) {
//  const bool sizeAndType = mat.rows
//      == 1&& mat.cols == sizeof(NCCBlock) && mat.type() == CV_8UC1;
//
//  double* min;
//  double* max;
//  minMaxIdx(mat, min, max);
//  const bool allZeros = *min == 0 && *max == 0;
//
//  return sizeAndType && !allZeros;
//}
//
//  /**
//   * Converts a single-row Mat to an optional<NCCBlock>
//   */
//optional<NCCBlock> matToNCCBlock(const Mat& mat) {
//  CV_Assert(mat.rows == 1);
//  CV_Assert(mat.cols == sizeof(NCCBlock));
//  CV_Assert(mat.type() == CV_8UC1);
//
//  optional<NCCBlock> out;
//
//  double* min;
//  double* max;
//  minMaxIdx(mat, min, max);
//  if (*min != 0 || *max != 0) {
//    out = *reinterpret_cast<const NCCBlock*>(&mat.at<uint8_t>(0, 0));
//  }
//
//  return out;
//}
//
///**
// * Converts a Mat to several optional<NCCBlock>.
// */
//vector<optional<NCCBlock> > matToNCCBlocks(const Mat& mat) {
//  vector<optional<NCCBlock> > out;
//  for (int row = 0; row < mat.rows; ++row) {
//    out.push_back(matToNCCBlock(mat.row(row)));
//  }
//  return out;
//}
//
//Mat extract(const double minRadius, const double maxRadius, const int numScales,
//            const int numAngles, const double blurWidth, const Mat& image,
//            const vector<KeyPoint>& keyPoints) {
//  const vector<optional<NCCBlock> > blockOptions = extractInternal(
//      NCCLogPolarExtractor(minRadius, maxRadius, numScales, numAngles,
//                           blurWidth),
//      image, keyPoints);
//  CV_Assert(blockOptions.size() == keyPoints.size());
//
////  cout << "blockOptions.size " << blockOptions.size() << endl;
//  const Mat blockMat = nccBlocksToMat(blockOptions);
//  CV_Assert(blockMat.rows == keyPoints.size());
//  CV_Assert(blockMat.cols == sizeof(NCCBlock));
////  cout << "blockMat.rows " << blockMat.rows << endl;
////  cout << "blockMat.cols " << blockMat.cols << endl;
//  return blockMat.clone();
//}
//
////optional<NCCBlock> extractSingle(const NCCLogPolarExtractor& self,
////                                 const Mat& image, const KeyPoint& keyPoint) {
////  vector<KeyPoint> keyPoints;
////  keyPoints.push_back(keyPoint);
////  const vector<optional<NCCBlock> > blockOptions = extractInternal(self, image,
////                                                                   keyPoints);
////
////  return blockOptions.at(0);
////}
//
///**
// * Determine what the dot product would have been had the vectors been
// * normalized first.
// */
//double nccFromUnnormalized(const NormalizationData& leftData,
//                           const NormalizationData& rightData,
//                           const double unnormalizedInnerProduct) {
//  CV_Assert(leftData.size == rightData.size);
//
//  // Suppose we observe the inner product between two vectors
//  // (a_x * x + b_x) and (a_y * y + b_y), where x and y are normalized.
//  // Note (a_x * x + b_x)^T (a_y * y + b_y) is
//  // (a_x * x)^T (a_y * y) + a_y * b_x^T y + a_x * b_y^T x + b_x^T b_y.
//  // Thus we can solve for the normalized dot product:
//  // x^T y = ((a_x * x)^T (a_y * y) - a_y * b_x^T y - a_x * b_y^T x - b_x^T b_y) / (a_x * a_y).
//  const double aybxy = rightData.affinePair.scale * leftData.affinePair.offset
//      * rightData.elementSum;
//
//  const double axbyx = leftData.affinePair.scale * rightData.affinePair.offset
//      * leftData.elementSum;
//
//  const double bxby = leftData.size * leftData.affinePair.offset
//      * rightData.affinePair.offset;
//
//  const double numerator = unnormalizedInnerProduct - aybxy - axbyx - bxby;
//  const double denominator = leftData.affinePair.scale
//      * rightData.affinePair.scale;
//  CV_Assert(denominator != 0);
//
//  const double correlation = numerator / denominator;
//  cout << correlation << endl;
//  CV_Assert(correlation <= 1 + epsilon);
//  CV_Assert(correlation >= -1 - epsilon);
//  return correlation;
//}
//
///**
// * Performs correlation (not convolution) between two signals, assuming
// * they were originally purely real and the have already been mapped
// * into Fourier space.
// */
//Mat correlationFromPreprocessed(const Mat& left, const Mat& right) {
//  CV_Assert(left.type() == CV_64FC2);
//  CV_Assert(left.channels() == 2);
//
//  CV_Assert(left.rows == right.rows);
//  CV_Assert(left.cols == right.cols);
//  CV_Assert(left.channels() == right.channels());
//  CV_Assert(left.type() == right.type());
//
//  vector<Mat> leftLayers;
//  split(left, leftLayers);
//  CV_Assert(leftLayers.size() == 2);
//  const auto& leftReal = leftLayers.at(0);
//  const auto& leftImaginary = leftLayers.at(1);
//
//  vector<Mat> rightLayers;
//  split(right, rightLayers);
//  CV_Assert(rightLayers.size() == 2);
//  const auto& rightReal = rightLayers.at(0);
//  const auto& rightImaginary = rightLayers.at(1);
//
//  // Now we do pairwise multiplication of the _conjugate_ of the left
//  // matrix and the right matrix.
//  const auto realPart =
//      leftReal.mul(rightReal) + leftImaginary.mul(rightImaginary);
//  const auto imaginaryPart =
//      leftReal.mul(rightImaginary) - leftImaginary.mul(rightReal);
//
//  vector<Mat> dotTimesLayers = {realPart, imaginaryPart};
//  Mat dotTimes;
//
//
//
//  merge(dotTimesLayers, dotTimes);
//
////  // The complex conjugate of the left Mat.
////  Mat leftConjugate(left.rows, left.cols, left.type());
////  for (int row = 0; row < left.rows; ++row) {
////    // The input matrices store complex values, with the odd number indices
////    // storing the imaginary parts. So we start at 1 and stride through by 2.
////    for (int col = 1; col < left.cols; col += 2) {
////      leftConjugate.at<double>(row, col) = -left.at<double>(row, col);
////    }
////  }
////
////  // Now we do a pairwise multiplication of complex values.
////  Mat dotTimes;
////  for (int row = 0; row < left.rows; ++row) {
////    for (int col = 0; col < left.cols; col += 2) {
////      const double leftReal = left.at<double>(row, col);
////      const double leftImaginary = left.at<double>(row, col + 1);
////      const double rightReal = right.at<double>(row, col);
////      const double rightImaginary = right.at<double>(row, col + 1);
////
////      const double productReal = leftReal * rightReal
////          - leftImaginary * rightImaginary;
////      const double productImaginary = leftReal * rightImaginary
////          + leftImaginary * rightReal;
////      dotTimes.at<double>(row, col) = productReal;
////      dotTimes.at<double>(row, col + 1) = productImaginary;
////    }
////  }
//
//  return ifft2DDouble(dotTimes);
//}
//
///**
// * The correct implementation of a % b.
// */
//int mod(const int a, const int b) {
//  if (a >= 0)
//    return a % b;
//  else
//    return (b + (a % b)) % b;
//}
//
///**
// * The map of normalized correlations.
// */
//Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
//                   const NCCBlock& rightBlock) {
//  CV_Assert(leftBlock.fourierData.rows == rightBlock.fourierData.rows);
//  CV_Assert(leftBlock.fourierData.cols == rightBlock.fourierData.cols);
//  // The data has been zero padded in the vertical direction, which is
//  // why we're dividing by 2 here.
//  CV_Assert(scaleSearchRadius < leftBlock.fourierData.rows / 2);
//
//  cout << leftBlock.fourierData.rows << endl;
//  cout << leftBlock.fourierData.cols << endl;
//  cout << leftBlock.fourierData.channels() << endl;
//
//  // This is real valued.
//  const Mat correlation = correlationFromPreprocessed(rightBlock.fourierData,
//                                                      leftBlock.fourierData);
//  CV_Assert(correlation.type() == CV_64FC1);
//
//  cout << correlation << endl;
//
//  Mat normalized(correlation);
//  for (int scaleOffset = -scaleSearchRadius; scaleOffset <= scaleSearchRadius;
//      ++scaleOffset) {
//    const int rowIndex = mod(scaleOffset, leftBlock.fourierData.rows);
//    for (int col = 0; col < correlation.cols; ++col) {
//      cout << scaleOffset << endl;
//      const double dotProduct = correlation.at<double>(rowIndex, col);
//      cout << dotProduct << endl;
//      const double normalized = nccFromUnnormalized(
//          leftBlock.scaleMap.data.at(scaleOffset),
//          rightBlock.scaleMap.data.at(-scaleOffset), dotProduct);
//    }
//  }
//
//  cout << normalized << endl;
//
//  return normalized;
//}
//
///**
// * Assuming the dot product is between two unit length vectors, find
// * their l2 distance.
// * Divides by sqrt(2) to undo a previous normalization.
// */
//double dotProductToL2Distance(const double dotProduct) {
//  return sqrt(2 - 2 * dotProduct) / sqrt(2);
//}
//
///**
// * The map of distances.
// */
//Mat responseMapToDistanceMap(const Mat& responseMap) {
//  CV_Assert(responseMap.type() == CV_64FC1);
//
//  Mat distances(responseMap.size(), responseMap.type());
//
//  MatConstIterator_<double> response = responseMap.begin<double>();
//  MatIterator_<double> distance = distances.begin<double>();
//  for (; response != responseMap.end<double>(); ++response, ++distance) {
//    *distance = dotProductToL2Distance(*response);
//  }
//  return distances;
//}
//
//Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
//                   const NCCBlock& right) {
//  const Mat responseMap = getResponseMap(self.scaleSearchRadius, left, right);
//  return responseMapToDistanceMap(responseMap);
//}
//
///**
// * The distance between two descriptors.
// */
//double distanceInternal(const NCCLogPolarMatcher& self, const NCCBlock& left,
//                        const NCCBlock& right) {
//  const Mat distanceMap = getDistanceMap(self, left, right);
//  return *min_element(distanceMap.begin<double>(), distanceMap.end<double>());
//}
//
////double distance(const int scaleSearchRadius, const Mat& leftBlock,
////                const Mat& rightBlock) {
////  return distanceInternal(NCCLogPolarMatcher(scaleSearchRadius),
////                          matToNCCBlock(leftBlock).get(),
////                          matToNCCBlock(rightBlock).get());
////}
//
///**
// * Match all descriptors on the left to all descriptors on the right,
// * return distances in a Mat where row indexes left and col indexes right.
// * Distances are -1 where invalid.
// * If the distance is symmetric, there is redundancy across the diagonal.
// */
//Mat matchAllPairs(const int scaleSearchRadius, const Mat& leftBlocks,
//                  const Mat& rightBlocks) {
//  const NCCLogPolarMatcher matcher(scaleSearchRadius);
//
//  const vector<optional<NCCBlock> > lefts = matToNCCBlocks(leftBlocks);
//  const vector<optional<NCCBlock> > rights = matToNCCBlocks(rightBlocks);
//
//  Mat distances(leftBlocks.rows, rightBlocks.cols, CV_64FC1, Scalar(-1));
//  for (int row = 0; row < distances.rows; ++row) {
//    for (int col = 0; col < distances.cols; ++col) {
//      const optional<NCCBlock>& left = lefts.at(row);
//      const optional<NCCBlock>& right = rights.at(col);
//
//      if (left.is_initialized() && right.is_initialized()) {
//        distances.at<double>(row, col) = distanceInternal(matcher, left.get(),
//                                                          right.get());
//      }
//    }
//  }
//  return distances;
//}
//
/////**
//// * For debugging.
//// */
////Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
////                                const int numScales, const int numAngles,
////                                const double blurWidth,
////                                const int scaleSearchRadius, const Mat& image,
////                                const KeyPoint& left, const KeyPoint& right) {
////  const NCCLogPolarExtractor extractor(minRadius, maxRadius, numScales,
////                                       numAngles, blurWidth);
////
////  const optional<NCCBlock> leftDescriptorOption = extractSingle(extractor,
////                                                                image, left);
////  const optional<NCCBlock> rightDescriptorOption = extractSingle(extractor,
////                                                                 image, right);
////
////  if (!leftDescriptorOption.is_initialized()
////      || !rightDescriptorOption.is_initialized()) {
////    return Mat();
////  } else {
////    const NCCBlock leftDescriptor = leftDescriptorOption.get();
////    const NCCBlock rightDescriptor = rightDescriptorOption.get();
////
////    const NCCLogPolarMatcher matcher(scaleSearchRadius);
////
////    return getDistanceMap(matcher, leftDescriptor, rightDescriptor);
////  }
////}
//
//}
