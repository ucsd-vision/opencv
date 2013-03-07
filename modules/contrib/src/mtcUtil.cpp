#include "opencv2/contrib/mtcUtil.hpp"

////////////////////////////////////////////

namespace cv {

using namespace std;

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
 * The correct implementation of a % b.
 */
int mod(const int a, const int b) {
  if (a >= 0)
    return a % b;
  else
    return (b + (a % b)) % b;
}

double epsilon() { return 0.00001; }

void assertNear(const double left, const double right) {
  CV_Assert(std::abs(left - right) < epsilon());
}

Mat fft2DDouble(const Mat& spatialData) {
  CV_Assert(spatialData.type() == CV_64FC1);
  CV_Assert(spatialData.channels() == 1);

  Mat fourierData;
  dft(spatialData, fourierData, DFT_COMPLEX_OUTPUT, 0);
  return fourierData;
}

Mat ifft2DDouble(const Mat& fourierData) {
  CV_Assert(fourierData.type() == CV_64FC2);
  CV_Assert(fourierData.channels() == 2);

  Mat spatialData;
  idft(fourierData, spatialData, DFT_REAL_OUTPUT | DFT_SCALE, 0);
  return spatialData;
}

/**
 * Assuming the dot product is between two unit length vectors, find
 * their l2 distance.
 * Divides by sqrt(2) to undo a previous normalization.
 */
double dotProductToL2Distance(const double dotProduct) {
  return sqrt(2 - 2 * dotProduct) / sqrt(2);
}

}
