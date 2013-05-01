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
  CV_DbgAssert(std::abs(left - right) < epsilon());
}

void fft2D(const Mat& spatialData, cv::Mat &fourierData) {
  CV_DbgAssert(spatialData.type() == CV_32FC1 || spatialData.type() == CV_64FC1);

  dft(spatialData, fourierData, DFT_COMPLEX_OUTPUT, 0);
}

void ifft2D(const Mat& fourierData, cv::Mat &spatialData) {
  CV_DbgAssert(fourierData.type() == CV_32FC2 || fourierData.type() == CV_64FC2);

  idft(fourierData, spatialData, DFT_REAL_OUTPUT | DFT_SCALE, 0);
}

}
