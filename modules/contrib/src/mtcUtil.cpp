#include "opencv2/contrib/mtcUtil.hpp"

////////////////////////////////////////////

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

const double epsilon = 0.00001;

void assertNear(const double left, const double right) {
  CV_Assert(std::abs(left - right) < epsilon);
}

/**
 * Normalize descriptor to have zero mean and unit norm.
 */
Mat normalizeL2(const Mat& descriptor) {
  const AffinePair affinePair = getAffinePair(descriptor);
  return (descriptor - affinePair.offset) / affinePair.scale;
}
