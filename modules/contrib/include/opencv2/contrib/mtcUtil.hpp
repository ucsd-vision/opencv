#ifndef __OPENCV_MTCUTIL_HPP__
#define __OPENCV_MTCUTIL_HPP__

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

/////////////////////////////////////////////////////

namespace cv {

using boost::optional;

/**
 * Only nonnegative powers of 2.
 */
bool isPowerOfTwo(const int x);

void assertNear(const double left, const double right);

Mat normalizeL2(const Mat& descriptor);

}

#endif
