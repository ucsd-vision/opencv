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

///////////////////////////////

/**
 * Like Scala's Option:
 * http://www.scala-lang.org/api/current/index.html#scala.Option
 * Implemented as a vector, which is wasteful but convenient.
 */
template<typename A>
using Option = vector<A>;

template<typename A>
Option<A> Some(const A& a) {
  const vector<A> option = { a };
  return option;
}

template<typename A>
Option<A> None() {
  const vector<A> option;
  return option;
}

template<typename A>
bool isDefined(const Option<A>& option) {
  // The empty list is the empty option, and the list with one element
  // is the defined option.
  CV_DbgAssert(option.size() == 0 || option.size() == 1);
  return option.size() == 1;
}

template<typename A>
const A& get(const Option<A>& option) {
  CV_DbgAssert(isDefined(option));
  return option.at(0);
}

//////////////////////////////////////////////////////

/**
 * Only nonnegative powers of 2.
 */
bool isPowerOfTwo(const int x);

int mod(const int a, const int b);

double epsilon();

void assertNear(const double left, const double right);

CV_EXPORTS_W void fft2D(const Mat& spatialData, Mat & fourierData);

CV_EXPORTS_W void ifft2D(const Mat& fourierData, Mat & spatialData);
}

#endif
