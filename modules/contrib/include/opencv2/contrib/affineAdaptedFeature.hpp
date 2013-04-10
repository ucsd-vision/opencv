#ifndef __OPENCV_AFFINEADAPTEDFEATURE_HPP__
#define __OPENCV_AFFINEADAPTEDFEATURE_HPP__

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

using namespace std;

CV_WRAP void detectAndExtractDescriptorsASIFT(
		const Mat& image,
		vector<KeyPoint>& keyPoints,
		Mat& descriptors);

}

#endif
