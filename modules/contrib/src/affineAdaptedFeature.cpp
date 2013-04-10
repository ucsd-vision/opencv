#include "opencv2/contrib/affineAdaptedFeature.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

namespace cv {

void detectAndExtractDescriptorsASIFT(const Mat& image,
		vector<KeyPoint>& keyPoints, Mat& descriptors) {
//	const SiftFeatureDetector siftDetector;
//	const SiftDescriptorExtractor siftExtractor;

	const Ptr<FeatureDetector> siftDetector(new SiftFeatureDetector());
	const Ptr<DescriptorExtractor> siftExtractor(new SiftDescriptorExtractor());
	//	const Ptr<FeatureDetector> siftDetector = FeatureDetector::create(
	//			FeatureDetector::SIFT);
	//	const Ptr<FeatureExtractor> siftExtractor = FeatureExtractor::create(
	//			FeatureExtractor::SIFT);
//	const Ptr<Feature2D> sift(new SIFT());

//    const AffineAdaptedFeature2D asift(sift);
	const AffineAdaptedFeature2D asift(siftDetector, siftExtractor);
    asift(image, noArray, keyPoints, descriptors);
}

}
