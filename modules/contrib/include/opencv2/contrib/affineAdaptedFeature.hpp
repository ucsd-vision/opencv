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

class AffineAdaptedFeature2D: public cv::Feature2D {
public:
	AffineAdaptedFeature2D(const cv::Ptr<cv::Feature2D>& feature2d);
	AffineAdaptedFeature2D(const cv::Ptr<cv::FeatureDetector>& featureDetector,
			const cv::Ptr<cv::DescriptorExtractor>& descriptorExtractor);

	virtual int descriptorSize() const;
	virtual int descriptorType() const;

	virtual void operator()(cv::InputArray image, cv::InputArray mask,
			std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
			bool useProvidedKeypoints = false) const;

protected:
	void initialize();

	void detectAndComputeImpl(const cv::Mat& image, const cv::Mat& mask,
			std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

	void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
			cv::Mat& descriptors) const;
	void detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
			const cv::Mat& mask = cv::Mat()) const;

	cv::Ptr<cv::Feature2D> feature2d;
	cv::Ptr<cv::FeatureDetector> featureDetector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

	std::vector<cv::Vec2f> affineTransformParams;
};

CV_WRAP Mat detectAndExtractDescriptorsASIFT(
		const Mat& image,
		vector<KeyPoint>& keyPoints,
		Mat& descriptors);

}

#endif
