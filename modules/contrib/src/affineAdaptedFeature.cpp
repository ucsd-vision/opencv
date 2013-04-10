#include "opencv2/contrib/affineAdaptedFeature.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static Mat_<float> affineSkew(const Mat& image, const Mat& mask, float tilt,
		float phi, Mat& transformedImage, Mat& transformedMask) {
	transformedImage = image.clone();
	transformedMask = mask.clone();

	Mat_<float> A(2, 3);
	A << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f;

	CV_Assert(tilt > 1.f);

	// rotate image
	if (phi != 0.f) {
		const float sinPhi = std::sin(phi / 180.f * CV_PI);
		const float cosPhi = std::cos(phi / 180.f * CV_PI);

		Mat_<float> R(2, 2);
		R << cosPhi, -sinPhi, sinPhi, cosPhi;

		Mat_<float> rectPoints(4, 2);
		rectPoints << 0.f, 0.f, 0.f, image.rows, image.cols, image.rows, image.cols, 0.f;
		Mat transformedRectPoints = R * rectPoints.t();
		transformedRectPoints = transformedRectPoints.t();
		transformedRectPoints = transformedRectPoints.reshape(2, 1);

		Rect r = boundingRect(transformedRectPoints);

		A << cosPhi, -sinPhi, -r.x, sinPhi, cosPhi, -r.y;

		Mat dst;
		warpAffine(transformedImage, dst, A, r.size(), INTER_LINEAR,
				BORDER_REPLICATE);
		transformedImage = dst;
	}

	// blur image
	const float sigmaX = 0.8f * sqrt(tilt * tilt - 1.f);
	const float sigmaY = 0.01f;
	Mat bluredImage;
	GaussianBlur(transformedImage, bluredImage, Size(0, 0), sigmaX, sigmaY);

	// tilt image
	resize(bluredImage, transformedImage, Size(0, 0), 1.f / tilt, 1.f,
			INTER_NEAREST);
	A.row(0) /= tilt;

	// transform mask if it's need
	if (phi != 0.f || !transformedMask.empty()) {
		if (transformedMask.empty())
			transformedMask = Mat(image.size(), CV_8UC1, Scalar::all(255));

		Mat dst;
		warpAffine(transformedMask, dst, A, transformedImage.size(),
				INTER_NEAREST);
		transformedMask = dst;
	}

	Mat Ainv;
	invertAffineTransform(A, Ainv);

	return Ainv;
}

namespace cv {

AffineAdaptedFeature2D::AffineAdaptedFeature2D(const Ptr<Feature2D>& feature2d) :
		feature2d(feature2d) {
	initialize();
}

AffineAdaptedFeature2D::AffineAdaptedFeature2D(
		const Ptr<FeatureDetector>& featureDetector,
		const Ptr<DescriptorExtractor>& descriptorExtractor) :
		featureDetector(featureDetector), descriptorExtractor(
				descriptorExtractor) {
	initialize();
}

int AffineAdaptedFeature2D::descriptorSize() const {
	if (feature2d)
		return feature2d->descriptorSize();

	CV_Assert(descriptorExtractor);
	return descriptorExtractor->descriptorSize();
}

int AffineAdaptedFeature2D::descriptorType() const {
	if (feature2d)
		return feature2d->descriptorType();

	CV_Assert(descriptorExtractor);
	return descriptorExtractor->descriptorType();
}

//void AffineAdaptedFeature2D::detectAndComputeImpl(const Mat& image,
//		const Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors) const {
//	if (feature2d)
//		(*feature2d)(image, mask, keypoints, descriptors);
//	else {
//		CV_Assert(featureDetector);
//		CV_Assert(descriptorExtractor);
//
//		featureDetector->detect(image, keypoints, mask);
//		descriptorExtractor->compute(image, keypoints, descriptors);
//	}
//}

void AffineAdaptedFeature2D::detectAndComputeImpl(const Mat& image,
		const Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors) const {
	if (feature2d) {
		CV_Assert(false);
		(*feature2d)(image, mask, keypoints, descriptors);
	} else {
		CV_Assert(featureDetector);
		CV_Assert(descriptorExtractor);

		CV_Assert(keypoints.size() > 0);
//		featureDetector->detect(image, keypoints, mask);
		descriptorExtractor->compute(image, keypoints, descriptors);
	}
}

void AffineAdaptedFeature2D::initialize() {
	// Geberate affine transformation parameters

	affineTransformParams.clear();

	const float a = sqrt(2);
	const float b = 72.f;

	// 0 - tilt
	// 1 - phi
	float tilt = 1.f;
	const int tiltN = 5;
	for (int deg = 0; deg <= tiltN; deg++) {
		if (tilt == 1.f) {
			affineTransformParams.push_back(Vec2f(tilt, FLT_MAX));
		} else {
			float phi = 0.f;
			const int phiN = static_cast<int>(std::floor(180.f * tilt / b));
			for (int step = 0; step <= phiN; step++) {
				affineTransformParams.push_back(Vec2f(tilt, phi));
				phi += b / tilt;
			}
		}
		tilt *= a;
	}
}

//void AffineAdaptedFeature2D::operator()(InputArray _image, InputArray _mask,
//		vector<KeyPoint>& _keypoints, OutputArray _descriptors,
//		bool useProvidedKeypoints) const {
//	Mat image = _image.getMat();
//	Mat mask = _mask.getMat();
//
//	CV_Assert(useProvidedKeypoints == false);
//	CV_Assert(!affineTransformParams.empty());
//
//	vector < vector<KeyPoint> > keypoints(affineTransformParams.size());
//	vector<Mat> descriptors(affineTransformParams.size());
//
////#pragma omp parallel for
//	for (size_t paramsIndex = 0; paramsIndex < affineTransformParams.size();
//			paramsIndex++) {
//		const Vec2f& params = affineTransformParams[paramsIndex];
//		const float tilt = params[0];
//		const float phi = params[1];
//
//		if (tilt == 1.f) // tilt
//				{
//			detectAndComputeImpl(image, mask, keypoints[paramsIndex],
//					descriptors[paramsIndex]);
//		} else {
//			Mat transformedImage, transformedMask;
//			Mat Ainv = affineSkew(image, mask, tilt, phi, transformedImage,
//					transformedMask);
//
//			detectAndComputeImpl(transformedImage, transformedMask,
//					keypoints[paramsIndex], descriptors[paramsIndex]);
//
//			// correct keypoints coordinates
//			CV_Assert(Ainv.type() == CV_32FC1);
//			const float* Ainv_ptr = Ainv.ptr<const float>();
//			for (size_t kpIndex = 0; kpIndex < keypoints[paramsIndex].size();
//					kpIndex++) {
//				KeyPoint& kp = keypoints[paramsIndex][kpIndex];
//				float tx = Ainv_ptr[0] * kp.pt.x + Ainv_ptr[1] * kp.pt.y
//						+ Ainv_ptr[2];
//				float ty = Ainv_ptr[3] * kp.pt.x + Ainv_ptr[4] * kp.pt.y
//						+ Ainv_ptr[5];
//				kp.pt.x = tx;
//				kp.pt.y = ty;
//			}
//		}
//	}
//
//	// copy keypoints and descriptors to the output
//	_keypoints.clear();
//	Mat allDescriptors;
//	for (size_t paramsIndex = 0; paramsIndex < affineTransformParams.size();
//			paramsIndex++) {
//		_keypoints.insert(_keypoints.end(), keypoints[paramsIndex].begin(),
//				keypoints[paramsIndex].end());
//		allDescriptors.push_back(descriptors[paramsIndex]);
//	}
//
//	_descriptors.create(allDescriptors.size(), allDescriptors.type());
//	Mat _descriptorsMat = _descriptors.getMat();
//	allDescriptors.copyTo(_descriptorsMat);
//}

void AffineAdaptedFeature2D::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& _keypoints, OutputArray _descriptors,
		bool useProvidedKeypoints) const {
	Mat image = _image.getMat();
	Mat mask = _mask.getMat();

	CV_Assert(useProvidedKeypoints == true);
	CV_Assert(!affineTransformParams.empty());
	CV_Assert(_keypoints.size() == 1);
	const auto kp = _keypoints.at(0);

//	vector < vector<KeyPoint> > keypoints(affineTransformParams.size());
	// Each Mat will be single-row.
	vector<Mat> descriptors(affineTransformParams.size());

//#pragma omp parallel for
	for (size_t paramsIndex = 0; paramsIndex < affineTransformParams.size();
			paramsIndex++) {
		const Vec2f& params = affineTransformParams[paramsIndex];
		const float tilt = params[0];
		const float phi = params[1];

		if (tilt == 1.f) // tilt
				{
			detectAndComputeImpl(image, mask, _keypoints,
					descriptors.at(paramsIndex));

			cout << "untilted keypoint is " << kp.pt.x << " " << kp.pt.y << " " << kp.pt.size << " " << endl;
		} else {
			Mat transformedImage, transformedMask;
			Mat Ainv = affineSkew(image, mask, tilt, phi, transformedImage,
					transformedMask);

			Mat A;
			invertAffineTransform(Ainv, A);
			CV_Assert(A.rows == 2);
			CV_Assert(A.cols == 3);

			cout << A << endl;

			CV_Assert(A.type() == CV_32FC1);
			const float* A_ptr = A.ptr<const float>();
			float tx = A_ptr[0] * kp.pt.x + A_ptr[1] * kp.pt.y
					+ A_ptr[2];
			float ty = A_ptr[3] * kp.pt.x + A_ptr[4] * kp.pt.y
					+ A_ptr[5];

			const KeyPoint newKeyPoint(tx, ty, -1);
			vector<KeyPoint> newKeyPoints = { newKeyPoint };
			CV_Assert(newKeyPoints.size() == 1);

			cout << "tilted image size is " << transformedImage.rows << " " << transformedImage.cols << endl;
			cout << "tilted keypoint is " << newKeyPoint.pt.x << " " << newKeyPoint.pt.y << endl;


			detectAndComputeImpl(transformedImage, transformedMask,
					newKeyPoints, descriptors.at(paramsIndex));

			cout << "tilted is empty " << descriptors.at(paramsIndex).empty() << endl;
		}
	}

  vector<double> allDescriptorData;
  for (const Mat descriptor : descriptors) {
	  if (!descriptor.empty()) {
		  CV_Assert(descriptor.rows == 1);
		  CV_Assert(descriptor.cols == 128);

		  for (int col = 0; col < 128; ++col) {
			  CV_Assert(descriptor.type() == CV_32F);
			  allDescriptorData.push_back(descriptor.at<float>(0, col));
		  }
	  }
  }

  CV_Assert(allDescriptorData.size() % 128 == 0);

  Mat allDescriptors(allDescriptorData.size() / 128, 128, CV_64F, &allDescriptorData[0]);
  allDescriptors.copyTo(_descriptors);
}

void AffineAdaptedFeature2D::computeImpl(const Mat& /*image*/,
		vector<KeyPoint>& /*keypoints*/, Mat& /*descriptors*/) const {
	CV_Error(CV_StsNotImplemented,
			"Not implemented method because it's not efficient to split feature detection and description extraction here\n");
}

void AffineAdaptedFeature2D::detectImpl(const Mat& /*image*/,
		vector<KeyPoint>& /*keypoints*/, const Mat& /*mask*/) const {
	CV_Error(CV_StsNotImplemented,
			"Not implemented method because it's not efficient to split feature detection and description extraction here\n");
}

void detectAndExtractDescriptorsASIFT(const Mat& image,
		const vector<KeyPoint>& keyPoints, Mat& descriptors) {
    CV_Assert(keyPoints.size() == 1);

	vector<KeyPoint> keyPoints_copy = keyPoints;

	const Ptr<FeatureDetector> siftDetector(new SiftFeatureDetector());
	const Ptr<DescriptorExtractor> siftExtractor(new SiftDescriptorExtractor());

	const AffineAdaptedFeature2D asift(siftDetector, siftExtractor);
	asift(image, Mat(), keyPoints_copy, descriptors, true);
}

}
