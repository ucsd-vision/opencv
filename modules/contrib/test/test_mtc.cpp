#include "test_precomp.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace std;

TEST(Contrib_MTC, CrashTest) {
  cout << "here" << endl;

  Mat samples(32, 32, CV_8U);
  RNG().fill(samples, RNG::UNIFORM, NULL, NULL);

//  cout << samples << endl;

  getScaleMap(samples);

  getNCCBlock(samples);

//
//  const auto image = imread("/wg/u/echristiansen/Dropbox/goldfish_girl.bmp");
//  ASSERT_FALSE(image.empty());
//
//  const KeyPoint keyPoint(50, 50, 0);
//  const vector<KeyPoint> keyPoints { keyPoint };
//
//  const auto sampleOptions = rawLogPolarSeq(2, 32, 8, 16, 2, image, keyPoints);
//  ASSERT_TRUE(sampleOptions.at(0).rows > 0);
//  const auto sample = sampleOptions.at(0);
//
////  cout << sample << endl;
//
//  const auto nData = getNormalizationData(sample);
//
//  const auto scaleMap = getScaleMap(sample);
//
//  const auto nccBlock = getNCCBlock(sample);
//
////  cout << nccBlock.fourierData << endl;
//
//  const auto extractor = NCCLogPolarExtractor(2, 32, 2, 4, 2);
//
//  const vector<optional<NCCBlock>> descriptors = extractInternal(extractor,
//                                                                 image,
//                                                                 keyPoints);
//
//  const NCCBlock descriptor = descriptors.at(0).get();
//
//  const auto matcher = NCCLogPolarMatcher(2);
//
//  const auto distanceMap = getResponseMap(
//      0, descriptor, descriptor);
//
////  const auto d = distanceInternal(
////      matcher, descriptor, descriptor);
////
////  cout << d << endl;
}
