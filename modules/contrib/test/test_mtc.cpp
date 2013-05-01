#include "test_precomp.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace std;

TEST(Contrib_MTC, CrashTest) {
  const auto image = imread("/wg/u/echristiansen/Dropbox/goldfish_girl.bmp");
  ASSERT_FALSE(image.empty());

  // You should detect a bunch of keypoints here.
  const KeyPoint keyPoint(50, 50, 0);
  const vector<KeyPoint> keyPoints { keyPoint };

  // These functions defined in mtcExtractor.* and mtcMatcher.*

  // Log-polar grid params and blur.
  const double minRadius = 4;
  const double maxRadius = 32;
  const int numScales = 8;
  const int numAngles = 16;
  const double blurWidth = 1.2;
  const auto extractor = NCCLogPolarExtractor(
		  minRadius, maxRadius, numScales, numAngles, blurWidth);

  // These are optional values. Basically either a descriptor or null.
  // Used to model the fact not all keypoints can become descriptors.
  const VectorOptionNCCBlock descriptorOptions = extract(extractor,
          image,
          keyPoints);

  // This unboxes the descriptors, and throws away the nulls.
  const VectorNCCBlock descriptors = flatten(descriptorOptions);

  // Controls the amount of overlap in the cylinder alignment process.
  const int scaleSearchRadius = 4;
  const auto matcher = NCCLogPolarMatcher(scaleSearchRadius);

  // This matches the descriptors to themselves.
  // Distances is a distance matrix.
  const Mat distances = matchAllPairs(matcher, descriptors, descriptors);
}
