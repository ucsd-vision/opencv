/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_HIGHGUI_HPP__
#define __OPENCV_HIGHGUI_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui_c.h"

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/**
 * Added by Eric Christiansen.
 */
#include "opencv2/features2d/features2d.hpp"
#include <algorithm>
#include <vector>

#include <iostream>
#include <iomanip>

#include "boost/tuple/tuple.hpp"
#include "boost/optional/optional.hpp"
#include <boost/foreach.hpp>
#include <cmath>

//using namespace boost;
using boost::optional;
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

#ifdef __cplusplus

struct CvCapture;
struct CvVideoWriter;

namespace cv
{

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/**
 * Added by Eric Christiansen.
 */
/**
 * Only nonnegative powers of 2.
 */
bool isPowerOfTwo(const int x);

/**
 * The two values that characterize a 1D affine function.
 */
struct CV_EXPORTS_W AffinePair {
  // Stupid fucking assignment operator. How do I make this const?
  CV_WRAP double scale;
  CV_WRAP double offset;

  AffinePair() {}

  AffinePair(const double scale_, const double offset_)
      : scale(scale_),
        offset(offset_) {
  }
};

//CV_EXPORTS_W void getAffinePair(AffinePair& pair);

//CV_EXPORTS_W void getAffinePair(Ptr<int> pair);

CV_EXPORTS_W AffinePair* getAffinePair();

/**
 * Data needed to determine normalized dot product from dot product
 * of unnormalized vectors.
 */
struct CV_EXPORTS_W NormalizationData {
  CV_WRAP AffinePair affinePair;
  // This is the sum of the elements of the normalized vector.
  CV_WRAP double elementSum;
  CV_WRAP int size;

  NormalizationData() {}

  NormalizationData(const AffinePair& affinePair_, const double elementSum_,
                    const int size_)
      : affinePair(affinePair_),
        elementSum(elementSum_),
        size(size_) {
  }
};

/**
 * A mapping from scale levels. Scale levels must be a sequential and
 * symmetric about zero.
 */
template<class A>
struct ScaleMap {
  std::map<int, A> data;

  ScaleMap() {}

  ScaleMap(const std::map<int, A>& data_)
      : data(data_) {
    vector<int> keys;

    // BOOST_FOREACH can't handle this.
    // "typename" added as magic at compiler's suggestion.
    for (typename std::map<int, A>::const_iterator keyValue = data.begin();
        keyValue != data.end(); ++keyValue) {
      keys.push_back(keyValue->first);
    }

    // Now keys is sorted.
    sort(keys.begin(), keys.end());
    const int minKey = keys.at(0);
    const int maxKey = keys.at(keys.size() - 1);

    CV_Assert(-minKey == maxKey);
    for (int index = 0; index < keys.size(); ++index) {
      CV_Assert(keys.at(index) == index + minKey);
    }
  }
};

/**
 * The descriptor. Contains a Fourier-space version of the log polar
 * data as well as normalization data for each scale.
 */
struct CV_EXPORTS_W NCCBlock {
  CV_WRAP Mat fourierData;
  CV_WRAP ScaleMap<NormalizationData> scaleMap;

  NCCBlock() {}

  NCCBlock(const Mat& fourierData_,
           const ScaleMap<NormalizationData>& scaleMap_)
      : fourierData(fourierData_),
        scaleMap(scaleMap_) {
    CV_Assert(fourierData.rows - 1 == scaleMap.data.size());
  }
};

/**
 * The extractor.
 * numScales and numAngles must be powers of 2.
 * numAngles must be >= 2.
 */
struct CV_EXPORTS_W NCCLogPolarExtractor {
  CV_WRAP double minRadius;
  CV_WRAP double maxRadius;
  CV_WRAP int numScales;
  CV_WRAP int numAngles;
  CV_WRAP double blurWidth;

  NCCLogPolarExtractor() {}

  NCCLogPolarExtractor(const double minRadius_, const double maxRadius_,
                       const int numScales_, const int numAngles_,
                       const double blurWidth_)
      : minRadius(minRadius_),
        maxRadius(maxRadius_),
        numScales(numScales_),
        numAngles(numAngles_),
        blurWidth(blurWidth_) {
    CV_Assert(isPowerOfTwo(numScales));
    CV_Assert(numAngles > 1 && isPowerOfTwo(numAngles));
  }
};

struct CV_EXPORTS_W NCCLogPolarMatcher {
  CV_WRAP int scaleSearchRadius;

  NCCLogPolarMatcher() {}

  NCCLogPolarMatcher(const int scaleSearchRadius_)
      : scaleSearchRadius(scaleSearchRadius_) {
    CV_Assert(scaleSearchRadius >= 0);
  }
};

CV_EXPORTS_W vector<double> getScaleFactors(const double samplingRadius,
                               const double minRadius, const double maxRadius,
                               const int numScales);

CV_EXPORTS_W Mat getRealScaleTargetsMat(const vector<double>& idealScalingFactors, const int imageWidth,
                                        const int imageHeight);

CV_EXPORTS_W vector<Mat> scaleImagesOnly(const double samplingRadius, const double minRadius,
                            const double maxRadius, const double numScales,
                            const double blurWidth, const Mat& image);

CV_EXPORTS_W int sampleSubPixelGray(const Mat& image, double x, double y);

CV_EXPORTS_W Point2f samplePoint(const double samplingRadius,
                                 const int numAngles,
                                 const double realScaleFactorX,
                                 const double realScaleFactorY,
                                 const int angleIndex, const Point2f& keyPoint);

NormalizationData getNormalizationData(const Mat& descriptor);

//CV_EXPORTS_W NormalizationData* getNormalizationDataPointer(const Mat& descriptor);

CV_EXPORTS_W void* getNormalizationDataVoidPointer(const Mat& descriptor);

ScaleMap<NormalizationData> getScaleMap(const Mat& descriptor);

NCCBlock getNCCBlock(const Mat& samples);

vector<optional<NCCBlock> > extractInternal(const NCCLogPolarExtractor& self,
                                            const Mat& image,
                                            const vector<KeyPoint>& keyPoints);

CV_EXPORTS_W vector<Mat> rawLogPolarSeq(
    const double minRadius, const double maxRadius, const int numScales,
    const int numAngles, const double blurWidth, const Mat& image,
    const vector<KeyPoint>& keyPoints);

CV_EXPORTS_W Mat fft2DDouble(const Mat& spatialData);

CV_EXPORTS_W Mat ifft2DDouble(const Mat& fourierData);

//CV_EXPORTS_W Mat fft2DInteger(const Mat& spatialData);
//
//CV_EXPORTS_W Mat ifft2DInteger(const Mat& fourierData);

CV_EXPORTS_W Mat extract(const double minRadius, const double maxRadius, const int numScales,
            const int numAngles, const double blurWidth, const Mat& image,
            const vector<KeyPoint>& keyPoints);

Mat getResponseMap(const int scaleSearchRadius, const NCCBlock& leftBlock,
                   const NCCBlock& rightBlock);

Mat responseMapToDistanceMap(const Mat& responseMap);

Mat getDistanceMap(const NCCLogPolarMatcher& self, const NCCBlock& left,
                   const NCCBlock& right);

CV_EXPORTS_W Mat matchAllPairs(const int scaleSearchRadius, const Mat& leftBlocks,
                  const Mat& rightBlocks);

double distanceInternal(const NCCLogPolarMatcher& self, const NCCBlock& left,
                        const NCCBlock& right);

//CV_EXPORTS_W Mat distanceMapBetweenKeyPoints(const double minRadius, const double maxRadius,
//                                const int numScales, const int numAngles,
//                                const double blurWidth,
//                                const int scaleSearchRadius, const Mat& image,
//                                const KeyPoint& left, const KeyPoint& right);

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

enum {
    // Flags for namedWindow
    WINDOW_NORMAL   = CV_WINDOW_NORMAL,   // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
    WINDOW_AUTOSIZE = CV_WINDOW_AUTOSIZE, // the user cannot resize the window, the size is constrainted by the image displayed
    WINDOW_OPENGL   = CV_WINDOW_OPENGL,   // window with opengl support

    // Flags for set / getWindowProperty
    WND_PROP_FULLSCREEN   = CV_WND_PROP_FULLSCREEN,  // fullscreen property
    WND_PROP_AUTOSIZE     = CV_WND_PROP_AUTOSIZE,    // autosize property
    WND_PROP_ASPECT_RATIO = CV_WND_PROP_ASPECTRATIO, // window's aspect ration
    WND_PROP_OPENGL       = CV_WND_PROP_OPENGL       // opengl support
};

CV_EXPORTS_W void namedWindow(const string& winname, int flags = WINDOW_AUTOSIZE);
CV_EXPORTS_W void destroyWindow(const string& winname);
CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

CV_EXPORTS_W int waitKey(int delay = 0);

CV_EXPORTS_W void imshow(const string& winname, InputArray mat);

CV_EXPORTS_W void resizeWindow(const string& winname, int width, int height);
CV_EXPORTS_W void moveWindow(const string& winname, int x, int y);

CV_EXPORTS_W void setWindowProperty(const string& winname, int prop_id, double prop_value);//YV
CV_EXPORTS_W double getWindowProperty(const string& winname, int prop_id);//YV

enum
{
    EVENT_MOUSEMOVE      =0,
    EVENT_LBUTTONDOWN    =1,
    EVENT_RBUTTONDOWN    =2,
    EVENT_MBUTTONDOWN    =3,
    EVENT_LBUTTONUP      =4,
    EVENT_RBUTTONUP      =5,
    EVENT_MBUTTONUP      =6,
    EVENT_LBUTTONDBLCLK  =7,
    EVENT_RBUTTONDBLCLK  =8,
    EVENT_MBUTTONDBLCLK  =9
};

enum
{
    EVENT_FLAG_LBUTTON   =1,
    EVENT_FLAG_RBUTTON   =2,
    EVENT_FLAG_MBUTTON   =4,
    EVENT_FLAG_CTRLKEY   =8,
    EVENT_FLAG_SHIFTKEY  =16,
    EVENT_FLAG_ALTKEY    =32
};

typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);

//! assigns callback for mouse events
CV_EXPORTS void setMouseCallback(const string& winname, MouseCallback onMouse, void* userdata = 0);


typedef void (CV_CDECL *TrackbarCallback)(int pos, void* userdata);

CV_EXPORTS int createTrackbar(const string& trackbarname, const string& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

CV_EXPORTS_W int getTrackbarPos(const string& trackbarname, const string& winname);
CV_EXPORTS_W void setTrackbarPos(const string& trackbarname, const string& winname, int pos);

// OpenGL support

typedef void (*OpenGlDrawCallback)(void* userdata);
CV_EXPORTS void setOpenGlDrawCallback(const string& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

CV_EXPORTS void setOpenGlContext(const string& winname);

CV_EXPORTS void updateWindow(const string& winname);

//Only for Qt

CV_EXPORTS CvFont fontQt(const string& nameFont, int pointSize=-1,
                         Scalar color=Scalar::all(0), int weight=CV_FONT_NORMAL,
                         int style=CV_STYLE_NORMAL, int spacing=0);
CV_EXPORTS void addText( const Mat& img, const string& text, Point org, CvFont font);

CV_EXPORTS void displayOverlay(const string& winname, const string& text, int delayms CV_DEFAULT(0));
CV_EXPORTS void displayStatusBar(const string& winname, const string& text, int delayms CV_DEFAULT(0));

CV_EXPORTS void saveWindowParameters(const string& windowName);
CV_EXPORTS void loadWindowParameters(const string& windowName);
CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);
CV_EXPORTS  void stopLoop();

typedef void (CV_CDECL *ButtonCallback)(int state, void* userdata);
CV_EXPORTS int createButton( const string& bar_name, ButtonCallback on_change,
                             void* userdata=NULL, int type=CV_PUSH_BUTTON,
                             bool initial_button_state=0);

//-------------------------

enum
{
    // 8bit, color or not
    IMREAD_UNCHANGED  =-1,
    // 8bit, gray
    IMREAD_GRAYSCALE  =0,
    // ?, color
    IMREAD_COLOR      =1,
    // any depth, ?
    IMREAD_ANYDEPTH   =2,
    // ?, any color
    IMREAD_ANYCOLOR   =4
};

enum
{
    IMWRITE_JPEG_QUALITY =1,
    IMWRITE_PNG_COMPRESSION =16,
    IMWRITE_PNG_STRATEGY =17,
    IMWRITE_PNG_BILEVEL =18,
    IMWRITE_PNG_STRATEGY_DEFAULT =0,
    IMWRITE_PNG_STRATEGY_FILTERED =1,
    IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY =2,
    IMWRITE_PNG_STRATEGY_RLE =3,
    IMWRITE_PNG_STRATEGY_FIXED =4,
    IMWRITE_PXM_BINARY =32
};

CV_EXPORTS_W Mat imread( const string& filename, int flags=1 );
CV_EXPORTS_W bool imwrite( const string& filename, InputArray img,
              const vector<int>& params=vector<int>());
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );
CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst );
CV_EXPORTS_W bool imencode( const string& ext, InputArray img,
                            CV_OUT vector<uchar>& buf,
                            const vector<int>& params=vector<int>());

#ifndef CV_NO_VIDEO_CAPTURE_CPP_API

template<> void CV_EXPORTS Ptr<CvCapture>::delete_obj();
template<> void CV_EXPORTS Ptr<CvVideoWriter>::delete_obj();

class CV_EXPORTS_W VideoCapture
{
public:
    CV_WRAP VideoCapture();
    CV_WRAP VideoCapture(const string& filename);
    CV_WRAP VideoCapture(int device);

    virtual ~VideoCapture();
    CV_WRAP virtual bool open(const string& filename);
    CV_WRAP virtual bool open(int device);
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();

    CV_WRAP virtual bool grab();
    CV_WRAP virtual bool retrieve(CV_OUT Mat& image, int channel=0);
    virtual VideoCapture& operator >> (CV_OUT Mat& image);
    CV_WRAP virtual bool read(CV_OUT Mat& image);

    CV_WRAP virtual bool set(int propId, double value);
    CV_WRAP virtual double get(int propId);

protected:
    Ptr<CvCapture> cap;
};


class CV_EXPORTS_W VideoWriter
{
public:
    CV_WRAP VideoWriter();
    CV_WRAP VideoWriter(const string& filename, int fourcc, double fps,
                Size frameSize, bool isColor=true);

    virtual ~VideoWriter();
    CV_WRAP virtual bool open(const string& filename, int fourcc, double fps,
                      Size frameSize, bool isColor=true);
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();
    virtual VideoWriter& operator << (const Mat& image);
    CV_WRAP virtual void write(const Mat& image);

protected:
    Ptr<CvVideoWriter> writer;
};

#endif

}

#endif

#endif
