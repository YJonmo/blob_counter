#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "thirdparty/ndarray_converter.h"

namespace py = pybind11;


class CppBlobCounter {
    float multiplier;
private:
  cv::Mat image;
  cv::Mat image_with_kps;
  std::vector<cv::KeyPoint> kps;
  std::string image_path;
  // Set up SimpleBlobDetector parameters
  cv::SimpleBlobDetector::Params params;

public:
  // constructor
  CppBlobCounter(std::string image_path_init, bool filterByArea_init=false, 
            bool filterByConvexity_init=false, bool filterByInertia_init=false,
            bool filterByCircularity_init=false, int minArea_init=500){
    image_path = image_path_init;
    params.filterByArea = filterByArea_init;
    params.filterByConvexity = filterByConvexity_init;
    params.filterByInertia = filterByInertia_init;
    params.filterByCircularity = filterByCircularity_init;
    params.minArea = minArea_init;

    // Change parameters as needed
    params.minThreshold = 50;
    params.maxThreshold = 250;
    params.minCircularity = 0.8;
    params.minConvexity = 0.87;
    params.minInertiaRatio = 0.01;
  }  

  cv::Mat read_image() {
    cv::Mat image_temp = imread(image_path, cv::IMREAD_GRAYSCALE );
    image_temp = cv::Scalar(255, 255, 255) - image_temp;
    cv::copyMakeBorder(image_temp, image, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    return image;
  }

  cv::Mat detectBlob() {
    this->read_image();
    // Create the detector with the specified parameters
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    // Detect blobs
    detector->detect(image, kps);
    std::cout << "number of blobs is " << std::to_string(size(kps)) << std::endl;

    cv::drawKeypoints(image, kps, image_with_kps, cv::Scalar(255,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::putText(image_with_kps, std::to_string(size(kps)) + " blobs", cv::Point(40, 40), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.3,
            CV_RGB(255, 255, 0), //font color
            2);
    return image_with_kps;
  }

};

