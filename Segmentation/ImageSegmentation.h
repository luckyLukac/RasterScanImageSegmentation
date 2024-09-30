#pragma once
#include <string>

#include "../Image/Pixel.hpp"
#include "../Image/RawImage.h"


/// <summary>
/// Namespace for image segmentation using different methods.
/// </summary>
namespace ImageSegmentation {
    enum class Distance {
        onePixel,
        twoPixel,
        segmentAverage
    };


    /// <summary>
    /// Extraction of separate segments from an image representing a segmentation mask using the flood fill algorithm.
    /// </summary>
    /// <param name="segmentationMask">Segmentation mask</param>
    /// <returns>Vector of segments with their belonging pixels</returns>
    std::vector<std::vector<Pixel>> extractSegmentsFromSegmentationMask(const cv::Mat& segmentationMask);

    /// <summary>
    /// Dividing an image into separate segments using the K-means algorithm.
    /// </summary>
    /// <param name="image">Image to be segmentized</param>
    /// <param name="K">K in K-means</param>
    /// <returns>Vector of segments with their belonging pixels, segment image</returns>
    std::pair<std::vector<std::vector<Pixel>>, RawImage> segmentizeImageWithKMeans(const cv::Mat& image, const int K = 5);

    /// <summary>
    /// Dividing an image into separate segments using the DBSCAN algorithm.
    /// </summary>
    /// <param name="image">Image to be segmentized</param>
    /// <param name="epsilon">Size of the point neighborhood </param>
    /// <param name="minimumNumberOfNeighbourhoodPoints">Minimum number of points required to form a dense region</param>
    /// <returns>Vector of segments with their belonging pixels, segment image</returns>
    std::pair<std::vector<std::vector<Pixel>>, QImage> segmentizeImageWithDBSCAN(cv::Mat& image, const double epsilon = 5.0, const int minimumNumberOfNeighbourhoodPoints = 25, const int seed = 0);

    /// <summary>
    /// Dividing an image into separate segments using the sweep-line approach.
    /// </summary>
    /// <param name="image">Image to be segmentized</param>
    /// <param name="threshold">Maximum allowed difference of connected pixels that belong to the same segment</param>
    /// <returns>Vector of segments with their belonging pixels, segment image</returns>
    std::pair<std::vector<std::vector<Pixel>>, QImage> segmentizeImageWithSweepLine(cv::Mat& image, const double threshold, const int maxDeviation, const Distance& distance, const int seed);
};
