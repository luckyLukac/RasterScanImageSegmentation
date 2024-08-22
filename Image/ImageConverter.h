#pragma once
#include <QImage>
#include <vector>

#include "RawImage.h"


/// <summary>
/// Image converter between different image formats.
/// </summary>
namespace ImageConverter {
    /// <summary>
    /// Convertion of QImage to RawImage.
    /// </summary>
    /// <param name="image">QImage</param>
    /// <returns>Raw image</returns>
    std::vector<std::vector<int>> QImageToRaw(const QImage& image);

    /// <summary>
    /// Convertion of RawImage to QImage.
    /// </summary>
    /// <param name="rawImage">RawImage</param>
    /// <returns>QImage</returns>
    QImage rawImageToQImage(const RawImage& rawImage);

    /// <summary>
    /// Convertion of cv::Mat to RawImage.
    /// </summary>
    /// <param name="image">cv::Mat</param>
    /// <returns>Raw image</returns>
    std::vector<std::vector<int>> cvToRaw(const cv::Mat& image);

    /// <summary>
    /// Convertion of RawImage to cv::Mat.
    /// </summary>
    /// <param name="rawImage">RawImage</param>
    /// <returns>cv::Mat</returns>
    cv::Mat rawImageToCv(const RawImage& rawImage);
};