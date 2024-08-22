#include <chrono>
#include <stack>

#include "../Image/ImageConverter.h"
#include "ImageSegmentation.h"


std::vector<std::vector<Pixel>> ImageSegmentation::extractSegmentsFromSegmentationMask(const cv::Mat& segmentationMask) {
    cv::Mat cvSegmentationMask;
    segmentationMask.convertTo(cvSegmentationMask, CV_32S);
    
    std::vector<std::vector<Pixel>> segments;
    RawImage floodFillMask(cvSegmentationMask.cols, cvSegmentationMask.rows);  // Mask that indicates if a pixel has been already reached by flood fill (indicated with 1).

    for (int y = 0; y < cvSegmentationMask.rows; y++) {
        for (int x = 0; x < cvSegmentationMask.cols; x++) {
            // If the current pixel has already been checked, the loop is continued.
            if (floodFillMask.at(x, y) == 1) {
                continue;
            }

            // Flood fill is performed for each segment using stack in order to avoid nasty C++ recursion weirdnesses.
            std::vector<Pixel> segment;
            std::stack<Pixel> stack;
            stack.push(Pixel(x, y));
            floodFillMask.setAt(x, y, 1);                               // Setting the current pixel as checked.
            const int segmentColor = cvSegmentationMask.at<int>(y, x);  // The color of the current segment.

            while (!stack.empty()) {
                // Retrieving the current pixel from the stack and adding it to the current segment.
                const Pixel currentPixel = stack.top();
                stack.pop();
                segment.push_back(currentPixel);

                // Pushing the top pixel to the stack if the color is the same as the current one.
                if (currentPixel.y - 1 >= 0 &&
                    floodFillMask.at(currentPixel.x, currentPixel.y - 1) == 0 &&
                    static_cast<int>(cvSegmentationMask.at<int>(currentPixel.y - 1, currentPixel.x)) == segmentColor)
                {
                    stack.push(Pixel(currentPixel.x, currentPixel.y - 1));       // Adding the pixel to the stack.
                    floodFillMask.setAt(currentPixel.x, currentPixel.y - 1, 1);  // Setting the pixel as checked.
                }
                // Pushing the bottom pixel to the stack if the color is the same as the current one.
                if (currentPixel.y + 1 < cvSegmentationMask.rows &&
                    floodFillMask.at(currentPixel.x, currentPixel.y + 1) == 0 &&
                    static_cast<int>(cvSegmentationMask.at<int>(currentPixel.y + 1, currentPixel.x)) == segmentColor)
                {
                    stack.push(Pixel(currentPixel.x, currentPixel.y + 1));       // Adding the pixel to the stack.
                    floodFillMask.setAt(currentPixel.x, currentPixel.y + 1, 1);  // Setting the pixel as checked.
                }
                // Pushing the left pixel to the stack if the color is the same as the current one.
                if (currentPixel.x - 1 >= 0 &&
                    floodFillMask.at(currentPixel.x - 1, currentPixel.y) == 0 &&
                    static_cast<int>(cvSegmentationMask.at<int>(currentPixel.y, currentPixel.x - 1)) == segmentColor)
                {
                    stack.push(Pixel(currentPixel.x - 1, currentPixel.y));       // Adding the pixel to the stack.
                    floodFillMask.setAt(currentPixel.x - 1, currentPixel.y, 1);  // Setting the pixel as checked.
                }
                // Pushing the right pixel to the stack if the color is the same as the current one.
                if (currentPixel.x + 1 < cvSegmentationMask.cols &&
                    floodFillMask.at(currentPixel.x + 1, currentPixel.y) == 0 &&
                    static_cast<int>(cvSegmentationMask.at<int>(currentPixel.y, currentPixel.x + 1)) == segmentColor)
                {
                    stack.push(Pixel(currentPixel.x + 1, currentPixel.y));       // Adding the pixel to the stack.
                    floodFillMask.setAt(currentPixel.x + 1, currentPixel.y, 1);  // Setting the pixel as checked.
                }
            }

            segments.push_back(segment);  // Adding the current segment to the vector of all segments.
        }
    }

    return segments;
}


std::pair<std::vector<std::vector<Pixel>>, RawImage> ImageSegmentation::segmentizeImageWithKMeans(const cv::Mat& image, const int K) {
    // Preprocessing of the image for the K-means algorithm.
    cv::Mat blurredImage = image.clone();
    cv::GaussianBlur(image, blurredImage, cv::Size(11, 11), 5);                       // Applying Gaussian blur to the image in order to reduce the impact of noise on the segmentation.
    cv::Mat pixels = blurredImage.reshape(1, blurredImage.rows * blurredImage.cols);  // Reshape the image into a 2D array of pixels.
    pixels.convertTo(pixels, CV_32F);                                                 // Convert the pixels to float32 to ensure the correct input.

    // Segmentation of the image using the K-means algorithm.
    cv::Mat labels, centers;
    cv::kmeans(pixels, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Reshape labels to the original image shape
    cv::Mat segmentedImage = cv::Mat::zeros(image.size(), CV_8U);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const int pixelClass = labels.at<int>(y * image.cols + x);
            const int color = centers.at<int>(pixelClass);
            segmentedImage.at<uchar>(y, x) = color;
        }
    }
    cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)), cv::Point(-1, -1), 3);
    cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)), cv::Point(-1, -1), 3);

    const std::vector<std::vector<Pixel>> segments = extractSegmentsFromSegmentationMask(segmentedImage);
    return { segments, ImageConverter::cvToRaw(segmentedImage) };
}

std::pair<std::vector<std::vector<Pixel>>, RawImage> ImageSegmentation::segmentizeImageWithDBSCAN(const cv::Mat& image, const double epsilon, const int minimumNumberOfNeighbourhoodPoints) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Applying Gaussian blur to the image in order to reduce the impact of noise on the segmentation.
    cv::Mat blurredImage = image.clone();
    //cv::bilateralFilter(image, blurredImage, 5, 15, 15);

    // Converting the image into the Armadillo format.
    arma::mat data(3, blurredImage.rows * blurredImage.cols);
    for (int y = 0; y < blurredImage.rows; ++y) {
        for (int x = 0; x < blurredImage.cols; ++x) {
            const double pixel = static_cast<double>(blurredImage.at<uchar>(y, x));
            data.col(y * blurredImage.cols + x) = { pixel, static_cast<double>(x), static_cast<double>(y) };
        }
    }

    // Performing DBSCAN clustering.
    arma::Row<size_t> assignments;
    arma::mat centroids;
    mlpack::dbscan::DBSCAN<> model(epsilon, minimumNumberOfNeighbourhoodPoints);
    const size_t clusters = model.Cluster(data, assignments, centroids);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;

    // Creating a clustered image.
    cv::Mat clusteredImage(blurredImage.size(), CV_8UC1);
    for (int y = 0; y < blurredImage.rows; ++y) {
        for (int x = 0; x < blurredImage.cols; ++x) {
            // If the point assignment is -1, noise is indicated.
            if (assignments(y * blurredImage.cols + x) == -1) {
                clusteredImage.at<uchar>(y, x) = image.at<uchar>(y, x); //255;  // Noise points carry the average value.
            }
            else {
                clusteredImage.at<uchar>(y, x) = centroids.col(assignments(y * blurredImage.cols + x))[0];  // Setting the color to the point (according to the region centroid color).
            }
        }
    }

    //// Displaying the clustered image.
    //cv::imshow("Clustered Image", clusteredImage);
    //cv::waitKey(0);

    return { extractSegmentsFromSegmentationMask(clusteredImage), ImageConverter::cvToRaw(clusteredImage) };
}

std::pair<std::vector<std::vector<Pixel>>, RawImage> ImageSegmentation::segmentizeImageWithSweepLine(const cv::Mat& image, const double threshold) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat segments(image.rows, image.cols, CV_32S, cv::Scalar(-1));  // Image with segment indices.
    std::vector<double> averageColors;                                 // Average color of each segment.
    std::vector<int> segmentSizes;                                     // Number of pixels inside of each segment.
    std::vector<std::vector<Pixel>> pixelsBySegment;                   // Pixels that belong to each segment.

    //cv::Mat blurredImage;
    //cv::GaussianBlur(image, blurredImage, cv::Size(3, 3), 0);
    int segmentCounter = 0;  // Number of detected segments.

    // Iterating through an image (imitating a sweep-line).
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            const int currentPixelValue = static_cast<int>(image.at<uchar>(y, x));  // Retrieving the current pixel value.
            int upperPixelSegment = -1;
            double upperSegmentAverageColor = -1000.0;
            int leftPixelSegment = -1;
            double leftSegmentAverageColor = -1000.0;

            // Checking for the lower pixel segment.
            if (y > 0) {
                upperPixelSegment = segments.at<int>(y - 1, x);               // Retrieving the lower pixel segment.
                upperSegmentAverageColor = averageColors[upperPixelSegment];  // Retrieving the average color of the upper pixel segment.
            }

            // Checking for the left pixel segment.
            if (x > 0) {
                leftPixelSegment = segments.at<int>(y, x - 1);              // Retrieving the left pixel segment.
                leftSegmentAverageColor = averageColors[leftPixelSegment];  // Retrieving the average color of the left pixel segment.
            }


            // If the difference between the current pixel and both upper and left segments average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (leftPixelSegment != upperPixelSegment && std::abs(currentPixelValue - upperSegmentAverageColor) <= threshold && std::abs(currentPixelValue - leftSegmentAverageColor) <= threshold) {
                segments.at<int>(y, x) = upperPixelSegment;
                averageColors[upperPixelSegment] = (segmentSizes[upperPixelSegment] * averageColors[upperPixelSegment] + segmentSizes[leftPixelSegment] * averageColors[leftPixelSegment] + currentPixelValue) / (segmentSizes[upperPixelSegment] + segmentSizes[leftPixelSegment] + 1);
                segmentSizes[upperPixelSegment] += segmentSizes[leftPixelSegment] + 1;
                pixelsBySegment[upperPixelSegment].push_back(Pixel(x, y));

                //for (const Pixel& pixel : pixelsBySegment[leftPixelSegment]) {
                const int upperSize = pixelsBySegment[upperPixelSegment].size();
                const int leftSide = pixelsBySegment[leftPixelSegment].size();

                if (upperSize > leftSide) {
                    pixelsBySegment[upperPixelSegment].resize(pixelsBySegment[upperPixelSegment].size() + pixelsBySegment[leftPixelSegment].size());
                    for (int i = 0; i < pixelsBySegment[leftPixelSegment].size(); i++) {
                        segments.at<int>(pixelsBySegment[leftPixelSegment][i].y, pixelsBySegment[leftPixelSegment][i].x) = upperPixelSegment;
                    }
                    pixelsBySegment[upperPixelSegment].insert(pixelsBySegment[upperPixelSegment].end(), pixelsBySegment[leftPixelSegment].begin(), pixelsBySegment[leftPixelSegment].end());
                    averageColors[leftPixelSegment] = -1.0;
                    segmentSizes[leftPixelSegment] = -1;
                    pixelsBySegment[leftPixelSegment].clear();
                }
                else {
                    pixelsBySegment[leftPixelSegment].resize(pixelsBySegment[leftPixelSegment].size() + pixelsBySegment[upperPixelSegment].size());
                    for (int i = 0; i < pixelsBySegment[upperPixelSegment].size(); i++) {
                        segments.at<int>(pixelsBySegment[upperPixelSegment][i].y, pixelsBySegment[upperPixelSegment][i].x) = leftPixelSegment;
                    }
                    pixelsBySegment[leftPixelSegment].insert(pixelsBySegment[leftPixelSegment].end(), pixelsBySegment[upperPixelSegment].begin(), pixelsBySegment[upperPixelSegment].end());
                    averageColors[upperPixelSegment] = -1.0;
                    segmentSizes[upperPixelSegment] = -1;
                    pixelsBySegment[upperPixelSegment].clear();
                }

                continue;
            }

            // If the difference between the current pixel and the upper segment average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (std::abs(currentPixelValue - upperSegmentAverageColor) <= threshold) {
                segments.at<int>(y, x) = upperPixelSegment;
                averageColors[upperPixelSegment] = (segmentSizes[upperPixelSegment] * averageColors[upperPixelSegment] + currentPixelValue) / (segmentSizes[upperPixelSegment] + 1);
                segmentSizes[upperPixelSegment]++;
                pixelsBySegment[upperPixelSegment].push_back(Pixel(x, y));

                continue;
            }

            // If the difference between the current pixel and the left segment average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (std::abs(currentPixelValue - leftSegmentAverageColor) <= threshold) {
                segments.at<int>(y, x) = leftPixelSegment;
                averageColors[leftPixelSegment] = (segmentSizes[leftPixelSegment] * averageColors[leftPixelSegment] + currentPixelValue) / (segmentSizes[leftPixelSegment] + 1);
                segmentSizes[leftPixelSegment]++;
                pixelsBySegment[leftPixelSegment].push_back(Pixel(x, y));

                continue;
            }

            // Adding a new segment to the vector.
            averageColors.push_back(currentPixelValue);
            segmentSizes.push_back(1);
            segments.at<int>(y, x) = segmentCounter;
            segmentCounter++;
            pixelsBySegment.push_back({ Pixel(x, y) });
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;

    cv::Mat clusteredImage(image.size(), CV_8UC1);
    for (int y = 0; y < clusteredImage.rows; ++y) {
        for (int x = 0; x < clusteredImage.cols; ++x) {
            clusteredImage.at<uchar>(y, x) = static_cast<uchar>(averageColors[segments.at<int>(y, x)]);
        }
    }

    return { extractSegmentsFromSegmentationMask(segments), ImageConverter::cvToRaw(clusteredImage) };
}
