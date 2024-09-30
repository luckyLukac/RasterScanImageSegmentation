#include <chrono>
#include <stack>
#include <random>

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

std::pair<std::vector<std::vector<Pixel>>, QImage> ImageSegmentation::segmentizeImageWithDBSCAN(cv::Mat& image, const double epsilon, const int minimumNumberOfNeighbourhoodPoints, const int seed) {
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
    //std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    //std::cout << "Number of segments: " << centroids.size() << std::endl << std::endl;

    // Creating a clustered image.
    //cv::Mat clusteredImage(blurredImage.size(), CV_8UC1);
    //for (int y = 0; y < blurredImage.rows; ++y) {
    //    for (int x = 0; x < blurredImage.cols; ++x) {
    //        // If the point assignment is -1, noise is indicated.
    //        if (assignments(y * blurredImage.cols + x) == -1) {
    //            clusteredImage.at<uchar>(y, x) = image.at<uchar>(y, x); //255;  // Noise points carry the average value.
    //        }
    //        else {
    //            clusteredImage.at<uchar>(y, x) = centroids.col(assignments(y * blurredImage.cols + x))[0];  // Setting the color to the point (according to the region centroid color).
    //        }
    //    }
    //}

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distr(0, 255);

    int count = centroids.size();
    int noisyCount = 0;
    cv::Mat clusteredImage = cv::Mat::zeros(image.size(), CV_8UC3);
    std::vector<std::array<int, 3>> segmentColors(assignments.size());
    for (int i = 0; i < assignments.size(); i++) {
        segmentColors[i][0] = static_cast<uchar>(distr(gen));
        segmentColors[i][1] = static_cast<uchar>(distr(gen));
        segmentColors[i][2] = static_cast<uchar>(distr(gen));
    }


    for (int y = 0; y < blurredImage.rows; ++y) {
        for (int x = 0; x < blurredImage.cols; ++x) {
            cv::Vec3b& pixel = clusteredImage.at<cv::Vec3b>(y, x);

            if (assignments(y * blurredImage.cols + x) == -1) {
                noisyCount++;
                pixel[0] = static_cast<uchar>(distr(gen));
                pixel[1] = static_cast<uchar>(distr(gen));
                pixel[2] = static_cast<uchar>(distr(gen));
                continue;
            }

            pixel[0] = static_cast<uchar>(segmentColors[assignments(y * blurredImage.cols + x)][0]);
            pixel[1] = static_cast<uchar>(segmentColors[assignments(y * blurredImage.cols + x)][1]);
            pixel[2] = static_cast<uchar>(segmentColors[assignments(y * blurredImage.cols + x)][2]);
        }
    }

    std::vector<int> segmentSizesA(clusters);
    for (int i = 0; i < blurredImage.rows; i++) {
        for (int j = 0; j < blurredImage.cols; j++) {
            int xxxx = assignments(i * blurredImage.cols + j);
            if (xxxx != -1) {
                segmentSizesA[xxxx]++;
            }
        }
    }
    for (int i = 0; i < noisyCount; i++) {
        segmentSizesA.push_back(1);
    }


    double sum = std::accumulate(segmentSizesA.begin(), segmentSizesA.end(), 0.0);
    double mean = sum / segmentSizesA.size();
    double sq_sum = std::inner_product(segmentSizesA.begin(), segmentSizesA.end(), segmentSizesA.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / segmentSizesA.size() - mean * mean);
    std::cout << "Avg. segment: " << mean << " +- " << stdev << std::endl;

    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Number of segments: " << segmentSizesA.size() << std::endl << std::endl;
    //std::cout << "Number of noisy segments: " << noisyCount << std::endl << std::endl;

    QImage qimage(clusteredImage.data, clusteredImage.cols, clusteredImage.rows, clusteredImage.step, QImage::Format_RGB888);
    qimage = qimage.rgbSwapped();


    return { extractSegmentsFromSegmentationMask(clusteredImage), qimage };
}

std::pair<std::vector<std::vector<Pixel>>, QImage> ImageSegmentation::segmentizeImageWithSweepLine(cv::Mat& image, const double threshold, const int maxDeviation, const Distance& distance, const int seed) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat filteredImage = image.clone();
    cv::bilateralFilter(image, filteredImage, 3, 50, 50);
    cv::Mat segments(filteredImage.rows, filteredImage.cols, CV_32S, cv::Scalar(-1));  // Image with segment indices.

    std::vector<std::vector<Pixel>> pixelsBySegment;                   // Pixels that belong to each segment.
    std::vector<double> averageColors;                                 // Average color of each segment.
    std::vector<int> segmentSizes;                                     // Number of pixels inside of each segment.
    std::vector<int> segmentSums;                                      // Sum of pixel values inside of each segment.
    std::vector<std::pair<int, int>> segmentExtremes;                  // Extreme values in segments.

    int segmentCounter = 0;  // Number of detected segments.

    // Iterating through an image (imitating a sweep-line).
    for (int y = 0; y < filteredImage.rows; y++) {
        for (int x = 0; x < filteredImage.cols; x++) {
            const int currentPixelValue = static_cast<int>(filteredImage.at<uchar>(y, x));  // Retrieving the current pixel value.
            int segmentX1 = -1;
            int segmentX2 = -1;
            int segmentY1 = -1;
            int segmentY2 = -1;
            double distanceX = 100000.0;
            double distanceY = 100000.0;

            // Checking for the upper pixel segment.
            if (y > 0) {
                segmentY1 = segments.at<int>(y - 1, x);  // Retrieving the upper pixel segment.
            }
            if (y > 1) {
                segmentY2 = segments.at<int>(y - 2, x);  // Retrieving the upper upper pixel segment.
            }

            // Checking for the left pixel segment.
            if (x > 0) {
                segmentX1 = segments.at<int>(y, x - 1);  // Retrieving the left pixel segment.
            }
            if (x > 1) {
                segmentX2 = segments.at<int>(y, x - 2);  // Retrieving the left left pixel segment.
            }


            if (distance == Distance::onePixel) {
                if (x > 0)
                    distanceX = std::abs(currentPixelValue - filteredImage.at<uchar>(y, x - 1));
                if (y > 0)
                    distanceY = std::abs(currentPixelValue - filteredImage.at<uchar>(y - 1, x));
            }
            else if (distance == Distance::twoPixel) {
                if (x > 1)
                    distanceX = std::abs(currentPixelValue - (filteredImage.at<uchar>(y, x - 1) + filteredImage.at<uchar>(y, x - 2)) / 2.0);
                else if (x > 0)
                    distanceX = std::abs(currentPixelValue - filteredImage.at<uchar>(y, x - 1));

                if (y > 1)
                    distanceY = std::abs(currentPixelValue - (filteredImage.at<uchar>(y - 1, x) + filteredImage.at<uchar>(y - 2, x)) / 2.0);
                else if (y > 0)
                    distanceY = std::abs(currentPixelValue - filteredImage.at<uchar>(y - 1, x));
            }
            else if (distance == Distance::segmentAverage) {
                if (x > 0)
                    distanceX = std::abs(currentPixelValue - static_cast<double>(segmentSums[segmentX1]) / segmentSizes[segmentX1]);
                if (y > 0)
                    distanceY = std::abs(currentPixelValue - static_cast<double>(segmentSums[segmentY1]) / segmentSizes[segmentY1]);
            }

            //const int extremeDeviationX = std::max(segmentX1 == -1 ? 0 : std::abs(currentPixelValue - segmentExtremes[segmentX1].first), segmentX1 == -1 ? 0 : std::abs(currentPixelValue - segmentExtremes[segmentX1].second));
            //const int extremeDeviationY = std::max(segmentY1 == -1 ? 0 : std::abs(currentPixelValue - segmentExtremes[segmentY1].first), segmentY1 == -1 ? 0 : std::abs(currentPixelValue - segmentExtremes[segmentY1].second));
            const int extremeDeviationX = segmentX1 >= 0 ? (currentPixelValue >= (segmentSums[segmentX1] / segmentSizes[segmentX1]) - maxDeviation && currentPixelValue <= (segmentSums[segmentX1] / segmentSizes[segmentX1]) + maxDeviation) : 0;
            const int extremeDeviationY = segmentY1 >= 0 ? (currentPixelValue >= (segmentSums[segmentY1] / segmentSizes[segmentY1]) - maxDeviation && currentPixelValue <= (segmentSums[segmentY1] / segmentSizes[segmentY1]) + maxDeviation) : 0;

            // If the difference between the current pixel and both upper and left segments average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (segmentX1 != segmentY1 && x > 0 && y > 0 && distanceX <= threshold && distanceY <= threshold && extremeDeviationX && extremeDeviationY) {
                const int upperSize = pixelsBySegment[segmentY1].size();
                const int leftSide = pixelsBySegment[segmentX1].size();

                if (upperSize > leftSide) {
                    segments.at<int>(y, x) = segmentY1;
                    pixelsBySegment[segmentY1].resize(pixelsBySegment[segmentY1].size() + pixelsBySegment[segmentX1].size());
                    for (int i = 0; i < pixelsBySegment[segmentX1].size(); i++) {
                        segments.at<int>(pixelsBySegment[segmentX1][i].y, pixelsBySegment[segmentX1][i].x) = segmentY1;
                    }

                    segmentSizes[segmentY1] += segmentSizes[segmentX1] + 1;
                    segmentSums[segmentY1] += segmentSums[segmentX1] + currentPixelValue;
                    pixelsBySegment[segmentY1].push_back(Pixel(x, y));

                    pixelsBySegment[segmentY1].insert(pixelsBySegment[segmentY1].end(), pixelsBySegment[segmentX1].begin(), pixelsBySegment[segmentX1].end());
                    segmentSizes[segmentX1] = -1;
                    pixelsBySegment[segmentX1].clear();

                    //segmentExtremes[segmentY1].first = std::min({ currentPixelValue, segmentExtremes[segmentX1].first, segmentExtremes[segmentY1].first });
                    //segmentExtremes[segmentY1].second = std::max({ currentPixelValue, segmentExtremes[segmentX1].second, segmentExtremes[segmentY1].second });
                }
                else {
                    segments.at<int>(y, x) = segmentX1;
                    pixelsBySegment[segmentX1].resize(pixelsBySegment[segmentX1].size() + pixelsBySegment[segmentY1].size());
                    for (int i = 0; i < pixelsBySegment[segmentY1].size(); i++) {
                        segments.at<int>(pixelsBySegment[segmentY1][i].y, pixelsBySegment[segmentY1][i].x) = segmentX1;
                    }

                    segmentSizes[segmentX1] += segmentSizes[segmentY1] + 1;
                    segmentSums[segmentX1] += segmentSums[segmentY1] + currentPixelValue;
                    pixelsBySegment[segmentX1].push_back(Pixel(x, y));

                    pixelsBySegment[segmentX1].insert(pixelsBySegment[segmentX1].end(), pixelsBySegment[segmentY1].begin(), pixelsBySegment[segmentY1].end());
                    segmentSizes[segmentY1] = -1;
                    pixelsBySegment[segmentY1].clear();

                    //segmentExtremes[segmentX1].first = std::min({ currentPixelValue, segmentExtremes[segmentX1].first, segmentExtremes[segmentY1].first });
                    //segmentExtremes[segmentX1].second = std::max({ currentPixelValue, segmentExtremes[segmentX1].second, segmentExtremes[segmentY1].second });
                }

                continue;
            }

            // If the difference between the current pixel and the upper segment average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (y > 0 && distanceY <= threshold && extremeDeviationY) {
                segments.at<int>(y, x) = segmentY1;
                segmentSizes[segmentY1]++;
                segmentSums[segmentY1] += currentPixelValue;
                pixelsBySegment[segmentY1].push_back(Pixel(x, y));

                if (currentPixelValue < segmentExtremes[segmentY1].first) {
                    //segmentExtremes[segmentY1].first = currentPixelValue;
                }
                if (currentPixelValue > segmentExtremes[segmentY1].second) {
                    //segmentExtremes[segmentY1].second = currentPixelValue;
                }

                continue;
            }

            // If the difference between the current pixel and the left segment average color is below
            // the given threshold, the current pixel is appended to the observed segment.
            if (x > 0 && distanceX <= threshold && extremeDeviationX) {
                segments.at<int>(y, x) = segmentX1;
                segmentSizes[segmentX1]++;
                segmentSums[segmentX1] += currentPixelValue;
                pixelsBySegment[segmentX1].push_back(Pixel(x, y));

                if (currentPixelValue < segmentExtremes[segmentX1].first) {
                    //segmentExtremes[segmentX1].first = currentPixelValue;
                }
                if (currentPixelValue > segmentExtremes[segmentX1].second) {
                    //segmentExtremes[segmentX1].second = currentPixelValue;
                }

                continue;
            }

            // Adding a new segment to the vector.
            segmentSizes.push_back(1);
            segmentSums.push_back(currentPixelValue);
            segments.at<int>(y, x) = segmentCounter;
            segmentCounter++;
            pixelsBySegment.push_back({ Pixel(x, y) });
            segmentExtremes.push_back({ currentPixelValue - maxDeviation, currentPixelValue + maxDeviation });
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start);


    int thres = 0;
    int counterActualSegments = 0;
    int counterNoise = 0;
    for (int i = 0; i < pixelsBySegment.size(); i++) {
        if (pixelsBySegment[i].size() > thres) {
            counterActualSegments++;
        }
        else if (pixelsBySegment[i].size() < thres) {
            counterNoise++;
        }
    }

    std::vector<int> segmentSizesA(pixelsBySegment.size());
    int counterSegmentSize = 0;
    for (int i = 0; i < segments.rows; i++) {
        for (int j = 0; j < segments.cols; j++) {
            segmentSizesA[segments.at<int>(i, j)]++;
        }
    }

    segmentSizesA.erase(std::remove(segmentSizesA.begin(), segmentSizesA.end(), 0), segmentSizesA.end());
    segmentSizesA.shrink_to_fit();

    double sum = std::accumulate(segmentSizesA.begin(), segmentSizesA.end(), 0.0);
    double mean = sum / segmentSizesA.size();
    double sq_sum = std::inner_product(segmentSizesA.begin(), segmentSizesA.end(), segmentSizesA.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / segmentSizesA.size() - mean * mean);
    //std::cout << "Avg. segment: " << mean << " +- " << stdev << std::endl;

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distr(0, 255);

    cv::Mat clusteredImage = cv::Mat::zeros(filteredImage.size(), CV_8UC3);
    std::vector<std::array<int, 3>> segmentColors(pixelsBySegment.size());
    for (int i = 0; i < pixelsBySegment.size(); i++) {
        segmentColors[i][0] = static_cast<uchar>(distr(gen));
        segmentColors[i][1] = static_cast<uchar>(distr(gen));
        segmentColors[i][2] = static_cast<uchar>(distr(gen));
    }
    for (int i = 0; i < pixelsBySegment.size(); i++) {
        for (int j = 0; j < pixelsBySegment[i].size(); j++) {
            cv::Vec3b& pixel = clusteredImage.at<cv::Vec3b>(pixelsBySegment[i][j].y, pixelsBySegment[i][j].x);
            pixel[0] = static_cast<uchar>(segmentColors[i][0]);
            pixel[1] = static_cast<uchar>(segmentColors[i][1]);
            pixel[2] = static_cast<uchar>(segmentColors[i][2]);
        }
    }

    std::cout << /*"Elapsed time: " << */duration.count() /*<< " seconds" */<< std::endl;
    //std::cout <</* "Number of segments: " << */counterActualSegments + counterNoise << std::endl/* << std::endl*/;
    //std::cout << "Number of noisy segments: " << counterNoise << std::endl << std::endl;

    QImage qimage(clusteredImage.data, clusteredImage.cols, clusteredImage.rows, clusteredImage.step, QImage::Format_RGB888);
    qimage = qimage.rgbSwapped();

    return { extractSegmentsFromSegmentationMask(segments), qimage };
}
