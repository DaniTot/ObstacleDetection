#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

#include <iostream>
#include <tuple>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Mat clone;
int thresh = 10;
int kernel = 2;
RNG rng(12345);
void thresh_callback(int, void*);
int contour_detection(fs::path img_path)
{
	Mat src = imread(img_path.string());
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}
	cout << img_path.string();
	cvtColor(src, clone, COLOR_BGR2GRAY);

	const char* source_window = "Source";
	namedWindow(source_window);
	imshow(source_window, src);
	const int max_thresh = 30;
	const int max_kernel = 20;
	createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
	createTrackbar("Kernel:", source_window, &kernel, max_kernel, thresh_callback);
	thresh_callback(0, 0);
	waitKey();
	return 0;
}
void thresh_callback(int, void*)
{
	Mat src_gray;
	int krnl;
	krnl = kernel * 2 - 1;
	//blur(clone, src_gray, Size(krnl, krnl));
	//medianBlur(clone, src_gray, krnl);
	/*bilateralFilter(clone, src_gray, krnl, double(krnl) * 2, krnl);*/
	GaussianBlur(clone, src_gray, Size(krnl, krnl), 0, 0);
	imshow("gray", src_gray);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat grad;
	Sobel(src_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow("sobel", grad);

	//Mat filt_grad;
	//bilateralFilter(grad, filt_grad, krnl, double(krnl) * 2, krnl);

	Mat canny_output;
	Canny(grad, canny_output, thresh * 10, thresh * 10 * 2);
	imshow("canny", canny_output);

	vector<vector<Point> > contours;
	findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(contours[i]);
		//if (contours[i].size() > 5)
		//{
		//	minEllipse[i] = fitEllipse(contours[i]);
		//}
	}
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		// contour
		drawContours(drawing, contours, (int)i, color);
		// ellipse
		/*ellipse(drawing, minEllipse[i], color, 2);*/
		// rotated rectangle
		Point2f rect_points[4];
		minRect[i].points(rect_points);
		for (int j = 0; j < 4; j++)
		{
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color);
		}
	}
	imshow("Contours", drawing);
}