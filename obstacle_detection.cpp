#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

#include <iostream>
#include <tuple>
#include <fstream>
#include <vector>

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

//#include "contour_detection_slider.cpp"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int disp_img(fs::path image_path) {

	Mat img = imread(image_path.string(), IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Could not read image: " << image_path << endl;
		return 1;
	}

	imshow("Display window", img);
	int k = waitKey(0);

	if (k == 's')
	{
		imwrite(image_path.filename().string(), img);
	}
	return 0;
}

/// <summary>
/// Counts the number of regular files in a library.
/// https://stackoverflow.com/questions/41304891/how-to-count-the-number-of-files-in-a-directory-using-standard/41305019
/// </summary>
/// <param name="dir"></param>
/// <returns></returns>
std::size_t number_of_files_in_directory(std::filesystem::path path)
{
	using std::filesystem::directory_iterator;
	using fp = bool (*)(const std::filesystem::path&);
	return std::count_if(directory_iterator(path), directory_iterator{}, (fp)std::filesystem::is_regular_file);
}

fs::path select_image(int dir_idx, bool random_image = true, fs::path image_name = "none") {
	fs::path dataset_dir = fs::current_path().parent_path() /= "AE4317_2019_datasets";
	string folder_list[14] = {
		"calibration_bottomcam",					// 0
		"calibration_frontcam",						// 1
		"cyberzoo_aggressive_flight",				// 2
		"cyberzoo_aggressive_flight_bottomcam",		// 3
		"cyberzoo_bottomcam",						// 4
		"cyberzoo_canvas_approach",					// 5
		"cyberzoo_poles",							// 6
		"cyberzoo_poles_panels",					// 7
		"cyberzoo_poles_panels_mats",				// 8
		"cyberzoo_poles_panels_mats_bottomcam",		// 9
		"sim_poles",								// 10
		"sim_poles_bottomcam",						// 11
		"sim_poles_panels",							// 12
		"sim_poles_panels_mats"						// 13
	};

	// Select the image set here by changing the name from the array
	fs::path target_folder = dataset_dir /= folder_list[dir_idx];

	// Find the directory containing the images. That directory should be the only directory.
	for (auto const& entry : fs::directory_iterator(target_folder)) {
		if (fs::is_directory(entry.path())) {
			target_folder = entry.path();
			break;
		}
	}

	fs::path target_img;
	if (random_image) {
		// Normalised (between 0 and 1) random number.
		float random_n = rand() / float(RAND_MAX);
		int target_counter = int(random_n * number_of_files_in_directory(target_folder));
		int counter = 0;

		for (auto const& entry : fs::directory_iterator(target_folder)) {
			if (counter == target_counter) {
				target_img = entry.path();
				break;
			}
			else {
				counter++;
			}
		}
	}
	else {
		target_img = target_folder /= image_name;
	}

	// Make sure an image was selected
	if (fs::is_empty(target_img) && fs::is_regular_file(target_img)) {
		cout << "Something went wrong with image selection. The selected path is " << target_img;
	}

	return target_img;
}

Mat load_bw_image(fs::path image_path) {
	Mat src = imread(image_path.string(), IMREAD_GRAYSCALE);
	rotate(src, src, ROTATE_90_COUNTERCLOCKWISE);
	Mat src_gray;
	//cvtColor(src, src_gray, COLOR_BGR2GRAY);

	if (src.empty()) {
		cout << "Could not read image: " << image_path << endl;
	}

	return src_gray;
}
Mat load_cl_image(fs::path image_path) {
	Mat src = imread(image_path.string(), IMREAD_COLOR);
	rotate(src, src, ROTATE_90_COUNTERCLOCKWISE);

	if (src.empty()) {
		cout << "Could not read image: " << image_path << endl;
	}

	return src;
}


/// <summary>
/// Reads the pitch and roll angles of the drone from the csv log, corresponding to the given image frame.
/// </summary>
/// <param name="img_path">Path to the image frame</param>
/// <returns>Tuple of roll and pitch angles (in same unit as in the csv log).</returns>
std::tuple<float, float> retrieve_attitude(fs::path img_path, int flight_case) {
	// File pointer 
	fstream fin;

	// TODO: add the csv files from the other folders!
	// Get the path to the csv containing the flight log.
	string folder_list[14] = {
		"calibration_bottomcam",										// 0
		"calibration_frontcam",											// 1
		"cyberzoo_aggressive_flight",									// 2
		"cyberzoo_aggressive_flight_bottomcam",							// 3
		"cyberzoo_bottomcam",											// 4
		"cyberzoo_canvas_approach",										// 5
		"cyberzoo_poles\\20190121-135121.csv",							// 6
		"cyberzoo_poles_panels\\20190121-140303.csv",					// 7
		"cyberzoo_poles_panels_mats\\20190121-142943.csv",				// 8
		"cyberzoo_poles_panels_mats_bottomcam",							// 9
		"sim_poles\\20190121-160857.csv",								// 10
		"sim_poles_bottomcam",											// 11
		"sim_poles_panels\\20190121-161439.csv",						// 12
		"sim_poles_panels_mats\\20190121-161955.csv"					// 13
	};

	fs::path csv_path;
	for (auto const& entry : fs::directory_iterator(img_path.parent_path().parent_path())) {
		if (fs::is_directory(entry.path())) {
			csv_path = entry.path().parent_path().parent_path();
			csv_path /= folder_list[flight_case];
			break;
		}
	}

	string pathstring = csv_path.string();

	// Open an existing file 
	//ifstream logcsv("C:\\Users\\tothd\\Documents\\TU Delft\\Msc\\Autonomous MAV\\AE4317_2019_datasets\\sim_poles\\20190121-160857.csv");
	ifstream logcsv(pathstring);


	float time_stamp, roll_rad, pitch_rad;
	// Read the Data from the file 
	// as String Vector 
	vector<string> row;
	string line, word, temp;
	int line_count = 0;
	if (logcsv.is_open()) {
		while (getline(logcsv, line)) {
			line_count++;
			row.clear();

			// used for breaking words 
			stringstream s(line);

			// read every column data of a row and 
			// store it in a string variable, 'word' 
			while (getline(s, word, ',')) {
				// add all the column data 
				// of a row to a vector 
				row.push_back(word);
			}
			if (row[0] != "time") {
				// convert string to integer for comparision 
				time_stamp = stof(row[0]);

				// Find the time stamp (roughly) matching the name of the image
				// For the comparission, convert/round the image stem and the time stemp to a 4 digit intiger. 
				if (int(nearbyint(stof(img_path.stem().string()) / 10000)) == int(nearbyint(time_stamp * 100))) {
					roll_rad = stof(row[7]);
					pitch_rad = stof(row[8]);
					break;
				}
			}
		}
		logcsv.close();
	}

	return {roll_rad, pitch_rad};
}


/// <summary>
/// Calculate the horizont line, and delete/blacken everything under the horizont.
/// </summary>
/// <param name="bw_img">Gray-scale image frame</param>
/// <param name="pitch_rad">Drone pitch angle in radiants.</param>
/// <param name="roll_rad">Drone roll angle in radiants.</param>
/// <returns></returns>
Mat horizont_filter(Mat img, float pitch_rad, float roll_rad) {
	// TODO: calculate the horizont line: https://stackoverflow.com/questions/60801612/coding-an-artificial-horizon-with-4-points-and-a-specified-bank-angle
	// TODO: create openCV mat with all black under horizont line
	// TODO: overlay the horizont mat over bw_img

	Point p_center, p_left, p_right;

	p_center.x = img.cols / 2;
	p_center.y = img.rows / 2;

	circle(img, p_center, 10, Scalar(0, 0, 255));

	p_right.x = (int)round(p_center.x + 2000 * cos(roll_rad));
	p_right.y = (int)round(p_center.y + 2000 * sin(roll_rad));

	p_left.x = (int)round(p_center.x - 2000 * cos(roll_rad));
	p_left.y = (int)round(p_center.y - 2000 * sin(roll_rad));

	line(img, p_left, p_right, Scalar(255, 0, 0), 1);

	return img;
}

enum smoothFilters {
	NORMALISED,			// Average.
	GAUSSIAN,			// Apparently most useful, but not the fastest
	MEDIAN,				// 
	BILATERAL			// Keeps the edges. https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
};
/// <summary>
/// Smoothen the image using one of the built-in openCV filters.
/// https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
/// </summary>
/// <param name="img">Mat img</param>
/// <param name="filter_option">NORMALISED, GAUSSIAN, MEDIAN, BILATERAL</param>
/// <param name="kernel_size"></param>
/// <returns>Mat img</returns>
Mat smooth(Mat img, int filter_option = NORMALISED, int kernel_size = 5, int sigma_color_factor = 10) {
	Mat clone = img.clone();
	Mat out_img;
	switch (filter_option)
	{
	case NORMALISED:
		blur(clone, out_img, Size(kernel_size, kernel_size));
	case GAUSSIAN:
		GaussianBlur(clone, out_img, Size(kernel_size, kernel_size), 0, 0);
	case MEDIAN:
		medianBlur(clone, out_img, kernel_size);
	case BILATERAL:
		bilateralFilter(clone, out_img, kernel_size, double(kernel_size) * sigma_color_factor, kernel_size);
		break;
	default:
		break;
	}
	return out_img;
}

Mat edge_detection(Mat img, int low_treshhold, int treshhold_factor = 2) {
	Mat edges;
	edges.create(img.size(), img.type());
	Canny(img, edges, low_treshhold, double(low_treshhold) * treshhold_factor);
	return edges;
}


//int main() {
//	srand(123);
//	int flight_case = 10;
//	fs::path img = select_image(flight_case);
//	cout << img << endl;
//
//
//
//	//contour_detection(img);
//
//
//
//	auto [roll, pitch] = retrieve_attitude(img, flight_case);
//	cout << "roll: " << roll << ", " << "pitch: " << pitch << endl;
//
//	Mat img_bw = load_cl_image(img);
//	horizont_filter(img_bw, pitch, roll);
//	imshow("img", img_bw);
//	waitKey();
//
//
//	return 0;
//}



Mat color_filter(
	Mat img, bool show_img = false,
	Scalar low_hsv = Scalar(35, 100, 90), Scalar high_hsv = Scalar(45, 255, 255), 
	int kernel = 9, int blurring_type = NORMALISED)
{

	Mat img_HSV, threshold, threshold_BGR, combined;

	// Convert from BGR to HSV colorspace
	img = smooth(img, blurring_type, kernel);
	cvtColor(img, img_HSV, COLOR_BGR2HSV);
	// Detect the object based on HSV Range Values
	inRange(img_HSV, low_hsv, high_hsv, threshold);

	if (show_img) {
		cvtColor(threshold, threshold_BGR, COLOR_GRAY2BGR);
		addWeighted(img, 1, threshold_BGR, 1, 0, combined);
		imshow("filtered", img);
		imshow("threshold", threshold);
		imshow("combined", combined);
	}

	return threshold;
}

int bound_int(int num, int min, int max) {
	int out;
	if (num < min) {
		out = min;
	}
	else if (num > max) {
		out = max;
	}
	else {
		out = num;
	}
	return out;
}


/// <summary>
/// Checks the bottom_count of pixels at the bottom of each column if they are white.
/// If more than certainty many black pixels are found at the bottom of the column, it is considered unsafe.
/// </summary>
/// <param name="img"> BW colorfiltered image in openCV::Mat </param>
/// <param name="bottom_count"> The width of the band on the bottom to scan for black </param>
/// <param name="certainty"> The number of black pixels in a column of the bottom_count band, that makes that direction unsafe. </param>
/// <returns> Array with indexes representing the column index, and values: 1 is safe, 2 is obstacle, 0 is outside of frame </returns>
int* ground_obstacle_detect(Mat img, int safe_vector[], int bottom_count = 20, int certainty = 1) {
	// (0, 0) is top left corner
	Mat edited;
	int threat;

	edited = img.clone();

	for (int col = 0; col < img.cols; col=col+1) {
		threat = 0;
		for (int row = int(img.rows) - 1; row >= 0; row--) {
			if (int(img.at<uchar>(row, col)) == 0) {
				if (row >= int(img.rows) - bottom_count) {
					threat++;
				}

			} else if (int(img.at<uchar>(row, col)) == 255) {

				threat--;

			} else {
				cout << "wrong" << endl;
				cout << img.at<uchar>(row, col) << endl;
				break;
			}

			threat = bound_int(threat, 0, certainty);

			if (threat == certainty) {
				safe_vector[col] = 2;
			}
			else if (threat == 0) {
				safe_vector[col] = 1;
			}
			//cout << "(" << col << ", " << row << ") ->  threat = " << threat << ",  safe_vector = " << safe_vector[col] << endl;

		}

		if (safe_vector[col] == 2) {
			line(edited, Point(col, 0), Point(col, edited.rows / 5), Scalar(255));
		}

	}
	//line(edited, Point(0, edited.rows - bottom_count), Point(edited.cols, edited.rows - bottom_count), Scalar(255));
	//imshow("asd", edited);
	//waitKey();
	return safe_vector;
}


int main() {
	Mat frame;
	srand(42069);
	int flight_case = 13, cols, rows;
	while (true) {
		fs::path img = select_image(flight_case);
		cout << img;
		frame = load_cl_image(img);



		Mat greens = color_filter(frame);

		int safe_array[520] = { };
		int *safe_array_pt = ground_obstacle_detect(greens, safe_array);

		for (int i = 0; i < 520; i++) {
			if (safe_array[i] == 2) {
				line(frame, Point(i, 0), Point(i, frame.rows/5), Scalar(0, 0, 255));
			}
		}
		imshow("asd", frame);
		waitKey();
	}
	return 0;
}