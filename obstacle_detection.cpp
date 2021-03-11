#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

#include <iostream>
#include <tuple>
#include <fstream>
#include <vector>

#include "contour_detection_slider.cpp"

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
	Mat src = imread(image_path.string());
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	if (src.empty()) {
		cout << "Could not read image: " << image_path << endl;
	}

	return src_gray;
}
Mat load_cl_image(fs::path image_path) {
	Mat src = imread(image_path.string());

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
Mat horizont_filter(Mat bw_img, float pitch_rad, float roll_rad) {
	// TODO: calculate the horizont line: https://stackoverflow.com/questions/60801612/coding-an-artificial-horizon-with-4-points-and-a-specified-bank-angle
	// TODO: create openCV mat with all black under horizont line
	// TODO: overlay the horizont mat over bw_img
	return bw_img;
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

void find_params(
	int iterations = 10, int case_type = 10,
	int smooth_factor_low = 10, int smooth_factor_high = 30, int smooth_factor_step = 10, int type = NORMALISED, int kernel = 3,
	int canny_low_trsh_low = 20, int canny_low_trsh_high = 40, int canny_low_trsh_step = 10)
{	
	fs::path img_path;
	string name;
	Mat img, filtered, edges;

	for (int i = 0; i < iterations; i++) {
		img_path = select_image(case_type);
		img = load_bw_image(img_path);
		name = img_path.stem().string() + "_orig.jpg";
		imshow(name, img);
		int l = waitKey(0);
		imwrite(name, img);

		for (int smooth_factor = smooth_factor_low; smooth_factor <= smooth_factor_high; smooth_factor = smooth_factor + smooth_factor_step) {
			filtered = smooth(img, type, kernel, smooth_factor);
			for (int low_trsh = canny_low_trsh_low; low_trsh <= canny_low_trsh_high; low_trsh = low_trsh + canny_low_trsh_step) {
				edges = edge_detection(filtered, low_trsh);
				name = img_path.stem().string() + "_" + "sf" + to_string(smooth_factor) + "_" + "lt" + to_string(low_trsh) + ".jpg";
				imshow("sf" + to_string(smooth_factor) + "_" + "lt" + to_string(low_trsh), edges);
				int l = waitKey(0);
				imwrite(name, edges);
			}
		}
	}
	return;
}


int main() {
	srand(123);
	int flight_case = 10;
	fs::path img = select_image(flight_case);
	cout << img << endl;



	//contour_detection(img);



	auto [roll, pitch] = retrieve_attitude(img, flight_case);
	cout << "roll: " << roll << ", " << "pitch: " << pitch << endl;

	Mat img_bw = load_bw_image(img);
	horizont_filter(img_bw, pitch, roll);



	return 0;
}



