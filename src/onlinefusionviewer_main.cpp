#include <auxiliary/multivector.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Geometry>
#include <list>
#include "/home/u20/dataset/code/paper/read.hpp"
#include <fusion/geometryfusion_mipmap_cpu.hpp>
#include "onlinefusionviewer.hpp"
#include <qapplication.h>
#include <fusion/mesh.hpp>

#define BOXDELTA 0.001
//#define VOLUMERADIUS 1.5
#define VOLUMERADIUS 1.4
#define USE_ORIGINAL_VOLUME 1
using namespace std;
using namespace cv;
#include <fusion/definitions.h>

#include <deque>
#include <list>

CameraInfo kinectPoseFromEigen(std::pair<Eigen::Matrix3d, Eigen::Vector3d> pos, float fx, float fy, float cx, float cy)
{
	CameraInfo result;
	cv::Mat intrinsic = cv::Mat::eye(3, 3, cv::DataType<double>::type);
	// Kinect Intrinsic Parameters
	intrinsic.at<double>(0, 0) = fx;
	intrinsic.at<double>(1, 1) = fy;
	intrinsic.at<double>(0, 2) = cx;
	intrinsic.at<double>(1, 2) = cy;

	result.setIntrinsic(intrinsic);
	Eigen::Matrix3d rotation = pos.first;
	cv::Mat rotation2 = cv::Mat::eye(3, 3, cv::DataType<double>::type);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			rotation2.at<double>(i, j) = rotation(i, j);
	result.setRotation(rotation2);
	Eigen::Vector3d translation = pos.second;
	cv::Mat translation2 = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
	for (int i = 0; i < 3; i++)
		translation2.at<double>(i, 0) = translation(i);
	result.setTranslation(translation2);
	return result;
}

int main(int argc, char *argv[])
{
	bool volumeColor = true;

	unsigned int startimage = 1;

	float maxCamDistance = MAXCAMDISTANCE;
	float scale = DEFAULT_SCALE;
	float threshold = DEFAULT_SCALE;

	bool threadMeshing = true;
	bool threadFusion = false;
	bool threadImageReading = false;
	bool performIncrementalMeshing = true;
	int depthConstistencyChecks = 0;
	std::string patth = std::string(argv[1]);
	DataSet<double> dt(patth);
	float imageDepthScale = dt.depth_factor;
	unsigned int endimage = dt.endimage;
	CameraInfo startpos;

	std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> poses_from_assfile;
	std::vector<std::vector<std::string>> depthNames;
	std::vector<std::vector<std::string>> rgbNames;
	std::vector<std::vector<CameraInfo>> poses;

	fprintf(stderr, "\nBuilding a single Trajectory...");
	poses.push_back(std::vector<CameraInfo>());
	depthNames.push_back(std::vector<std::string>());
	rgbNames.push_back(std::vector<std::string>());
	std::vector<CameraInfo> &trajectory = poses.back();
	std::vector<std::string> &depthNamesLast = depthNames.back();
	std::vector<std::string> &rgbNamesLast = rgbNames.back();
	for (size_t ant = 0; ant < dt.pose.qs.size(); ant++)
	{
		poses_from_assfile.push_back(std::pair<Eigen::Matrix3d, Eigen::Vector3d>(dt.pose.qs[ant], dt.pose.ts[ant]));
		depthNamesLast.push_back(dt.depth_path[ant]);
		rgbNamesLast.push_back(dt.color_path[ant]);
	}
	for (unsigned int i = 0; i < poses_from_assfile.size(); i++)
	{
		trajectory.push_back(kinectPoseFromEigen(poses_from_assfile[i], dt.fx, dt.fy, dt.cx, dt.cy));
		trajectory.back().setExtrinsic(startpos.getExtrinsic() * trajectory.back().getExtrinsic());
	}
	fprintf(stderr, "\n poses_from_assfile %ld", poses_from_assfile.size());

	if (startimage >= depthNames.front().size())
		startimage = depthNames.front().size() - 1;
	if (endimage >= depthNames.back().size())
		endimage = depthNames.back().size() - 1;

	FusionMipMapCPU *fusion = new FusionMipMapCPU(0, 0, 0, scale, threshold, 0, volumeColor);

	fusion->setThreadMeshing(threadMeshing);
	fusion->setDepthChecks(depthConstistencyChecks);
	fusion->setIncrementalMeshing(performIncrementalMeshing);

	QApplication application(argc, argv);
	OnlineFusionViewerManipulated viewer(false);
	viewer.saveFileName = std::string(argv[1]) + "/fastfusion";
	viewer.DEPTHSCALE = imageDepthScale;
	viewer._fusion = fusion;
	viewer.setWindowTitle("Fusion Volume");
	viewer._poses = poses;
	viewer._depthNames = depthNames;
	viewer._rgbNames = rgbNames;
	viewer._threadFusion = threadFusion;
	viewer._threadImageReading = threadImageReading;
	viewer.show();
	viewer._imageDepthScale = imageDepthScale;
	viewer._maxCamDistance = maxCamDistance;
	viewer._firstFrame = (long int)startimage;
	viewer._currentFrame = (long int)startimage - 1;
	fprintf(stderr, "\nSet Viewer Frame to %li", (long int)viewer._currentFrame);
	viewer._nextStopFrame = endimage;
	application.exec();
	return 0;
}
