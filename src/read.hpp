#pragma once
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace cv;
#include <sys/time.h>
// // template < T>
template <class _Tp>
class Pose
{
public:
    vector<Eigen::Quaternion<_Tp>> qs;
    vector<Eigen::Matrix<_Tp, 3, 1>> ts;
    vector<string> depths;
    vector<string> rgbs;
    vector<uint64_t> times;
    uint16_t frames = 0;

    Eigen::Matrix<_Tp, 4, 4> operator()(int i)
    {
        Eigen::Matrix<_Tp, 4, 4> T2 = Eigen::Matrix<_Tp, 4, 4>::Identity();
        // Eigen::Matrix<_Tp,4,4> T2;
        // T2.setIdentity();
        // Eigen::Matrix4d T2;
        //  T2.block<3,3>(0, 0)
        //    t1 =
        T2.topLeftCorner(3, 3) = qs[i].toRotationMatrix();
        T2.topRightCorner(3, 1) = ts[i];

        // Eigen::Isometry3f T;
        // T.setIdentity();
        // T.pretranslate(ts[id]).rotate(qs[id]);
        return T2;
    }
    void totxt()
    {
        cout << "#id  times x y z  (x y z w)" << endl;
        for (int i = 0; i < times.size(); i++)
        {
            // cout << times[i] <<ts[0]<<ts[1]<<ts[2]<< endl;
            printf("%d %ld %f %f %f ", i + 1, times[i], ts[i][0], ts[i][1], ts[i][2]);
            printf("%f %f %f %f\n", qs[i].x(), qs[i].y(), qs[i].z(), qs[i].w());
        }
    }

    void load(string str)
    {
        fstream fs(str, ios::in);
        if (!fs.is_open())
        {
            cout << str << endl;
            assert(0);
        }

        string temp;
        getline(fs, temp);
        while (true)
        {
            _Tp idx;
            fs >> idx;
            uint64_t start;
            fs >> start;
            _Tp _pose[7];
            for (int i = 0; i < 7; i++)
                fs >> _pose[i];

            if (!fs.good())
                break;
            Eigen::Quaternion<_Tp> q(_pose[6], _pose[3], _pose[4], _pose[5]); // w xyz
            qs.push_back(q);
            ts.push_back(Eigen::Matrix<_Tp, 3, 1>(_pose[0], _pose[1], _pose[2]));
            times.push_back(start);
        }
        frames = times.size();
    }

    void loadtum(string str)
    {
        fstream fs(str, ios::in);
        if (!fs.is_open())
        {
            cout << str << endl;
            assert(0);
        }

        string temp;

        while (true)
        {
            double idx;
            fs >> idx;
            uint64_t start = idx * 1000000;
            // fs >> start;
            _Tp _pose[7];
            for (int i = 0; i < 7; i++)
                fs >> _pose[i];

            fs >> temp;
            depths.push_back(temp);
            auto depth = cv::imread("../../" + temp, 2);
            // Mat depth5,rgb5;
            // depth.convertTo(depth5,-1, 0.2, 0);
            cv::imwrite(cv::format("../../depth/%ld.pgm", qs.size()), depth);
            fs >> temp;
            auto rgb = cv::imread("../../" + temp);
            cv::imwrite(cv::format("../../rgb/%ld.ppm", qs.size()), rgb);
            rgbs.push_back(temp);
            if (!fs.good())
                break;
            Eigen::Quaternion<_Tp> q(_pose[6], _pose[3], _pose[4], _pose[5]); // w xyz
            qs.push_back(q);
            ts.push_back(Eigen::Matrix<_Tp, 3, 1>(_pose[0], _pose[1], _pose[2]));
            times.push_back(start);
        }
        frames = times.size();
    }
};

// static __time_t GetUTC() //   微秒
// {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     // stringstream s;
//     // s<<tv.tv_sec;
//     // printf("second:%ld \n", tv.tv_sec);                                 //秒
//     // printf("millisecond:%ld \n", tv.tv_sec * 1000 + tv.tv_usec / 1000); //毫秒
//     // printf("microsecond:%ld \n", tv.tv_sec * 1000000 + tv.tv_usec);     //微秒
//     return tv.tv_sec * 1000000 + tv.tv_usec;
// }
// /*void read_ic()
// {
//     fstream fs("/home/lei/dataset/paper/0/livingRoom.gt.freiburg", ios::in);

//     assert(fs.is_open());
//     Pose p;
//     uint64_t start = GetUTC();
//     while (true)
//     {
//         double idx;
//         fs >> idx;
//         double _pose[7];

//         for (int i = 0; i < 7; i++)
//         {
//             fs >> _pose[i];
//             // std::cout<<_pose[i]<<" ";
//         }

//         double marix[9];

//         Eigen::Quaterniond q(_pose[6], _pose[3], _pose[4], _pose[5]); // w xyz
//         p.qs.push_back(q);
//         p.ts.push_back(Eigen::Vector3d(_pose[0], _pose[1], _pose[2]));
//         p.times.push_back(start);
//         start += 30000;
//         if (!fs.good())
//             break;
//     }
// }*/
template <typename _Tp>
class DataSet
{
private:
    /* data */
public:
    Pose<_Tp> pose;
    Mat cam_K;
    vector<string> color_path;
    vector<string> depth_path;
    int endimage = 100;
    struct
    {
        float fx, fy, cx, cy;
    };
    DataSet(string datapath = "/home/u20/dataset/paper/0", string yamlpath = "../info.yaml")
    {
        cv::FileStorage fs(datapath + "/info.yaml", cv::FileStorage::READ);
        fs["cam_K"] >> cam_K;
        string posefile;
        fs["posefile"] >> posefile;
        pose.load(datapath + posefile);
        fs["posefile"] >> endimage;
        fs.release();
        cout << cam_K << endl;
        fx = cam_K.at<float>(0, 0);
        fy = cam_K.at<float>(1, 1);
        cx = cam_K.at<float>(0, 2);
        cy = cam_K.at<float>(1, 2);

        for (int i = 0; i < pose.frames; i++)
        {

            color_path.push_back(datapath + cv::format("/rgb/%d.png", i));
            depth_path.push_back(datapath + cv::format("/depth/%d.png", i));
        }
        // pose.totxt();
    }
};

// // int main()
// // {
// //     DataSet dt("/home/lei/dataset/paper/0");
// // }
