#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
vector<Mat> imgs; //保存拼接的原始图像向量
 
//导入所有原始拼接图像函数
void parseCmdArgs(int argc, char** argv);
 
int main(int argc, char* argv[])
{
	//导入拼接图像
	parseCmdArgs(argc, argv);	
	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(imgs, pano);//拼接
	if (status != Stitcher::OK) //判断拼接是否成功
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}
	namedWindow("全景拼接",0);
	imshow("全景拼接",pano);
	imwrite("全景拼接.jpg",pano);
	waitKey();   
	return 0;
}
 
//导入所有原始拼接图像函数
void parseCmdArgs(int argc, char** argv)
{
	for(int i=1;i<argc;i++)
	{
		Mat img = imread(argv[i]);
		if (img.empty())
		{
			cout << "Can't read image '" << argv[i] << "'\n";
		}
		imgs.push_back(img);
	}
}

