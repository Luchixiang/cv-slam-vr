#ifndef DEPTH_H
#define DEPTH_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;

#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>


//获取深度像素对应长度单位（米）的换算比例,D435默认1毫米
float get_depth_scale(rs2::device dev)
{
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        //检查设备是否是深度相机
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("设备不存在深度传感器！");
}

//深度图对齐到彩色图
Mat align_Depth2Color(Mat depth,Mat color,rs2::pipeline_profile profile){
    //声明数据流
    auto depth_stream=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream=profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    //获取内参
    const auto intrinDepth=depth_stream.get_intrinsics();
    const auto intrinColor=color_stream.get_intrinsics();

    //直接获取从深度摄像头坐标系到彩色摄像头坐标系的欧式变换矩阵
    //auto  extrinDepth2Color=depth_stream.get_extrinsics_to(color_stream);
    rs2_extrinsics  extrinDepth2Color;
    rs2_error *error;
    rs2_get_extrinsics(depth_stream,color_stream,&extrinDepth2Color,&error);

    //平面点定义
    float pd_uv[2],pc_uv[2];
    //空间点定义
    float Pdc3[3],Pcc3[3];

    //获取深度像素与现实单位比例（D435默认1毫米）
    float depth_scale = get_depth_scale(profile.get_device());
    int y=0,x=0;
    //初始化结果
    //Mat result=Mat(color.rows,color.cols,CV_8UC3,Scalar(0,0,0));
    Mat result=Mat(color.rows,color.cols,CV_16U,Scalar(0));
    //对深度图像遍历
    for(int row=0;row<depth.rows;row++){
        for(int col=0;col<depth.cols;col++){
            //将当前的(x,y)放入数组pd_uv，表示当前深度图的点
            pd_uv[0]=col;
            pd_uv[1]=row;
            //取当前点对应的深度值
            uint16_t depth_value=depth.at<uint16_t>(row,col);
            //换算到米
            float depth_m=depth_value*depth_scale;
            //将深度图的像素点根据内参转换到深度摄像头坐标系下的三维点
            rs2_deproject_pixel_to_point(Pdc3,&intrinDepth,pd_uv,depth_m);
            //将深度摄像头坐标系的三维点转化到彩色摄像头坐标系下
            rs2_transform_point_to_point(Pcc3,&extrinDepth2Color,Pdc3);
            //将彩色摄像头坐标系下的深度三维点映射到二维平面上
            rs2_project_point_to_pixel(pc_uv,&intrinColor,Pcc3);

            //取得映射后的（u,v)
            x=(int)pc_uv[0];
            y=(int)pc_uv[1];
            //if(x<0||x>color.cols)
            //    continue;
            //if(y<0||y>color.rows)
            //    continue;

            //最值限定
            x=x<0? 0:x;
            x=x>depth.cols-1 ? depth.cols-1:x;
            y=y<0? 0:y;
            y=y>depth.rows-1 ? depth.rows-1:y;

            result.at<uint16_t>(y,x)=depth_value;
        }
    }
    //返回一个与彩色图对齐了的深度信息图像，最终展示不需要给出深度图
    return result;
}

void measure_distance(Mat &color,Mat depth,cv::Size range,
        rs2::pipeline_profile profile, int X, int Y, bool Distance_low)
{

    //获取深度像素与现实单位比例（D435默认1毫米）
    float depth_scale = get_depth_scale(profile.get_device());//结果=0.001(m)

    /******************************
     * 单个像素点输出深度可行 但稳定性太差 采取平均深度更好
    int x=RectRange.x,y=RectRange.y;
    if(depth.at<uint16_t>(x,y)){
        effective_distance=depth_scale*depth.at<uint16_t>(x,y);
    }
    cout<<x<<" "<<y<<endl;
     ****************************/
    ////定义图像中心点
    ////cv::Point center(color.cols/2,color.rows/2);

    //定义从主函数传递过来门的坐标(取中心值)
    cv::Point center(X,Y);

    //定义计算距离的范围为该坐标为中心，向四个方向拓展像素点，取平均深度
    cv::Rect RectRange(center.x-range.width/2,center.y-range.height/2,range.width,range.height);

    //遍历该范围
    float distance_sum=0;
    int effective_pixel=0;
    float effective_distance=0;

    for(int y=RectRange.y;y<RectRange.y+RectRange.height;y++){
        for(int x=RectRange.x;x<RectRange.x+RectRange.width;x++){
            //如果深度图下该点像素不为0，表示有距离信息
            if(depth.at<uint16_t>(y,x)){
                distance_sum+=depth_scale*depth.at<uint16_t>(y,x);
                effective_pixel++;
            }
        }
    }

    //cout<<"遍历完成，有效像素点:"<<effective_pixel<<endl;
    effective_distance=distance_sum/effective_pixel;
    //cout<<"目标距离："<<effective_distance<<"m"<<endl;

    char distance_str[100];

    //弹窗、提示、什么的在这里
    if (effective_distance>0.40){
        //float太长，我们只需要小数点后两位
        sprintf(distance_str,"Cam is%5.2fm away from the door!",effective_distance);
        Distance_low = false;
    }
    else{
        sprintf(distance_str,"Distance is lower than 0.4m now!");
        Distance_low = true;
    }



    cv::rectangle(color,RectRange,Scalar(0,0,255),2,8);
    cv::putText(color,(string)distance_str,cv::Point(color.cols*0.02,color.rows*0.05),
                cv::FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);
}

//测距功能入口,最后返回的是布尔变量判断距离门是否小于0.4米
float getdepth(int X, int Y, int width_for_avg, int height_for_avg)
{
    //定义判别距离是否小于0.4米
    bool Distance_lower = false;

    //分别定义实时深度图与彩色图窗口——不过这里深度图不需要显示
    //const char* depth_win="depth_Image";
    //namedWindow(depth_win,WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);

    //深度图像颜色map
    rs2::colorizer c; //着色深度图像

    //创建数据管道
    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);

    //start()函数返回数据管道的profile
    rs2::pipeline_profile profile = pipe.start(pipe_config);

    //定义变量转换深度到距离
    float depth_clipping_distance = 1.f;

    //声明数据流
    auto depth_stream=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream=profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    //获取内参
    auto intrinDepth=depth_stream.get_intrinsics();
    auto intrinColor=color_stream.get_intrinsics();

    //直接获取从深度摄像头坐标系到彩色摄像头坐标系的欧式变换矩阵
    auto extrinDepth2Color=depth_stream.get_extrinsics_to(color_stream);

    //cvGetWindowHandle(depth_win)&&
    while (cvGetWindowHandle(color_win))  //RBG窗口没有崩溃就重复刷新帧
    {
        //堵塞程序直到新的一帧捕获
        rs2::frameset frameset = pipe.wait_for_frames();
        //取深度图和彩色图
        rs2::frame color_frame = frameset.get_color_frame();//processed.first(align_to);
        rs2::frame depth_frame = frameset.get_depth_frame();
        rs2::frame depth_frame_4_show = frameset.get_depth_frame().apply_filter(c);
        //获取宽高
        const int depth_w=depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h=depth_frame.as<rs2::video_frame>().get_height();
        const int color_w=color_frame.as<rs2::video_frame>().get_width();
        const int color_h=color_frame.as<rs2::video_frame>().get_height();

        //创建OPENCV类型 并传入数据
        Mat depth_image(Size(depth_w,depth_h),
                        CV_16U,(void*)depth_frame.get_data(),Mat::AUTO_STEP);
        Mat depth_image_4_show(Size(depth_w,depth_h),
                               CV_8UC3,(void*)depth_frame_4_show.get_data(),Mat::AUTO_STEP);
        Mat color_image(Size(color_w,color_h),
                        CV_8UC3,(void*)color_frame.get_data(),Mat::AUTO_STEP);

        //实现深度图对齐到彩色图
        Mat result=align_Depth2Color(depth_image,color_image,profile);

        //实时测距
        measure_distance(color_image,result,cv::Size(width_for_avg,height_for_avg),
                profile,X,Y,Distance_lower);

        //显示，不需要给出初始深度图跟颜色深度图，只给实时图像就行了
        //imshow(depth_win,depth_image_4_show);
        //imshow("result",result);
        imshow(color_win,color_image);

        if(Distance_lower){
            /*****
             * 距离小于0.4米的时候提示进门balabala
             * 可以对接到视觉组，也可以单独再添加弹窗什么的
             ****/
        }

        waitKey(1);
    }

    return Distance_lower;

}

#endif