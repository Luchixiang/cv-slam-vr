## 大学生创新创业项目 身临其境， 基于slam的vr采集合成小车

该项目分为三个大模块 计算机视觉， slam算法以及vr采集合成，意在实现一个可以放入一个房间，自动驶入每个房间，生成房间地图以及房间vr图的小车

![项目流程图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519234247185.png)

### 关于此项目

成果展示视频链接如下: 

- 视觉:https://www.bilibili.com/video/BV1nV411C7B8/ 
- SLAM:https://www.bilibili.com/video/bv1i54y1X7fM 
- VR: https://www.bilibili.com/video/BV1af4y1U7bE/
- VR 全景图及效果网盘下载链接:
  https://pan.baidu.com/s/1LIg6G0qZvX1uQEKMdu_c1Q
  提取码:9blo

### 计算机视觉

此次视觉部分，我们分为两个子任务

#### 门的检测

为了让小车能够进入每个房间，小车必须具备识别门的能力，第二是道路检测，小车要认识哪些可以走，哪些无法走。首先，门的检测 使用 yolov3，通过自己采集了部分门的数据进行人工标注，经过数据加强后混入coco数据集进行训练

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519234743764.png)

#### 道路检测

通过deeplabv3进行语义分割，采用deeplabv3， 用cityscape数据集进行训练

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519234724403.png)

### SLAM

#### 基于Realsense SDK与D435摄像头的实时测距

1. 仍然是获取深度像素转换到现实中长度单位（米）的换算比例，然后定义传递过来门的中心坐标(x,y)，并且定义计算平均深度需要的范围：以该坐标为中心，像四周拓展像素长度（该长度视width与height的大小而定，测试中取20*20 ）所形成的矩形框。
2. 遍历该矩形范围，对每一个点进行判断，若深度图下该点像素不为0，则是有效点，存在深度信息。将该深度距离（单位为米）累加到变量distancesum上，并统计有效像素数量effectivepixel的大小，最后二者相除，得到指定点(x,y)附近的平均深度。
3. 每获取一帧的时候判断距离是否小于0.4米，并在输出的Window窗口下实时显示、反馈深度。
   最后，考虑到本模块是为上两章节所阐述的视觉算法服务的，为了调用方便，拟将本模块打包成头文件后导入视觉算法的工程源码中，拓宽了其应用范围。

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519235024557.png)

#### 基于ORB_SLAM2框架的稀疏点阵建图

ORB_SLAM2是PTAM的继承者们非常有名的一位。它是现代SLAM系统中做的非常完善易用的系统之一，代表着主流的特征点SLAM的一个高峰。许多研究工作都以ORB_SLAM作为标准，或者在它的基础上进行后续的开发。它的代码以清晰易读著称，有完善的注释，供后来的研究者进一步们进一步理解。它是一套基于单目、双目以及RGB-D的完整方案，可以实现地图重用、回环检测以及重新定位的功能。后端主要采用BA优化方法，内部包含了一个轻量级的定位模型，实现利用VO 追踪未建图区域和与地图点匹配实现零漂移定位。

![image-20200519235217255](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519235217255.png)

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519235236005.png)

### VR

#### 基于OpenCV的全景图像拼接

1. 拍摄照片
2. sift特征点选取
3. 进行特征点匹配
4. 拼接图像

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519235439279.png)

#### 生成全景漫游文件

本功能采用全景漫游软件krpano进行全景图的三维化

![效果图](https://github.com/Luchixiang/cv-slam-vr/blob/master/img/image-20200519235531696.png)