#include "depth.h"

using namespace std;
using namespace cv;


int main(){

    //定义640*480图像的中心点，这里的定义在集成的时候全部扔掉
    //然后把它变成逐帧更新的 门 中心点坐标即可
    int X = 320;
    int Y = 240;
    //定义计算平均深度的范围，集成的时候全部扔掉
    //可以更换成识别门之后的宽×高，但是计算量相应会上升
    int width = 20,height = 20;

    bool Test = getdepth(X,Y,width,height);

    return 0;
}