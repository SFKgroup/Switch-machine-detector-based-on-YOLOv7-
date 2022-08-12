# Readme自述文件

### 基于YOLOv7的转辙机动静节点距离检测

#### 一、项目简介

​	本项目使用图像识别的技术，定位接线柱的公头与母头，通过判断公头与母头间图像上的欧氏距离，以公头大小为参照，从而获取转辙机部件的实际间距，判断其运行状态。

#### 二、数据集介绍

​	本项目使用coco2017格式的数据集，位于项目根目录coco文件夹内。

​	数据集中图像为转辙机动静节点组部分，原始图像50张，通过亮度更改进行数据倍增。全数据集图像总计100张，JPEG格式，80张作为训练集，20张作为测试集。另从中随机选取10张进行扭曲、调色处理，加上1张完全不同的照片和其旋转处理后版本(共12张)作为结果测试数据。

#### 三、算法简述

​	本项目使用了YOLOv7算法，训练了神经网络模型，对公头和母头在图像上的位置进行定位。同时将返回的[左上坐标,右下坐标]转换为[中心点坐标,宽,高]的形式。通过公头中心点和母头box的位置关系，获取公母对应关系。之后判断对应母头位于全部母头中的上排或下排，借此获取母头根部坐标位置，与公头中心计算欧氏距离并将其除以公头宽度，消除透视影响。

​	我们还对识别结果进行进一步的筛选与过滤，如：计算box的长宽比例，对明显不合常规的box进行过滤；对明显不位于母头上的公头进行过滤；根据模型测试结果，将过滤box的置信度的阈值分开设定。根据实验，过滤阈值如下：

 公头长宽比：（0.8  , 1.2 ）

 母头长宽比：（1.65 , 2.35）

 公头置信度最小值：0.7

 母头置信度最小值：0.85

​	我们将对状态的评估根据根据运算得出的欧氏距离与公头宽度的比分为3个区间：

 **绿** : (0 , 2.5); **黄** : [2.5 , 2.9); **红** : [2.9 , +∞)

​	同时，程序根据一张图片中绿黄红的个数，将转辙机状态分为“Correct,Warn,WARN,Error”四个类别，显示于图片左上角。前两个状态被认为是正常状态，合理误差；“WARN”为有风险；“Error”为未正确连接到位。在运行结束后，程序会显示判定为“WARN”和“Error”的图像，供运维人员进行人工复核和故障诊断。

#### 四、程序输出

输出文件夹结构如下：

```
./run/detect/exp
│ all.json
│
├─img
│   test.jpg
│   …
└─json
​     test.json
​     …
```

​	all.json

```json
{
    "Correct": [9, [对应图片路径1,对应图片路径2,…,…]],
    "Error":   [2, [对应图片路径1,对应图片路径2]],
    “WARN”:    [0, []],
    “Warn”:    [1, [对应图片路径]]
}
```

​	test.json

```json
{
    "correct": 0,
    "error": 2,
    "female": [[左上坐标,右下坐标,置信度],… ,… ],
    "male":   [[左上坐标,右下坐标,置信度],… ,… ],
    "female_mp": [[中心坐标,宽,高,置信度],… ,… ],
    "male_mp":   [[中心坐标,宽,高,置信度],… ,… ],
    "warn": 1,
    "wrong": true
}
```

#### 五、修改内容

​	根据GPL v3.0协议内容，我们在此注明修改的代码：detect.py，plots.py(第62,75行)
