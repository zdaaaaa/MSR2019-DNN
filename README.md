# MSR2019-DNN
北邮计算机学院软件安全课程期末论文代码复现，复现的论文是《Automated Software Vulnerability Assessment with Concept Drift》

data文件夹存放的是从NVD爬取的CVSS2描述及分类的原始数据，没有上传

整个网络架构用的是pytorch

dataset得到所有训练数据和测试数据

utils文件夹中存放的是网络用到的函数，封装了几个常用的函数

model文件夹下有dnn.py，也就是这次复现所用到的DNN模型

data.py是用来对得到的txt结果文件进行分析得到csv表格

main.py是主函数

通过python main.py运行即可
