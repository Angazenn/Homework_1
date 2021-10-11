# 环境
torch 1.9.0+cu111
运行于google colab

# 数据集
[下载数据集](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)。下载后使用unpickle函数读取。

# 学习率调整
<img src="https://github.com/Angazenn/files/blob/main/result.png" width="1128" height="592" >
从训练集准确度的角度的来看，当学习率较大时，准确度在初期上升的较快，但很快就不再增长；当学习率初步减小时，准确度初期上升变慢，但后期能达到更高的数值；当学习率进一步减小，准确度初期上升速率、后期结果都有明显下降。

# 结果
采用学习率0.0002，最后测试集准确度为41.4%。