# NLP实践-中文预训练模型泛化能力挑战赛（文本分类，bert）专题 学习笔记

![avatar](/image/env-flow.png?raw=true)

- [Docker 安装与使用](#docker------)
- [阿里云开通镜像仓库](#---------)
- [获取baseline](#--baseline)
- [配置代码环境](#------)
- [本机运行并提交](#-------)

> 配置
>
> - 操作系统：windows10家庭中文版  
> - 显卡：NVIDIA GeForce GTX 1060  
> - 环境：pytorch1.6.0（GPU版）+ CUDA10.2 + VSCode + windows版Docker
>

## Docker 安装与使用

首先了解Docker是什么及安装方法，还有简单常用的命令了解一下。参考博客：https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html

或者可以看dataWhale的Docker环境配置指南，里面包含了Docker安装，创建镜像仓库及提交到天池：https://mp.weixin.qq.com/s/JiimSmuD3S5lSS9MmH2GJw

在Windows安装时，注意可能需要安装WSL2更新。

![avatar](/image/wsl2.png?raw=true)

如果出现提示，可以点进链接按照步骤操作：

https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-4---download-the-linux-kernel-update-package

安装好docker以后，可以使用cmd 命令或 PowerShell 对docker进行操作，注意Windows里使用docker的命令，前面不需要加sudo, 直接docker...就好。

安装完成，测试一下能用即可，然后继续下面的步骤。

## 阿里云开通镜像仓库

注册阿里云容器镜像服务  
参考教程：https://tianchi.aliyun.com/competition/entrance/231759/tab/226  
这个讲的挺详细了，在创建完自己的镜像仓库以后，再登陆进来，记得需要选一下地址才能看到：

![avatar](/image/alicould.png?raw=true)

点进去以后，可以看到下面有一些自己镜像仓库常用的命令，比如登录，推送等。如果是在windows里使用，把命令前的sudo去掉就行了

![avatar](/image/aliregistry.png?raw=true)

## 获取baseline

来源地址：https://github.com/finlay-liu/tianchi-multi-task-nlp

在本地安装git，把这个库里的代码clone下来。这里附上一个之前总结的git bash常用使用命令流程
![avatar](/image/git-flow.png?raw=true)

## 配置代码环境

首先看一下自己电脑上有没有CUDA，是什么版本，然后再开始安装对应支持的pytorch版本，这个比较重要。

## 本机运行并提交

按照baseline上的流程，把文件下载下来放在几个文件夹里，然后开始顺序运行generate_data.py -> train.py -> inference.py 几个文件。

我在运行的过程中遇到了 CUDA out of memory的错误。

![avatar](/image/batchsize.png?raw=true)

可以把执行train函数里的参数 batch_size 调小一点。同时，如果电脑配置不高，希望快点出结果的话，可以把epoch调少一点，比如1或者2。 

```python
train(epochs=2, batchSize=6, device='cuda:0', load_saved=True ,a_step=16, lr=0.0001,  pretrained_model=pretrained_model, tokenizer_model=tokenizer_model, weighted_loss=True)
```

在运行的过程中，因为我把batchsize改小了，出现了一个bug：

![avatar](/image/cur_error.png?raw=true)

去仔细看了下代码，出现bug这个原因是为什么呢？在get_next_batch()函数里面， 当总剩余数据量>0时，只分别给出了几个分数据集剩余数据量>0的情况，没有考虑到分数据集剩余数据量=0的情况。所以当总数据集长度不为0，但某个数据集长度已经为0的情况下，tnews_cur就没有在引用前被定义，就会报错。
![avatar](/image/curfuc_raw.png?raw=true)

这里附上修改后的elif部分的逻辑判断，改完以后，就可以正常运行了

```python
elif total_len > batchSize:
    if ocnli_len > 0:
        ocnli_tmp_len = int((ocnli_len / total_len) * batchSize)
        ocnli_cur = self.ocnli_ids[:ocnli_tmp_len]
        self.ocnli_ids = self.ocnli_ids[ocnli_tmp_len:]
    elif ocnli_len == 0:
        ocnli_cur = self.ocnli_ids
        self.ocnli_ids = []
    if ocemotion_len > 0:
        ocemotion_tmp_len = int((ocemotion_len / total_len) * batchSize)
        ocemotion_cur = self.ocemotion_ids[:ocemotion_tmp_len]
        self.ocemotion_ids = self.ocemotion_ids[ocemotion_tmp_len:]
    elif ocemotion_len ==0:
        ocemotion_cur = self.ocemotion_ids
        self.ocemotion_ids = []
    if tnews_len > 0:
        tnews_tmp_len = batchSize - len(ocnli_cur) - len(ocemotion_cur)
        tnews_cur = self.tnews_ids[:tnews_tmp_len]
        self.tnews_ids = self.tnews_ids[tnews_tmp_len:]
    elif tnews_len == 0:
        tnews_cur = self.tnews_ids
        self.tnews_ids = []

```

一些概念理解：
把所有样本都跑一遍，就叫完成一期训练，是一个epoch。batch size是一次训练所抓取的数据样本数量。在这个项目中，我们三个数据集总数据量大概为143443条数据，如果设置batch size=4, 则一共要跑143443/4=35860个batch。那么我们一个epoch结束，看打印出来的batch数量就差不多是35000 th了。

![avatar](/image/batch_num.png?raw=true)

最后生成结果以后，需要在文件夹里打包一下几个结果json文件。按照比赛提交要求命名。

![avatar](/image/submission.png?raw=true)

在PowerShell 或者 CMD命令窗口中，将目录切换到当前submission目录下，按照basline里面的说明，登录阿里云镜像仓库（把阿里云镜像仓库的网页打开，里面会用到的命令），构建镜像，提交（tag和push）镜像到远端。注意版本号是自己取，1.0 2.0之类的。

![avatar](/image/push.png?raw=true)

最后在比赛提交页面，填写镜像路径+版本号，以及用户名和密码。稍微等一会就可以看成绩啦。

![avatar](/image/score.png?raw=true)

## 模型提升

本机配置太差，batchsize不能超过6，最开始效果一直不好。提升了一点，主要是增大了a_step参数。

![avatar](/image/improve.png?raw=true)

## 其他

### 服务器运行踩坑

因为本机配置不够高，去租了服务器，涉及到一些Linux文件运行问题，特记录一下：

1.文件传输

可以用Xftp,官网有个人版/学生版 是免费使用的，非常方便，可以去官网下载

2.在Linux下执行python脚本文件

参阅：https://blog.csdn.net/qq_28267025/article/details/60337293

如果出现类似于这种报错:

```python
/usr/bin/env: ‘python3\r’: No such file or directory
```

可以参阅这篇解答：

https://askubuntu.com/questions/896860/usr-bin-env-python3-r-no-such-file-or-directory