# **skill_accumulation_my**

## Repository Introduction

&#x1F60A;This repository retains relevant records of technology accumulation in computer vision and other related fields during my postgraduate stage, which are only used for learning purposes.

## Related Links

&#x1F60A;Welcome to visit my Zhihu homepage, Likou homepage, and CSDN blog homepage to get the latest computer vision technology blogs related to defect detection/diffusion model/comparative learning and front-end/back-end development related technology blogs.

&#x1F449;[My CSDN Blog Home Page](https://blog.csdn.net/weixin_43749999)
&#x1F449;[Zhihu homepage](https://www.zhihu.com/people/the-wang-15)
&#x1F449;[LeetCode Home Page](https://leetcode.cn/u/wyypersist)

## &#x1F4E3;News Update[Recently Posted First Show]

&#x1F449;**20230503 update: following articles are what I recently shared on my CSDN blog. If you are interested, you can take a look**

1. [使用MMDeploy（预编译包）转换MMxx(MMDeploy支持库均可)pth权重到onnx，并使用python SDK进行部署验证](https://blog.csdn.net/weixin_43749999/article/details/130307058?spm=1001.2014.3001.5501)
2. [使用MMDeploy（预编译包）转换MMxx(MMDeploy支持库均可)pth权重到onnx，并使用C++ SDK加载onnx得到dll动态链接库，实现在windows平台中调用（linux也适用）](https://blog.csdn.net/weixin_43749999/article/details/130308470?spm=1001.2014.3001.5501)
3. [图像融合方向：《Deep Image Blending》论文理解](https://blog.csdn.net/weixin_43749999/article/details/130312466?spm=1001.2014.3001.5501)
4. [图像融合方向：《GP-GAN: Towards realistic high-resolution image blending》论文理解](https://blog.csdn.net/weixin_43749999/article/details/130372603?spm=1001.2014.3001.5501)
5. [图像拼接《Leveraging Line-Point Consistence To Preserve Structures for Wide Parallax Image Stitching》论文理解](https://blog.csdn.net/weixin_43749999/article/details/130373860?spm=1001.2014.3001.5501)
6. [图像拼接方向：《Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images》论文阅读理解](https://blog.csdn.net/weixin_43749999/article/details/130375844?spm=1001.2014.3001.5501)
7. [python中使用ctypes库调用使用MMDeploy C++ SDK编译得到的dll文件时，出现WinError126的解决方法](https://blog.csdn.net/weixin_43749999/article/details/130413951?spm=1001.2014.3001.5501)
8. [《MedSegDiff Medical Image Segmentation with Diffusion Probabilistic Model》论文阅读理解](https://blog.csdn.net/weixin_43749999/article/details/129915838?spm=1001.2014.3001.5501)

<br>
<br>

### Papers and related links added from 2022.10.9 to Now

&#x261D;&#x261D;&#x261D;Please see my latest technical blog update on the top&#x261D;&#x261D;&#x261D;

<br>
<br>
## --------Previously read articles related to defect detection--------

&#x1F449;**The future will follow the latest update format shown above**&#x2757;

&#x1F449;**Attention: Simply click on the paper title to get the pdf download link**&#x2757;

### Papers and related links read before 2022.10.9

### [基于深度学习的表面缺陷检测方法综述](http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c190811)

#### Content Summary

* 介绍了表面缺陷检测中不同场景下的成像方案，又从四个方向说明了表面缺陷检测的方法：基于全监督学习、半监督、其他；
* 在全监督学习方面，作者介绍了基于表征的学习和基于度量学习的方法；
* 在基于表征的学习方法中，文章从分类网络、检测网络、分割网络三个方面进行介绍（一阶段和二阶段网络：YOLO/AlexNet/SSD/Faster
  RCNN等）；
* 在基于度量学习的方法中，作者介绍了孪生网络的原理和应用案例；
* 在无监督方面，作者介绍了正常样本学习（其中包含有具体的GAN和Otsu方法对应的具体论文），其中又分为：基于图像空间的方法和基于特征空间的方法；
* 文中还介绍了半监督和弱监督方面的知识，但尚未总结概括；
* 在文章结束部分，作者还介绍了基于深度学习的缺陷检测方法和传统的基于原始图像的方法的对比以及未来缺陷检测领域可能发展的方向；

#### Ideas

* 更换网络结构；
* 网络训练学习（引入类脑等知识指导网络训练）；
* 异域联邦学习（对于不同工业场景数据集尽心充分地利用问题）；

### [Autonomous Structural Visual Inspection Using Region-Based Deep Learning for Detecting Multiple Damage Types](https://onlinelibrary.wiley.com/doi/10.1111/mice.12334)

#### Content Summary

* 文章使用改进后的Faster R-CNN网络对多种类型的结构损伤进行检测；
* 改进点：将作为RPN网络的ZF-net进行了修改，将其最后的max-pooling和FC层使用滑动卷积层CONV代替，然后接着设置了一个深度为256层的FC层，softmax层使用softmax层和回归层代替；
* 图像采集注意点：使用单反相机拍摄图片（500*375）、GeForce GTX 1080 GPU；
* 文中作者研究了9种不同比例的27种锚组合和6种不同锚尺寸的两种尺寸组合；
* 作者还增加了softmax分类概率为正类的阈值，将原始文献中的0.6增加为0.9，将帮助提高检测的精确度；
* 没有使用缩放；
* 同时作者还使用了Faster R-CNN在视频帧上进行测试，其中视频帧率为30.0，尺寸为1920*1080；

### [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)

#### Content Summary

* 文章引入了一种新的可视化技术-反卷积，使用该技术可以详细看到卷积神经网络中特征层的功能和分类器的操作；
* 文中指出：更大训练数据集/强大的GPU/更好的模型正则化策略等促使了CNN技术的进步；
* 作者基于多层反卷积网络提出了可视化技术，将特征映射重新输入到原像素中；
* 作者还通过遮挡输入图像的部分从而判断模型对输入中的哪个部分比较灵敏；
* 文中使用的反卷积可视化技术可以对整个特征图自上而下的投影，解释了每个patch中的结构；
* 文中的一些具体点：在卷积网络中，max pooling层的操作是不可逆的，但是文中作者通过记录在一组开关变量中每个pooling区域的最大值的位置来获得一个近似的逆，在反卷积网络中，反池化操作使用这些开关将来自上一层的重建数据放置到合适的位置，同时保留激活的结构。
* 在校正部分：卷积神经网络使用relu非线性校正特征映射，从而确保特征映射总是正的。为了在每一层获得有效的特征重构(这也应该是正的)，作者通过relu非线性传递重构信号。
* 在过滤部分：卷积神经网络使用学习过的滤波器对前一层的特征映射进行卷积。反过来，反卷积网络使用相同滤波器的转置版本，但应用于校正后的地图，而不是下方图层的输出。实际上，这意味着垂直和水平地翻转每个过滤器；
* 训练细节：在ImageNet2012训练集上进行训练的（130万张图片，分布在1000多个不同的class）。每个图像经过预处理，尺寸为256*256，然后使用大小为224*224的10个不同的子作物（角度+中心，水平翻转获得），使用128个mini-batch的随机梯度下降来更新参数，从10^-2 learning rate开始，加上0.9的momentum，同时设置dropout用于fc层，概率为0.5，所有权重初始化为10^-2；偏差bias设置为0；
* 重点：对训练过程中第一层过滤器可视化显示，其中一些过滤器占主导地位，那么使用RMS将值超过10^-1的固定半径的卷积层中每个filter重新归一化；
* 训练过程中的特征演化：作者使用不同的样本，并在特定epoch的不同层次的特征图上进行可视化，得到结论：在底层的特征图中需要经过较多的epoch才可以看到收敛，但是对高层的特征，只需几个epoch就可以收敛；
* 特征变换：作者又测试了5张样本图像被使用了平移、旋转、缩放、对比模型的上下两层特征向量相对于未使用变换的变化差别，小的转换在模型的第一层有显著的效果，但在顶层特征层的影响较小，对于平移和缩放来说是准线性的。网络输出对平移和缩放是稳定的。一般来说，输出不是旋转不变的，除了旋转对称的物体(如娱乐中心)；
* 可视化之后对网络结构的选择：第一层滤波器是极高频和低频信息的混合，几乎不覆盖中频。此外，第2层可视化显示了由第1层卷积中使用的大跨度4引起的混叠效应。为了解决这些问题，作者(i)将第一层过滤器的尺寸从11x11减少到7x7， (ii)将卷积的范围从4步扩大到2步；
* 遮挡敏感性研究：作者为了进一步探究网络究竟是否识别出了物体的具体位置，还是仅仅使用了物体周围的特征作为可视化来源，采用遮挡一部分图像像素的方法，发现分类器的准确率显著下降并且在第5层中的激活强度也发生了一些变化。同时，作者还根据最顶层的特征图的可视化和图中的活动与遮挡位置的函数关系，发现当封堵器覆盖在可视化中出现的图像区域时，作者会看到特征图中活动的强烈下降。这表明可视化确实对应于刺激该特征映射的图像结构，因此验证了图4和图2中显示的其他可视化；
* 作者还进行了相关性分析：作者通过在图像中遮盖不同的部分，来判定不同图像中特定部位之间的关系；
* 文中通过修改模型的不同层次从而得到了模型的整体深度对于获得良好的性能非常重要，改变FC层对性能影响不是很大，但是增加中间CONV层的大小可以在性能上获得有用的增益；
* 特征分析：作者发现随着在更深层次的特征层上使用支持向量机等分类器进行预测，得到的准确率会逐渐增大，也就是说随着特征层次的加深，可以学习到更加强大的特征；

#### Ideas

* 使用反卷积方法进行不同特征图的可视化，可以及时发现实验设置的某些参数的正确与否，及时调整训练配置；
* 使用遮挡不同图像的不同部位来判断模型对那些部分具有较强的敏感性；
* 遮盖图像不同部分，来判定不同图像中特定部委之间存在的特定关系；
* 通过在模型不同的层次上进行预测，从而判断模型的深度对图像预测性能的影响；

### [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://link.csdn.net/?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1409.1556.pdf)

#### Content Summary

* 文章主要研究了CNN的深度对网络识别图像的影响，并提出了VGG网络；
* 网络输入设置为234*224；
* 其中的一种网络架构设置：在其中一种网络架构中，作者使用了3x3的conv层，设置stride=1，在每次conv之后，进行padding=1的零填充，保持了featuremap的分辨率固定。使用5个max pooling进行空间池化操作，池化层跟在一些conv层后边，但不是所有的conv层后边都有池化层。Max pooling层选择2x2的窗口，stride=2。
* 网络的整体结构：一系列的卷积层conv堆栈，然后紧跟着三个FC层，前两个FC层维度为4096，最后一层FC的维度为1000（对应的ImageNet数据集中1000个类），最后一层是一个softmax层（为了实现分类）；
* 作者使用了更小的conv层且使用了更深的深度，训练开始时初始化了网络的权重，使得网络的收敛速度更快；
* 训练配置：采用带动量的小批量梯度下降优化多项logistic回归目标进行，批量大小设置为batch_size=256，动量设置为0.9，通过权重衰减（L2惩罚乘数设置为5*10^-4）并在前两个FC层之间添加了dropout层（概率设置为0.5），学习率初始化为10^-2；
* 作者表示文中提出的网络可以在更少的epoch后收敛；
* 对每个图像进行多次裁剪得到多个crops，然后对于每个crop，进行随机水平反转和随机RGB颜色变换；
* 在作者的实验中，评估了在尺度为256和384下的结果；在训练尺度为384的模型时，作者使用S=256时的权重对模型进行了初始化，且设置learning_rate_init=10^-3；
* 使用多尺度进行训练：每个训练图像从一定范围随机采样S，分别进行缩放，使用固定的S=384进行预训练；
* 在网络中使用额外的非线性确实有帮助，但是使用具有非平凡接受域的CONV filters来捕获空间上下文也很重要；
* 作者又得到了结论：带有小过滤器的深网络的性能要高于带有大过滤器的浅网络；
* 对于固定尺度训练的模型，使用多尺度数据测试的结果也会表现的更好；
* 多个crop评估性能高于密集评估，但是两者是互相补充的，因为两者共同使用的时候，性能会更好于分别使用两者的情况；
* 使用网络融合将多个性能较好的模型的评估结果进行平均之后得到的结果更好；

#### Ideas

* 增加网络深度有时候可以实现更好的性能；
* 使用1*1 CONV层有时候可以提高网络的性能，这和在网络中添加额外的非线性相关；
* 可以对每个图像进行多次裁剪得到多个crops来扩充数据集，或者水平翻转和随机RGB颜色变换等；
* 多尺度训练可以提高模型的性能，同时在测试时，无论模型是使用多尺度训练还是单一固定尺度训练，都会使得测试性能增加；
* 使用多个网络预测结果的平均会使得结果更加准确；

### [An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features](https://ieeexplore.ieee.org/document/8709818)

#### Content Summary

* 本文中，作者通过使用ResNet34/50作为骨干网络对于并结合MFN将不同等级的特征图进行融合，并使用RPN网络在融合后的完整特征图上进行预测，从而形成了本文中的DNN检测系统；
* 作者在NEU-DET数据集上进行了分类和检测任务的实验，并探究了MFN网络在区域建议数量不同时和使用的IoU交并比不同时的实验结果；
* 保持CONV层前后维度一致可以使用1*1CONV层；
* 提出了一个MFN网络，将baseline网络（这里是ResNet网络）中的不同层次的特征进行融合；
* 在融合不同层次的特征时，不使用相邻的两个层，因为相邻的层具有高度的局部相关性和覆盖范围；
* 作者在文中采用了组合ResNet每个Redisual残差块的最后一层特征进行融合；
* 作者提出的MFN网络可以修改1*1CONV层的数量来减少所需要的参数，可能会影响精度，但是可以在train_data不足的情况下防止过度拟合；
* 作者比较了不同proposals的数量下，使用MFN网络和使用其他网络的模型Recall，MFN可以帮助RPN从低级和中级特征中获得位置信息，使得RPN对严格的IoU阈值有更高的容忍度；
* 同时，作者比较了对于相同的proposals来说，使用MFN和传统的RPN等在不同的IoU阈值下的测试结果，得到了MFN只需要更少的proposals就可以得到类似的性能；
* 同时，作者为了测试融合哪几层特征会使得模型的性能更好，还测试了融合不同层次特征时模型的mAP（包含L2和不包含L2）；
* 使用前置1*1 CONV来增加维度，可以减少融合之后的特征的参数量，同时使用前置1*1 CONV可以保留更完整的信息；
* DDN在ImageNet上进行了微调；

#### Ideas

* 如果需要进行特征融合，那么使用前置1*1 CONV层来统一维度比较好，可以减少特征的参数量；
* 特征融合时，不使用相邻的两个层；
* 可以修改1*1 CONV层的数量来减少模型的参数；

### [Automatic surface defect detection for mobile phone screen glass based on machine vision](http://pdf.xuebalib.com:1262/grvLJPB6UR2.pdf)

#### Content Summary

* 文章主要贡献：文中提出了一个MPSG配准算法，并提出了一种基于轮廓的配准（CR）方法生成用于对齐MPSG图像的模板图像，且采用减法和投影相结合的方法对MPSG图像进行缺陷识别，消除了环境光照波动的影响，同时为了从含噪MPSG图像中分割出具有模糊灰色边界的缺陷，本文提出了一种改进的模糊c均值聚类算法（IFCM）；
* 分类缺陷检测（正常和异常的二元分类算法）、背景重建和移除、模板参考；
* 最初使用模板匹配检测的方法，是比较有效的方法，且运算速度较快，只涉及像素的算术运算；
* 模板匹配方法中存在的问题：图像失调、周围光照变化和模糊边界缺陷分割；
* 不使用对齐算法，在图像减法过程中会导致严重的错误；如果使用多个自由缺陷图像的平均值来创建模板，这些自由缺陷图像的错位会产生不准确的模板；
* 图像配准的方法：基于灰度的方法（速度慢且对光照变化非常敏感）和基于特征的方法（轮廓是用于对齐的常见特征之一）；GMM高斯混合模型可以较好地解决光照带来的变化；
* 使用灰度投影检测法，该方法与图像周围的光照水平无关；
* 文章通过将MPSG和模板对齐之后，再采用灰度投影的方法确定缺陷的存在与否；
* 提出的MPSG自动缺陷检测系统包括三个阶段：配准、缺陷检测和分割。在第一阶段，对齐多个无缺陷MPSG图像以生成模板；然后，将测试图像与模板对齐以进行缺陷检测。在第二阶段中，从测试图像中减去模板以产生残余图像。然后，使用残差图像的灰度投影来确定缺陷的存在或不存在。如果存在缺陷，将MPSG从制造过程中移除，并将相应的数据发送到下一阶段进行缺陷分割。在第三阶段，采用改进的FCM方法对缺陷进行精确分割；
* 阈值法是从图像背景中分割物体的常用技术；

#### Ideas

* 缺陷检测可以从模板匹配方法上入手，先将原始图像和模板进行配准，然后使用像素相减和基于灰度的投影方法来判断缺陷是否存在，而后使用改进的C均值聚类算法（IFCM）来分割出具有模糊灰色边界的缺陷；
* 使用Otsu方法可以将图像转换为二值图像（只有黑色或白色的图像，没有中间过滤值）;

### [An improved Otsu method using the weighted object variance for defect detection](https://www.sciencedirect.com/science/article/pii/S0169433215011319)

#### Content Summary

* 文章提出了一种目标加权目标方差WOV（等于缺陷发生累积概率的参数根据类间方差的对象方差进行加权。）；
* 当对象和图像背景具有相似的方差时，Otsu方法可以获得满意的分割效果，但是，如果对象和图像背景的大小相差很大，该方法将失败；
* Otsu方法对于双峰分布直方图的图像阈值化提供了令人满意的结果，但对于单峰分布或接近单峰分布的图像直方图，Otsu方法失败；
* 一维Otsu方法：只考虑灰度信息而不考虑空间邻域信息的Otsu方法；
* 只有当目标和图像背景具有相似的方差时，Otsu方法才能获得满意的分割效果；
* 针对缺陷检测的自动阈值方法，如VE和NVE方法的研究主要集中在缺陷图像的阈值选择上。忽略了无缺陷图像的分割，导致大多数阈值方法都能正确地将缺陷从背景中分离出来，但却错误地将无缺陷图像归纳为检测图像。如果检测到无缺陷图像是有缺陷的图像，则会导致错误的检测。理想的视觉检测系统应具有高DR和低FAR缺陷；
* 使用自适应权重解决权重变化问题；

#### Ideas

* 对于图像分割问题来说，需要同时考虑有缺陷图像和无缺陷图像；
* 使用自适应权重来解决权重变化问题；

### [Multi-Scale Pyramidal Pooling Network for Generic Steel Defect Classification](https://ieeexplore.ieee.org/document/6706920)

#### Content Summary

* 文章提出了一个多尺度金字塔池网络，提出了一个新的多尺度金字塔池层和一个新的编码层，且可以看作一个全监督的bag-of-features的扩展；
* 传统的CV系统都会采用BoF方法，对数据集中的图像提取一组特征然后使用基于词典的技术将这些特征编码为超完备的稀疏表示；
* 有监督的ML方法尝试将基于像素的表示直接映射到标签向量，从标签数据集中学习特征提取和编码；
* 特征编码最常用的方法为：选择一个基的矢量量化（VQ）、保持小子集的稀疏编码（SC）、局部约束线性编码（LLC）等；
* 使用具有最大池化层的金字塔池层，可以使得金字塔池化层操作更加有效，且可以加快网络的学习速度；

#### Ideas

* 金字塔多尺度池化层对不同尺度大小的特征都有比较好的提取能力；
* 对特征进行再编码，使用MLPDict字典进行编码，从而得到更有效的特征表示方式；

### [Convolutional Networks for Voting-based Anomaly Classification in Metal Surface Inspection](https://ieeexplore.ieee.org/document/7915495)

#### Content Summary

* 文章提出了一种多数投票机制的SVM分类器，该SVM分类器可以融合卷积神经网络的最后三层（不包含FC层和softmax分类层）特征；
* 典型的特征提取技术包括Gabor滤波器组、尺度不变特征变换（SIFT）和模糊特征；
* 使用卷积神经网络用作小型数据集强大的特征提取器；
* 使用了多个SVM分类器：SVM LC、SVM FFC、SVM SFC，分别从LC FFC SFC层训练深层次的特征并得到不同的分类结果，然后在预测之后，进行投票机制，最后票数多的SVM得到的类别即为最后的输出类别；
* 同时，最后的结果为三种不同SVM预测结果的大多数结果，条件优先级设置为大多数卷积神经网络中具有最佳平均性能的层，基于实验作者设置了FFC作为条件优先级；

#### Ideas

* 使用多数投票机制可以使得模型的性能更加泛化；

### [TDD-net: a tiny defect detection network for printed circuit boards](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/trit.2019.0019)

#### Content Summary

* 文章提出了一种微型的缺陷网络（TDD-NET），使用金字塔层和多尺度特征提取方法，使用k-means聚类的方法得到proposals相关的锚点，加强了CONV网络中不同层次之间的联系，使用硬示例挖掘技术更好地解决了小数据集的问题；
* 使用参考方法来检测PCB表面缺陷存在测试图像和检测模板需完全对齐的难点；
* 使用密度切片、区域分割和区域过滤等典型特征处理方法可以客服CONV网络在检测小缺陷方面的弱点；
* 在线示例挖掘（OHEM）可以用于训练任何基于区域的CONVNET，自动选择硬示例；
* 使用权重衰减0.0001和动量0.9，同时30k小批量的学习率为0.001；
* 设计了消融实验来验证设计；

#### Ideas

* 金字塔层和多尺度特征提取方法结合；
* 使用k-means聚类方法在train_set上进行聚类得到锚框的位置；
* 使用OHEM（在线硬示例挖掘方法）解决了小数据集的问题；

### [A CNN-Based Defect Inspection Method for Catenary Split Pins in High-Speed Railway](https://ieeexplore.ieee.org/document/8482333)

#### Content Summary

* 文章提出了一种改进的三级SPs缺陷检测系统（PVANET++）。使用该网络和霍夫变换以及Chan-Vese模型对Firs、sPsare进行定位，然后将三个准则应用于SPs缺陷检测；且使用了一种新的方法生成锚框作为合适的候选框，并结合具有多个层的特征构造有区别的超特征；
* 利用底层自然信息，在FEN中构建更具有鉴别能力的特征；
* 利用和CSD组件的规模和形状相关联的新锚定机制，在RPN中生成更高质量的proposals；
* 分为三个阶段baseline：第一阶段，第一个PVANET++应用于原始图像，定位组件位置，然后将位置裁剪之后送入到下一个PVANET++网络；第二个网络，用于预测裁剪关节组件图像中的pin_区域，而后继续被裁剪送入到下一个PVANET++中；
* 在第三个网络中，对于A类SPs，裁剪的pin_区域1、pin_区域2和pin_区域3被发送到第三个PV ANET++以定位特定零件的头部、车身和尾部。这些特定零件的定义将在第五节中介绍。由于销U区域3中的螺栓引起的堵塞问题，采用了两种不同的标准来分别检测A1型和A2型SPs。对于B型SPs，SPs的旋转会导致二维图像中复杂的SPs状态。使用HT&CVM块定位裁剪的pin_区域4中的某些特定零件，然后应用第三个标准进行缺陷检测；

#### Ideas

* 使用多级网络实现大像素图像中的微小缺陷重复定位之后再进行检测；
* 使用和图像中特定结构相关联的锚点生成机制在RPN中生成高质量的区域建议；

### [A Generic Deep-Learning-Based Approach for Automated Surface Inspection](https://ieeexplore.ieee.org/document/7864335)

### Content Summary

* 文章提出了一种只需要少量数据进行训练的ASI检测方法，根据图像块的特征构建分类器，同时将训练好的分类器输入图像上获得像素级预测；
* ASI技术的性能取决于对缺陷特征的建模程度；
* 本文提出了一种通用的ASI方法。该方法利用预训练的DL网络提取patch特征，根据patch特征生成缺陷热图，并对热图进行阈值分割，预测缺陷区域；
* 使用Otsu方法进行二值化操作；（Otsu方法同构最小化组内方差的最佳阈值来对图像进行二值化）
* 文中还使用Felzenswalb的分割来细化缺陷区域，该方法根据像素的颜色相似性对其进行分组；

#### Ideas

* 预训练DL提取特征，并分为多个patches；
* 使用patches生成缺陷热图（HM）；
* 对热图进行阈值分割，使用图分割的方法进一步细化缺陷区域；

### [A High-Precision Positioning Approach for Catenary Support Components With Multiscale Difference](https://ieeexplore.ieee.org/document/8824211)

#### Content Summary

* 文章将粗定位网络和精细定位网络进行结合，在粗定位网络中提出了一种基于先对定位信息的无监督剧烈算法对CSC图像进行分类，然后输入到CNN中进行特征提取，并生成带有标签的建议区域；
* 在精细定位网络中，使用改进的定位框架实现对CSCs的精确定位；
* 作为一种有效的深度学习框架，提出了CSCNET来提高多尺度差分CSCs同时定位的精度和速度。为了提高定位性能，该框架利用了CSCs的相对定位信息，增强了定位方法处理接触网图像多尺度差异的鲁棒性；
* 引入了CNN分类网络来生成带有类别标签的建议区域。它在克服尺度差异和不平衡数据集方面发挥着重要作用;
* 针对CSCNET中CNN分类网络的特殊分类要求，提出了一种基于CSCs定位信息和邻接表的无监督聚类算法；
* 在粗定位网络中对图像数据集的类别进行无监督聚类时，只有当图像中存在的CSC的位置批次最接近时，才可以将两幅图像划分为一个类别；
* 使用一种无监督聚类方法对train训练集进行聚类和预测框的定位，然后使用CNN分类网络对无监督聚类的第一阶段得到的大致分类数据集进行分类，然后根据分类数据集的结果和使用无监督聚类算法得到的图片数据集和预测框之间的映射信息得到输入图像集中的预测框位置；
* 在对CNN分类网络进行充分训练后，将调整大小的悬链线图像输入卷积网络，预测43个类的置信度。置信度最高的类标签被分配给输入图像。根据类与建议区域之间的映射关系，生成12个CSC类别的粗定位结果，称为带标签的建议区域；

#### Ideas

* 使用粗定位和精细定位网络进行连接的方法进行缺陷检测；
* 利用图像中的多个待检测缺陷的结构信息；
* 使用无监督聚类算法得到训练集中的图像数据集分类，然后对每个类别得到其中的缺陷检测框的位置作为映射，对测试集图片进行分类之后，取每张图片最大可能性的标签类别使用前边的映射来得到最终的粗略的标定框预测结果；

### [A_Machine_Vision_Apparatus_and_Method_for_Can-End_Inspection](https://ieeexplore.ieee.org/document/7476878)

#### Content Summary

* 文中提出了一种结合先验形状约束的熵率聚类算法来定位罐端面目标并将其划分为多个测量区域，然后，采用超像素分组和选择方案来查找平板中央面板内部的缺陷区域，对于其他三个环形测量区域，引入了一种多尺度脊线检测算法来沿其投影轮廓寻找缺陷和变形；
* 文中采用熵率聚类算法对图像进行分割，而后将图像分割为多个互不相交的区域；
* 作者在文中对罐的中央面板的缺陷检测采用了超像素分组和选择算法，首先使用熵率聚类算法在图像中得到多个超像素，然后在超像素的基础上进行分组，然后在每个分组中使用一个特定的指数函数来评估每个区域的灰度变化（首先使用带系数的二项式滤波器对于中央面板进行平滑处理），而后计算原始图像和平滑之间的差值，之后阈值来得到缺陷区域；
* 文中提出了一种多尺度脊检测算法，作者发现缺陷的共同特性为局部的最小值或最大值，其中的阈值的选择主要基于用户的先验统计分析；
* 卷曲宽度的测量对于罐端被冲压的缺陷非常重要，使用高斯核的二阶导数与投影轮廓进行直接卷积操作，然后将两个过零点之间的距离作为卷曲宽度，然后和阈值进行比较，如果卷曲宽度大于阈值，那么表示存在缺陷；
* 活动轮廓模型可以用于定位，是一种流行的边界提取算法；

#### Ideas

* 熵率聚类算法可以对图像任意K个超像素的划分；
* 使用超像素分组和选择算法实现缺陷区域的检测；
* 在计算原始图像和带有缺陷图像的差值之前，可以使用带有二项式的滤波器对图像进行平滑处理；

### [Automated defect analysis in electron microscopic images](https://www.nature.com/articles/s41524-018-0093-8.pdf)

#### Content Summary

* 文中使用了级联目标检测器，卷积神经网络和局部图像分析方法，使用级联目标检测器作为检测模块、使用CNN作为筛选模块、使用分水岭洪水算法寻找缺陷轮廓+使用区域属性分析得到轮廓的大小信息；
* 使用在增强的数据集中训练的级联目标检测器，构建了一个有环/没有环的CNN训练集，进行训练之后的CNN对前一阶段得到的所有的bbox中是否包含有环进行分类筛选，从而进一步提高了模型的性能；

#### Ideas

* 使用CNN对BBOX进行分类也可以作为筛选目标检测器得到的结果中是否真正包含缺陷的工具；

### [Automatic classification of defective photovoltaic module cells in electroluminescence images](https://arxiv.org/abs/1807.02894v2)

#### Content Summary

* 文章首先使用手工制作的特征和支持向量机对缺陷太阳能电池进行监督分类，然后又提出了一个使用CNN的分类框架，最后文章提供了一个数据集；
* 使用SVM的特征分类方法：从电池图像中提取局部描述符，然后从局部描述符中计算全局表示（编码），然后将全局描述符分为缺陷描述符和功能描述符；
* 使用Masking掩码技术，可以划分每个图像的前景和背景部分区别开来；
* 使用关键点检测和密集采样的方法对局部特征的位置进行采样，关键点检测器依赖于图像中的纹理，因此关键点的数量与高频元素的数量成比例，且关键点方法对图像的捕捉比例是不变的，无论图像是旋转还是分辨率变化；
* 编码的目的是从多个局部描述符中形成单个固定长度的全局描述符；
* 文章使用局部聚集描述符向量（VLAD）通过对训练集中的随机特征描述符子集进行K均值聚类而创建的，最后为了使得得到的VLAD向量具有鲁棒性，需要进行归一化操作；
* 为了增强VLAD对概率k均值聚类的潜在次优解的鲁棒性，我们使用不同的随机种子从不同的训练子集计算了五个VLAD表示。然后，通过PCA算法，对VLAD编码的串联，之后再进行归一化操作；
* 文章中为了分析CNN学习的特征，还采用了t-分布随机邻域嵌入的方法（实际使用的为该方法的变体：t-SNE的巴恩斯小屋变体），这是一种用于降低维度的流形学习技术；
* 文中为了突出图像中的类特定区分区域，使用了CAM类激活图，CAM可以补充全自动评估过程，并在视觉检查期间的复杂情况下提供决策支持；
* 文中提供的参考性结论：如果图像中关键点的空间分布相当稀疏，那么使用Masking相当有用；
* 根据图像单元中缺陷可能性的置信度按比例加权样本确实提高了学习分类器的泛化能力；

#### Ideas

* 对图像的采样可以考虑使用关键点或密集采样，当图像中关键点的空间分布相当稀疏，那么使用掩码技术相当有用；
* 可以使用t-SNE相关的方法来分析CNN对特征的学习能力；
* 对图像集中得到的特征描述符可以使用VLAD方法来得到全局可以用于分类的描述符；

### [Surface Defects Detection Based on Adaptive Multiscale Image Collection and Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8661668)

#### Content Summary

* 文章首先使用ImageNet数据集对检测网络进行预训练，然后建立AMIC增强数据集，其中包括自适应多尺度图像提取和训练图像的轮廓局部提取；

#### Ideas

* 使用AMIC对数据集进行自动增强；

### [Concrete bridge surface damage detection using a single-stage detector](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12500)

#### Content Summary

* 文章使用YOLO V3网络架构进行训练，并使用迁移学习的方法初始化了网络的权重，同时还引入了Batch Normalization和Focal loss损失函数；

#### Ideas

* 使用YOLO V3检测器对图像进行训练并结合BN和FL来使得模型达到更好的性能；

### [Surface defect classification and detection on extruded aluminum profiles using convolutional neural networks](https://link.springer.com/article/10.1007/s12289-019-01496-1)

#### Content Summary

* 在图像数据集进行采集的时候就使用了多个条件：在高亮度和低亮度下，从不同方向，从近距离和远距离拍摄缺陷；
* 使用数据增强技术来使得模型的泛化能力更加强大；
* 对选择的原始网络模型的结构进行修改：对于VGG16网络，在其前两个conv层中使用stride=2而不是1，以应对更大分辨率的图像；对于GoogleNet，将第一卷积层的步长从2调整为4，将第二卷积层的步长从1调整为2；对ResNet，在第一卷积层中应用4而不是2的步幅，在第一卷积块中应用2而不是1的步幅；

#### Ideas

* 在采集图像数据集时应该就考虑到模型的泛化能力问题；
* 使用数据增强技术来使得模型的泛化能力更加强大；

### [A fast and robust convolutional neural network-based defect detection model in product quality control](https://link.springer.com/article/10.1007/s00170-017-0882-0)

#### Content Summary

* 文中提出了一种精心设计的联合检测CNN架构来实现缺陷检测，对于图像样本，首先根据背景纹理信息确定样本的类别，然后判断其是否包含缺陷区域；
* 在CNN中采用的池化层使用都是max pooling层，因为其对小失真就有鲁棒性；

#### Ideas

* 使用联合CNN架构，先对图像进行类别的确定，然后再使用CNN判断每个图像中是否包含缺陷区域；
* 在CNN中采用的池化层使用都是max pooling层，因为其对小失真就有鲁棒性；

### [Tire Defect Detection Using Fully Convolutional Network](https://ieeexplore.ieee.org/document/8678643)

#### Content Summary

* 文中提出了一个具有两个过程的缺陷检测框架，对轮胎的侧面和正面进行了缺陷检测，通过将VGG16网络的三个pooling层进行融合得到了最佳的性能；
* 详细来说：在第一阶段，使用VGG16网络进行特征提取，在该过程中将全连接-FC层改为了卷积层CONV，从而使得得到的特征输出具有足够的空间信息。在第二阶段，通过添加采样层，使用双线性插值的方法在第一阶段产生的特征向量基础上生成了与初始图像相同大小的特征层，并进行融合。之后在融合的特征上使用softmax函数预测类别分数；
* 文章提出了将FC层替换为CONV层的思想，使用全卷积神经网络进行缺陷检测pipeline；
* 将特征进行上采样之后融合；
* 通过将FC层替换为CONV层来保留特征图中相对应的空间位置信息；
* 使用双线性插值的方法来使得特征图的尺寸达到一样，然后便于特征图之间的融合；

#### Ideas

* 通过将FC层替换为CONV层，来得到FCN全卷积神经网络且保持有较多的空间位置信息；
* 可以使用双线性插值的方法来上采样；

### [Automatic pixel-level multiple damage detection of concrete structure using fully convolutional network(DOI：10.1111/mice.12433)](#)

#### Content Summary

* 文中使用FCN网络检测图像中的缺陷，同时提供了像素级的缺陷检测，使用迁移学习进行权重的初始化；
* 作者通过微调DenseNet-121构建了FCN架构；
* 为了获得更好的反卷积性能，将所有平均池化层改为了最大池化层，并将第14层中的全局平均池也替换为核大小为2×2、步长为2的最大池层，DenseNet-121的最终分类器层被丢弃，完全连接层被转换为卷积层，然后是丢失率为0.5的Dropout层，附加具有1×1内核和五个输出通道（第18、21、24、27和30层）的卷积，以预测每个先前输出位置处每个类别（裂纹、剥落、风化、孔洞和背景）的分数，然后是反卷积层，以将先前输出增加采样到像素密集输出。FCN融合了来自DenseNet-121最后一层、所有池层和第一卷积层的预测。在每个反卷积层中，通过实现步长为2的上采样，先前输出的大小增加了一倍。最后，FCN的输出大小与输入大小相同；

#### Ideas

* 使用最大池化层替换原来FCN网络中存在的平均池化层可以实现更好的反卷积性能；
* 使用反卷积操作可以实现输入和输出保持相同的大小，且反卷积输出大小和stride有关；

### [Automatic Metallic Surface Defect Detection and Recognition with Convolutional Neural Networks](https://www.mdpi.com/2076-3417/8/9/1575)

#### Content Summary

* 文章提出了一种新的CASAE网络来处理金属表面的缺陷；
* 首先文章对原始的金属表面图像进行像素级的预测，然后针对预测像素（背景/缺陷）进行裁剪，然后将裁剪得到的crops区域进行灰度化处理，保证了缺陷检测的精度，接着调整处理后的图像集到统一大小，而后将图像集输入到CNN分类网络中实现分类（CNN中使用了ATRUS卷积，有效避免了CNN的感受野过小而导致的对原始的图像中尺度较大的目标缺陷检测不到的问题）；
* AE网络广泛应用于信息编码和重建，通常AE网络包含编码器网络和解码器网络，编码器网络是一个转换单元，将输入的图像转换为多维特征图像，用于特征提取和表示；
* 解码器网络通过合并来自所有中间层中学习的特征映射的上下文信息来微调像素级标签，同时解码器网络可以使用上采样操作将最终输出恢复为输入图像相同的大小；
* 作者为了排除编码器网络受图像中不同缺陷之间的不同模糊颜色影响，使用了归一化操作将图像转换为灰度图像；
* 文章中将上采样操作的结果连接到编码器部分的相应特征映射，以获得最终的特征映射，在最后一层作者将具有softmax层的1x1卷积连接到AE网络，以将输出转换为特征图，最终的预测掩码是缺陷概率图，该图最终又被调整为和输入图像相同的大小；
* 作者使用Atrus卷积以用于增加网络的接受域，来检测较大尺度的缺陷；
* 阿托斯卷积将卷积中求和的像素隔开，但求和像素与常规卷积相同。空白中萎缩卷积的权重为零，不参与卷积运算。因此，它们的有效感受野是7×7。AE网络编码器部分中的规则卷积被带填充1和步长1的阿托斯卷积取代；
* 阈值模块的设计主要是为了进一步细化预测掩码的结果，并对概率图应用逐像素的阈值操作；
* 文章中作者在完成了对所有可能的缺陷进行分割之后，又采用blob分析的方法对缺陷轮廓进行更加精确的分割，然后根据图像中的缺陷轮廓提取了最小封闭矩形区域（MER）；
* 然后作者采用放射变换将MER转换为正MER，且将正MER区域设置为RoI区域；

#### Ideas

* 首先使用分割网络得到缺陷的像素级分割结果，然后将得到的RoI区域输入到分类网络中进行缺陷的分类；
* AE编码器和解码器网络的应用；
* 使用Atrus卷积可以增加网络的接受域，来检测图像中存在的较大尺度的缺陷；
* 可以使用blob分析对缺陷的轮廓进行更加精确的分割；
* 使用放射变换可以调整最后RoI区域的映射方向；

### [Fully Convolutional Networks for Surface Defect Inspection in IndustrialEnvironment](https://www.researchgate.net/publication/320304926_Fully_Convolutional_Networks_for_Surface_Defect_Inspection_in_Industrial_Environment)

#### Content Summary

* 作者将ZFNet网络最后的FC层修改为CONV卷积层，并使用修改后的网络对缺陷进行像素级的划分，然后对第一阶段得到的RoI区域进行采样得到多个patches作为训练样本；之后，作者使用跨层融合的特征进行检测，并分别计算了不同层特征的得分map，之后对同一个特征层的得分map进行评测，使用投票策略进行决策得到最后的类别；然后对来自两个不同特征层的两个得分map也进行了投票决定最终该patch块的类别分数；
* 文中将第一个缺陷分割阶段中缺陷区域占总面积n%以上的patches作为下一个阶段的训练样本；

#### Ideas

* 将第一个阶段得到的粗略的缺陷分割区域中间缺陷面积占比大于阈值的部分作为patches提供给第二阶段的FCN分割网络中，且在第二阶段的FCN网络中使用了特征层融合的技术；

### [A Fast Detection Method via Region-Based Fully Convolutional Neural Networks for Shield Tunnel Lining Defects](https://dl.acm.org/doi/10.1111/mice.12367)

#### Content Summary

* 文章使用FCN分割网络对原始的缺陷图像进行分割，然后在得到的feature map上使用RPN网络得到的对应的RoI区域建议框，然后在特征图上使用卷积核进行操作得到多个对应的特征图，而后对每个特征图将得到的RoI区域映射其中；
* 之后，使用RoI池化层将每个RoI区域通过大小为w/k*h/k的规则网络划分为k*k个box；

#### Ideas

* 使用FCN网络得到的缺陷分割特征Map之后使用位置敏感的RoI方法来得到的区域建议框，然后进行softmax和bounding box regression操作；

### [Deep Learning-Based Intelligent Defect Detection of Cutting Wheels with Industrial Images in Manufacturing](https://www.sciencedirect.com/science/article/pii/S2351978920315808)

#### Content Summary(ellipsis)

#### Ideas(ellipsis)

### [Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](https://arxiv.org/abs/1903.08536)

#### Content Summary

* 文章提出了一个基于分割的两阶段的缺陷检测方法，第一阶段包括像素级标签上进行训练的分割网络，第二个阶段包括在分割网络上构建的附加决策网络，来预测整个图像中是否存在异常；
* 使用Max pooling层进行下采样可以保证较小的特征在网络的前向传播过程中保留下来；
* 决策网络实际上就是利用第一阶段产生的feature map进行二值分类；
* 训练细节：首先先训练FCN分割网络，然后冻结FCN网络中的权重来训练决策网络；

#### Ideas

* 仅使用Max pooling层进行下采样可以保证较小的特征在网络的前向传播过程中保留下来；
* 使用冻结训练来处理训练两个相互连接的网络但是Loss不同的问题；

### [Detection of Rail Surface Defects Based on CNN Image Recognition and Classification](https://ieeexplore.ieee.org/document/8323642)

#### Content Summary

* 文章中作者首先将图像进行预处理将其转换为灰色二值图像，然后使用Canny边缘检测器获得边缘点并保存边缘点列，然后使用图像中轨道的比率d/I（其中d表示轨道的宽度，I表示灰度图像的宽度），通过使用上述的比率来进一步删除保存的边缘点中的假边缘点，然后，对边缘点再次进行粗略和精细去除假边缘点，然后使用线性拟合的方法对上述过程中去除的真的边缘点进行拟合恢复；
* 第二阶段中，作者使用InceptionV3网络对第一阶段中得到的图像进行分类，同时在分类的时候使用一种新的Loss函数，向交叉熵中添加了F分数公式，实现了模型的精确度和召回率的平衡；
* 文章中说明利用轨道本身的几何特征；
* 文章中使用加权平均方法将原始的彩色图像转换为灰度图像，并使用自适应中值滤波，其可以根据噪声点调整滤波窗口的大小以减少噪声；
* 作者使用提出的约束条件来使得xi自适应地缩小动态范围（其中，xi为Canny边缘检测器获得的一行中每两个相邻边缘点之间的列差）；
* 实现迁移学习的两种方法：当目标域和源域之间的差异较小时，映射相应的数学关系以扩展目标域中的数据量+当目标域和源域之间存在较大差异时，转移学习建立在卷积神经网络层上，并完成特征的迁移；
* 作者为了检测定位算法的性能，使用cropped图像和原始的图像分别输入到CNN中来进行分类，从而来对比定位处理算法的性能；

#### Ideas

* 向交叉熵Loss中中添加了F分数公式，实现了模型的精确度和召回率的平衡；
* 需要注意利用图像中不同的组织结构之间的关联关系，从而得到更加明显的特征；
* 可以使用裁剪图像生成多个cropped块来实现对图像中缺陷的定位和检测（有利于大分辨率图像中的小尺度目标的检测）；
