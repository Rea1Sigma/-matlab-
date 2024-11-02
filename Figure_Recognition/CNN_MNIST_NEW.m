%% 清空环境
warning off
clc;
clear;
close all;

%% 加载 MNIST 数据集
% 使用 MATLAB 提供的内置 MNIST 数据集
[XTrain, YTrain] = digitTrain4DArrayData;  % 获取训练数据
[XTest, YTest] = digitTest4DArrayData;    % 获取测试数据

% 数据集信息
inputSize = [28, 28, 1];  % 图像尺寸
numClasses = numel(unique(YTrain));  % 类别数量

% 将标签转换为分类标签
trainLabels = categorical(YTrain);
testLabels = categorical(YTest);

%% 图像预处理
% 将图像的数值乘以255
XTrain = XTrain * 255;
XTest = XTest * 255;

% 初始化处理后的数据集
XTrainBinarized = zeros(size(XTrain), 'like', XTrain);
XTestBinarized = zeros(size(XTest), 'like', XTest);

% 对每一张图像进行二值化
for i = 1:size(XTrain, 4)
    XTrainBinarized(:,:,:,i) = imbinarize(XTrain(:,:,:,i));  % 逐张处理
end

for i = 1:size(XTest, 4)
    XTestBinarized(:,:,:,i) = imbinarize(XTest(:,:,:,i));  % 逐张处理
end

%% 使用二值化后的数据集
XTrain = XTrainBinarized;
XTest = XTestBinarized;

%% 创建CNN模型
layers = [
    imageInputLayer(inputSize, 'Name', 'input_layer')  % 输入层，28x28图像，单通道灰度图
    
    convolution2dLayer(3, 16, 'Padding', 'same')  % 卷积层，3x3滤波器，16个通道
    batchNormalizationLayer  % 批归一化
    reluLayer  % 激活层
    maxPooling2dLayer(2, 'Stride', 2)  % 池化层
    
    convolution2dLayer(3, 32, 'Padding', 'same')  % 第二个卷积层
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')  % 第三个卷积层
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

%% 设置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 5e-4, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 训练模型
% net = trainNetwork(XTrain, trainLabels, layers, options);
load("E:\模式识别\FinalReport\Figure_Recognition\CNN_Number_Test8_MNIST.mat");
%% 查看网络结构
analyzeNetwork(net);

%% 使用训练好的模型进行预测
predictedLabels = classify(net, XTest);
trueLabels = testLabels;

%% 计算并显示精确度
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Accuracy: ', num2str(100 * accuracy), '%']);

%% 生成并显示归一化混淆矩阵
figure;
confMat = confusionmat(trueLabels, predictedLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2)); % 归一化
confMatHeatmap = heatmap(confMat);

% 设置热图的 X 和 Y 数据
confMatHeatmap.XData = categories(predictedLabels);
confMatHeatmap.YData = categories(trueLabels);

% 设置标题，并调整字体大小和粗细
string = ["Normalized Confusion Matrix"; ['Accuracy = ', num2str(accuracy * 100), '%']];
confMatHeatmap.Title = string;
confMatHeatmap.TitleFontSize = 14; % 设置标题字体大小
confMatHeatmap.TitleFontWeight = 'bold'; % 设置标题字体加粗

% 设置 X 和 Y 轴标签，并调整字体大小和粗细
confMatHeatmap.XLabel = 'Predicted Labels';
confMatHeatmap.XLabelFontSize = 12; % 设置 X 轴标签字体大小
confMatHeatmap.XLabelFontWeight = 'bold'; % 设置 X 轴标签字体加粗

confMatHeatmap.YLabel = 'True Labels';
confMatHeatmap.YLabelFontSize = 12; % 设置 Y 轴标签字体大小
confMatHeatmap.YLabelFontWeight = 'bold'; % 设置 Y 轴标签字体加粗

%% 计算并显示精确度
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Accuracy: ', num2str(100 * accuracy), '%']);
