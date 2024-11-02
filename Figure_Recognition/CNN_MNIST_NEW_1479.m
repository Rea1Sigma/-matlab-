%% 清空环境
warning off
clc;
clear;
close all;

%% 加载 MNIST 数据集
% 使用 MATLAB 提供的内置 MNIST 数据集
[XTrain, YTrain] = digitTrain4DArrayData;  % 获取训练数据
[XTest, YTest] = digitTest4DArrayData;    % 获取测试数据

% 将需要筛选的数字转换为分类数组
selectedDigits = categorical([1, 4, 7, 9]);  % 要筛选的数字，转换为分类数组

% 筛选训练集和测试集中标签为 1, 4, 7, 9 的数据
trainMask = ismember(YTrain, selectedDigits);  % 训练集筛选掩码
testMask = ismember(YTest, selectedDigits);    % 测试集筛选掩码

% 筛选后的训练数据和标签
XTrain = XTrain(:,:,:,trainMask);
YTrain = YTrain(trainMask);

% 筛选后的测试数据和标签
XTest = XTest(:,:,:,testMask);
YTest = YTest(testMask);

% 重新映射标签，将 [1, 4, 7, 9] 映射为 [1, 2, 3, 4]
YTrainRemapped = double(YTrain);  % 先转换为 double 以便映射
YTestRemapped = double(YTest);    % 先转换为 double 以便映射

% 将标签为 1 的映射为 1，4 映射为 2，7 映射为 3，9 映射为 4
YTrainRemapped(YTrainRemapped == 1) = 1;
YTrainRemapped(YTrainRemapped == 4) = 2;
YTrainRemapped(YTrainRemapped == 7) = 3;
YTrainRemapped(YTrainRemapped == 9) = 4;

YTestRemapped(YTestRemapped == 1) = 1;
YTestRemapped(YTestRemapped == 4) = 2;
YTestRemapped(YTestRemapped == 7) = 3;
YTestRemapped(YTestRemapped == 9) = 4;

% 将标签转换为分类标签
trainLabels = categorical(YTrainRemapped);
testLabels = categorical(YTestRemapped);

% 更新类别数量
inputSize = [28, 28, 1];  % 图像尺寸
numClasses = 4;  % 类别数量更新为4

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
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')  % 使用更新后的类数
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

%% 设置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 5e-4, ...
    'MaxEpochs', 16, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 训练模型
net = trainNetwork(XTrain, trainLabels, layers, options);

%% 使用训练好的模型进行预测
predictedLabels = classify(net, XTest);
trueLabels = testLabels;

% 将预测的标签和真实标签进行映射回 [1, 4, 7, 9]
predictedLabelsMapped = double(predictedLabels);
trueLabelsMapped = double(trueLabels);

% 将数值1映射回1，数值2映射回4，数值3映射回7，数值4映射回9
predictedLabelsMapped(predictedLabelsMapped == 1) = 1;
predictedLabelsMapped(predictedLabelsMapped == 2) = 4;
predictedLabelsMapped(predictedLabelsMapped == 3) = 7;
predictedLabelsMapped(predictedLabelsMapped == 4) = 9;

trueLabelsMapped(trueLabelsMapped == 1) = 1;
trueLabelsMapped(trueLabelsMapped == 2) = 4;
trueLabelsMapped(trueLabelsMapped == 3) = 7;
trueLabelsMapped(trueLabelsMapped == 4) = 9;

%% 生成并显示归一化混淆矩阵
figure;
confMat = confusionmat(trueLabelsMapped, predictedLabelsMapped);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2)); % 归一化
confMatHeatmap = heatmap(confMat);
confMatHeatmap.XData = {'1', '4', '7', '9'};
confMatHeatmap.YData = {'1', '4', '7', '9'};
confMatHeatmap.Title = 'Normalized Confusion Matrix';
confMatHeatmap.XLabel = 'Predicted Labels';
confMatHeatmap.YLabel = 'True Labels';

%% 计算并显示精确度
accuracy = sum(predictedLabelsMapped == trueLabelsMapped) / numel(trueLabelsMapped);
disp(['Accuracy: ', num2str(100 * accuracy), '%']);