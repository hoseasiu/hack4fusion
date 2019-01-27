% make models
clearvars; close all; clc;

classification = 1;

% load('cleanedRegressionData');
load('cleanedClassificationData');


%%
if classification
    for i = 1:length(YTest)
        YTest{i} = categorical(YTest{i});
    end
    for i = 1:length(YTrain)
        YTrain{i} = categorical(YTrain{i});
    end
end

%% define network architecture

featureDimension = size(XTrain{1},1);
numHiddenUnits = 200;

if classification
    numResponses = 2;
    layers = [ ...
        sequenceInputLayer(featureDimension)
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(50)
        dropoutLayer(0.5)
        fullyConnectedLayer(numResponses)
        softmaxLayer
        classificationLayer];
else
    numResponses = size(YTrain{1},1);
    layers = [ ...
        sequenceInputLayer(featureDimension)
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(50)
        dropoutLayer(0.5)
        fullyConnectedLayer(numResponses)
        regressionLayer];
end

maxEpochs = 60;
miniBatchSize = 1000;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',1);



%% train the network
net = trainNetwork(XTrain,YTrain,layers,options);





%% test the regression

YPred = predict(net,XTest,'MiniBatchSize',1);

%%
YPredAll = [];
YTestAll = [];
for i = 1:length(YPred)
    YPredAll = [YPredAll YPred{i}];
    YTestAll = [YTestAll YTest{i}];
end

testAccuracy = rms(YPredAll-YTestAll);

%% visualize some of the predictions
idx = randperm(numel(YPred),9);
figure
for i = 1:numel(idx)
    subplot(3,3,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("Predicted Time Until Disruption")
end
legend(["Test Data" "Predicted"],'Location','southeast')








%% select a small part of the data for faster prototyping

disp('---');

shotIDs = unique(data.shot);
smallData = data(data.shot < shotIDs(500),:);       % grab a smaller dataset (the first n shots)

disp(['size of reduced dataset: ' num2str(size(smallData,1))]);

% split into train and test by shot (with small dataset for now)
smallDataShotIDs = unique(smallData.shot);
trainTestSplitIndex = round(length(smallDataShotIDs)*0.9);
trainTestSplit = smallDataShotIDs(trainTestSplitIndex);

smallDataTrain = smallData(smallData.shot<trainTestSplit,:);
smallDataTest = smallData(smallData.shot>=trainTestSplit,:);

disp(['train: ' num2str(size(smallDataTrain,1))]);
disp(['test: ' num2str(size(smallDataTest,1))]);


shot = smallData.shot;
smallData.shot = [];
disrupted = smallData.disrupted;
smallData.disrupted = [];

% removing a few things for convenience for now - will add them back in later
time_until_disrupt = smallData.time_until_disrupt;
smallData.time_until_disrupt = [];
smallData.intentional_disruption = [];

dataArray = table2array(smallData);             % turn into table
dataArray(isnan(dataArray)) = 0;                % turn NaNs into zeros


%% train and test for classification of disrupted vs not
tic;
disp('----- training KNN -----');
% [trainedClassifier, validationAccuracy] = trainFineTreeClassifier(smallDataTrain);
[trainedClassifier, validationAccuracy] = trainKNNClassifier(smallDataTrain);       % this takes a while, but gets >90% accuracy
disp(['validation accuracy: ' num2str(validationAccuracy)]);

yFitClassification = trainedClassifier.predictFcn(smallDataTest);

testAcc = sum(yFitClassification==smallDataTest.disrupted)/length(yFitClassification);
disp(['test accuracy: ' num2str(testAcc)]);
toc;

%% regression for time till disruption
smallDataWithDisruptions = smallData(disrupted,:);
smallDataWithDisruptions.time_until_disrupt = time_until_disrupt(disrupted);


