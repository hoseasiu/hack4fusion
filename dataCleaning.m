clearvars; close all; clc;

data = readtable('CMod_HackForFusion_v2.csv');
data{:,width(data)+1} = ~isnan(data.time_until_disrupt);
data.Properties.VariableNames{width(data)} = 'disrupted';

%%
classification = 0;         % classification (1) or regression (0)
includeDisruptedData = 1;

if ~includeDisruptedData
    data(~data.disrupted,:) = [];
end

%%

% figure; 
% histogram(data.time_until_disrupt);
% xlabel('time until disruption (s)');
% ylabel('counts');
% grid on;

%% data cleaning - constrain data values to valid ranges

valid_ranges = {...
    {'Greenwald_fraction',[0,1.5]},...
    {'Te_width',[0.04, 0.5]},...
    {'Wmhd',[0, 2e5]},...
    {'beta_p',[0, 1.1]},...
    {'beta_n',[0,2]},...
    {'kappa',[0.8, 2]},...
    {'li',[0.2, 4.5]},...
    {'lower_gap',[0.025, 0.3]},...
    {'p_icrf',[0, 6e6]},...
    {'p_lh',[0, 1e6]},...
    {'p_oh',[0, 20e6]},...
    {'p_rad',[0, 20e6]},...
    {'q0',[0, 10]},...
    {'q95',[0, 20]},...
    {'qstar',[0, 30]},...
    {'radiated_fraction',[0,3]},...
    {'upper_gap',[0, 0.21]},...
    {'v_loop',[-7, 26]},...
    {'Mirnov',[0, 50]},...              % I'm adding the ones below this
    {'zcur',[-0.15 0.15]},...         
    {'z_times_v_z',[-2,10]},...
    {'z_error',[-0.1,0.1]},...
    {'v_z',[-50,50]},...
    {'ssep',[-1 1]},...
    {'n_over_ncrit',[-10,10]},...
    {'dipprog_dt',[-5e6,5e6]}...
    };

% trim outlier datapoints to the reasonable ranges
for i = 1:length(valid_ranges)
    temp = valid_ranges{i};
    varName = temp{1};
    varRange = temp{2};     % min and max values
    disp([varName '   ' num2str(varRange(1)) '   '  num2str(varRange(2))]);

    eval(['data.' varName '(data.' varName '<' num2str(varRange(1)) ') = ' num2str(varRange(1)) ';']);
    eval(['data.' varName '(data.' varName '>' num2str(varRange(2)) ') = ' num2str(varRange(2)) ';']);
end

%% plot to check variable ranges
variableNames = data.Properties.VariableNames;

% for i = 1:length(variableNames)
%     figure;
%     histogram(eval(['data.' variableNames{i}]));
%     title(variableNames{i});
%     xlabel('value');
%     ylabel('counts');
% end



%% renaming the shots for my sanity (and to make functions easier later)
originalShotIDs = unique(data.shot);

for i = 1:length(originalShotIDs)
    data.shot(data.shot==originalShotIDs(i)) = i;
end

%% separate into features and labels and other information
disrupted = data.disrupted;
time = data.time;
time_until_disrupt = data.time_until_disrupt;
shot = data.shot;
intentional_disruption = data.intentional_disruption;

% make a metadata variable
metaData = data.disrupted;
metaData = [metaData data.time data.time_until_disrupt data.shot data.intentional_disruption];

% remove the metadata from features
data = removevars(data,{'disrupted','time','time_until_disrupt','shot','intentional_disruption'});

% now "data" only contains features
featureNames = data.Properties.VariableNames;

dataArr = table2array(data);

%%
if classification
    [X,Y,shotID,time] = prepareDataTrain([shot disrupted dataArr],time);
    disp('done splitting X and Y data for classification');
else            % regression - only train on disruped data
    [X,Y,shotID,time] = prepareDataTrain([shot(disrupted) time_until_disrupt(disrupted) dataArr(disrupted,:)],time(disrupted));
    disp('done splitting X and Y data for regression');
end


%% remove training data with constant values and set NaN to zero
% m = min([X{:}],[],2);
% M = max([X{:}],[],2);
% idxConstant = M == m;

% valuesRemoved = 0;
for i = 1:numel(X)
%     valuesRemoved = valuesRemoved + 1;
%     X{i}(idxConstant,:) = [];
    X{i}(isnan(X{i})) = 0;                % set all nans to zeros
end
% disp([num2str(valuesRemoved) ' constant values removed']);

%% normalize training predictors

mu = mean([X{:}],2);
sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end
disp('normalized predictors');

%% prepare data for padding

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);
time = time(idx);
shotID = shotID(idx);

%% view sorted sequences in a bar chart
figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

%% divide into training and test
rng(42);            % to get the same train and test splits each time
[trainInd,valInd,testInd] = dividerand(length(Y),0.8,0,0.2);

XTest = X(testInd);
YTest = Y(testInd);
IDTest = shotID(testInd);
timeTest = time(testInd);
XTrain = X(trainInd);
YTrain = Y(trainInd);
IDTrain = shotID(trainInd);
timeTrain = time(trainInd);

%%
% save('cleanedData2','disrupted','featureNames','intentional_disruption','sequenceLengths','shot','time','time_until_disrupt','XTrain','YTrain');
if classification
    save('cleanedClassificationData','XTest','YTest','IDTest','timeTest','XTrain','YTrain','IDTrain','timeTrain','featureNames','originalShotIDs');
%     save('cleanedClassificationData','XTest','YTest','XTrain','YTrain','featureNames','metaData');
    disp('done getting classification data');
else
    if includeDisruptedData
        save('cleanedRegressionDataWithDisruption','XTest','YTest','IDTest','timeTest','XTrain','YTrain','IDTrain','timeTrain','featureNames','originalShotIDs');
        disp('done getting regression data with disruptions (for testing)');
    else
        save('cleanedRegressionData','XTest','YTest','IDTest','timeTest','XTrain','YTrain','IDTrain','timeTrain','featureNames','originalShotIDs');
        disp('done getting regression data without disruptions (for training)');
    end
%     save('cleanedRegressionData','XTest','YTest','XTrain','YTrain','featureNames','metaData');
    
end
   
