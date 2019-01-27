clearvars; close all; clc;

data = readtable('CMod_HackForFusion_v2.csv');
% load('CMod_HackForFusion_v2');
% 
data{:,width(data)+1} = ~isnan(data.time_until_disrupt);
data.Properties.VariableNames{width(data)} = 'disrupted';
%%

% clearvars; close all; clc;
% 
% load('CMod_HackForFusion_v2_with_disrupted');


%%

figure; 
histogram(data.time_until_disrupt);
xlabel('time until disruption (s)');
ylabel('counts');
grid on;

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
    {'Mirnov',[0, 50]}...
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

%% renaming the shots for my sanity (and to make functions easier later)
shotIDs = unique(data.shot);

for i = 1:length(shotIDs)
    data.shot(data.shot==shotIDs(i)) = i;
end

% %%
% save('cleanedData','data');
% 
% %%
% clearvars; close all; clc;
% load('cleanedData');

%% separate into features and labels and other information
disrupted = data.disrupted;
time = data.time;
time_until_disrupt = data.time_until_disrupt;
shot = data.shot;
intentional_disruption = data.intentional_disruption;

data = removevars(data,{'disrupted','time','time_until_disrupt','shot','intentional_disruption'});

% now "data" only contains features
featureNames = data.Properties.VariableNames;

dataArr = table2array(data);

%%

altered_time_until_disrupt = time_until_disrupt;
altered_time_until_disrupt(isnan(altered_time_until_disrupt)) = 100;            % change all the non-disrupts to 100 seconds

[X,Y] = prepareDataTrain([shot altered_time_until_disrupt dataArr]);
disp('done splitting X and Y data');

%% remove training data with constant values
m = min([X{:}],[],2);
M = max([X{:}],[],2);
idxConstant = M == m;

for i = 1:numel(X)
    disp('removed a constant data value');
    X{i}(idxConstant,:) = [];
end

%% normalize training predictors

mu = mean([X{:}],2);
sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end

%% prepare data for padding

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);

%% view sorted sequences in a bar chart
figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

%% divide into training and test

[trainInd,valInd,testInd] = dividerand(length(Y),0.8,0,0.2);

XTest = X(testInd);
YTest = Y(testInd);
XTrain = X(trainInd);
YTrain = Y(trainInd);

%%
% save('cleanedData2','disrupted','featureNames','intentional_disruption','sequenceLengths','shot','time','time_until_disrupt','XTrain','YTrain');
save('cleanedData2','XTest','YTest','XTrain','YTrain');

disp('done');
