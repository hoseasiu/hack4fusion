function [ results, allPredictions, allLabels ] = crossValidateByGroup( learningArray, varargin )
%crossValidate - Takes in a learningArray where the first three columns are
%trial number, validation group number, and correct label. The rest are
%data. Performs cross-validation on the data according to the validation
%group numbers.

% trial number is just for tracking purposes, and doesn't affect how things
% are divided up for training/testing

allPredictions = [];
allLabels = [];
group = unique(learningArray(:,2));
results = [];
for g = 1:length(group)
    if isempty(varargin)
        training = learningArray(learningArray(:,2)~=g,3:end);  % select everything but the given group
        test = learningArray(learningArray(:,2)==g,3:end);
    elseif varargin{1} == 'flip'
    % here we're flipping the groups so it's the opposite split of usual
    % cross validation (training set is much smaller than test set)
        test = learningArray(learningArray(:,2)~=g,3:end);  % select everything but the given group
        training = learningArray(learningArray(:,2)==g,3:end);
    end
    
    % replace the following line with the classification method you want to
    % use
    [trainedClassifier] = trainKNNVariableInput(training);
    
    predictions = trainedClassifier.predictFcn(test(:,2:end));
    allPredictions = [allPredictions; predictions];
    allLabels = [allLabels; test(:,1)];
    result = sum(predictions==test(:,1))/length(predictions);
%     disp(result);
    results = [results; result];
end

end
