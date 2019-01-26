% data = readtable('CMod_HackForFusion_v2.csv');
% load('CMod_HackForFusion_v2');
% 
%data{:,width(data)+1} = ~isnan(data.time_until_disrupt);
%data.Properties.VariableNames{width(data)} = 'disrupted';
%%
load('CMod_HackForFusion_v2_with_disrupted');

%%

figure; 
histogram(data.time_until_disrupt);
xlabel('time until disruption (s)');
ylabel('counts');
grid on;

%% binning disruptions into categories

data{:,38} = zeros(size(data{:,37}));
data.Properties.VariableNames{38} = 'disrupt_category';

data.disrupt_category = zeros(size(data.disrupted));
t = 0;
dt = 0.5;
while t < max(data.time_until_disrupt)
    t = t + dt;
    data.disrupt_category = data.disrupt_category + (data.time_until_disrupt < t);
end

%% data cleaning

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
    {'qStar',[0, 30]},...
    {'radiated_fraction',[0,3]},...
    {'upper_gap',[0, 0.21]},...
    {'v_loop',[-7, 26]},...
    {'Mirnov',[0, 50]}...
    };

% trim outlier datapoints to the reasonable ranges
for i = 1:length(valid_ranges)
    temp = valid_ranges{1};
    varName = temp{1};
    varRange = temp{2};     % min and max values

    eval(['data.' varName '(data.' varName '<' num2str(varRange(1)) ') = ' num2str(varRange(1)) ';']);
    eval(['data.' varName '(data.' varName '>' num2str(varRange(2)) ') = ' num2str(varRange(2)) ';']);
end

%%
dataWithoutIntentionalDisruptions = data(data.intentional_disruption ~= 1,:);

dataWithDisruptions = data(data.disrupted,:);

%%

shotIDs = unique(data.shot);

for i = 1:10
    shotData = data(data.shot == shotIDs(i),:);
    figure;
    plot(shotData.time, shotData.ssep);
    if shotData.disrupted(1) == 1
        title(['shot ' num2str(shotIDs(i)) ' disrupted']);
    else
        title(['shot ' num2str(shotIDs(i)) ' not disrupted']);
    end
end
    
%% grab a smaller dataset
smallData = data(data.shot < shotIDs(100),:);

% test each shot separately

