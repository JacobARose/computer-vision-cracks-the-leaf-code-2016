% This code is used to reproduce the recognition results reported in 
% Table 1 of the manuscript. Note that before running this code, please
% make sure you have downloaded the SIFT descritpors and put them into the
% folder 'LeafSIFT'
rand('state', 0);
randn('state', 0);

%% Parameter setting
task = 'family'; % or 'order'
threshold = 100; % or 50
maximalSide = 1024; % the maximum dimension (length or width) in pixels
split = 0.5; % the proportion of total leaf images used for training
num_bases = 1024; % the number of elements of the learned sparse coding dictionary
lc = 100; % SVM parameter

num_patches = 200000; % the number of samples for sparse coding dictionary learning
param.K = num_bases;
param.lambda = 0.15; % regularization parameter for sparse coding
param.numThreads = 4;
param.batchsize = 400;
param.iter = 1000;

pyramid = [1, 2, 4]; % spatial pyramid parameters
nRounds = 10; % the number of running
bShow = 0;

structure_dir = 'data_structure'; % the directory of the data strucure file
sift_dir = 'LeafSIFT'; % the directory of the SIFT descriptors
feature_dir = 'features'; % the directory of the features
dictionary_dir = 'dictionary'; % the directory of the dictionary
result_dir = 'results'; % the directory of the results

%% load the strucutre file for the specific recognition task
data_structure_file = sprintf('%s/%s_%d.mat', structure_dir, task, threshold);
load(data_structure_file);
if strcmp(task, 'family')
    data_structure.class_names = family_names;
elseif strcmp(task, 'order')
    data_structure.class_names = order_names;
end
data_structure.image_names = image_names;
data_structure.labels = labels;

dim_features = sum(num_bases*pyramid.^2); % the dimension of the final feature vectors fed to SVM
num_images = length(data_structure.image_names);

clabel = unique(data_structure.labels);
nclass = length(clabel);

accuracy = zeros(nRounds, 1);
for r=1: nRounds
    
    feature_folder = sprintf('%s/Exp1_Task_%s_threshold_%d_Resize_%d_split_%4.2f_numBases_%d_Round_%d', ...
                            feature_dir, task, threshold, maximalSide, split, num_bases, r);
    if ~exist(feature_folder, 'dir')
        mkdir(feature_folder);
    end
    
    dictionary_folder = sprintf('%s/Exp1_Task_%s_threshold_%d_Resize_%d_split_%4.2f_numBases_%d_Round_%d', ...
                                dictionary_dir, task, threshold, maximalSide, split, num_bases, r);
    if ~exist(dictionary_folder, 'dir')
        mkdir(dictionary_folder);
    end
    % choose training samples and test samples for this round
    samples_file = sprintf('%s/samples_idx.mat', dictionary_folder);
    if ~exist(samples_file, 'file')
        tr_idx = [];
        ts_idx = [];
        for i = 1:nclass,
            idx_label = find(data_structure.labels == clabel(i));
            num = length(idx_label);
            if split < 1
                tr_num = round(num*split);
            else
                tr_num = split;
            end
            idx_rand = randperm(num);

            tr_idx = [tr_idx idx_label(idx_rand(1:tr_num))];
            ts_idx = [ts_idx idx_label(idx_rand(tr_num+1:end))];
        end
        save(samples_file, 'tr_idx', 'ts_idx');
    else
        load(samples_file);
    end
    
    % learning sparse coding dictionary
    dict_file = [dictionary_folder '/dict.mat'];
    if ~exist(dict_file, 'file')
        [patches, xs, ys] = Collect_SIFT_descriptors(sift_dir, data_structure, tr_idx, num_patches);
        B = mexTrainDL(patches,param);
        save(dict_file, 'B');
    else
        load(dict_file);
    end
    % Compute sparse coding features
    sc_fea_all = zeros(dim_features, num_images);
    sc_label_all = data_structure.labels;
    for i=1: num_images
    
        [~, fname] = fileparts(data_structure.image_names{i});
        f_sift_path = fullfile(rt_sift_dir, [fname, '_sift.mat']);
        f_sc_fea_path = fullfile(feature_folder, [fname, '.mat']);
        if ~exist(f_sc_fea_path, 'file')
            fprintf('Compute Sparse coding features for %d/%d image\n', i, num_images);
            load(f_sift_path);
            fea = Compute_Features(feaSet, B, pyramid, param);
            save(f_sc_fea_path, 'fea');
        else
            fprintf('Load Sparse coding features for %d/%d image\n', i, num_images);
            load(f_sc_fea_path);
        end
        sc_fea_all(:, i) = fea;
    end


    % Perform classification
    tr_fea = sc_fea_all(:, tr_idx)';
    tr_label = sc_label_all(tr_idx)';
    ts_fea = sc_fea_all(:, ts_idx)';
    ts_label = sc_label_all(ts_idx)';
    
    kparam = 1;
    Ktr = Compute_RBF_kernel(tr_fea, tr_fea, kparam);
    Kte = Compute_RBF_kernel(ts_fea, tr_fea, kparam);
    Ktr = double([(1:size(Ktr,1))' Ktr]);
    Kte = double([(1:size(Kte,1))' Kte]);
    
    option = ['-t 4 -c ' num2str(lc)];
    [C, decmatrix, traintime, testtime] = libsvmova(tr_label, Ktr, ts_label, Kte, lc);
    
    leaf_frame_acc = mean(C == ts_label);
    
    confusion_matrix = genConfus(ts_label,C,data_structure.class_names,bShow);
    accuracy(r) = mean(diag(confusion_matrix));
end
% save the results
save(result_file, 'accuracy');