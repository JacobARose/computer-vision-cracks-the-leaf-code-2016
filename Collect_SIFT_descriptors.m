function [descriptors, xs, ys] = Collect_SIFT_descriptors(sift_folder, data_structure, tr_idx,
num_descriptors)
% Functionality:
    % Collect a number of SIFT descriptors from the training leaf images
    % for learning the sparse coding dictionary
% Input:
    % sift_folder --- the folder where the SIFT descriptors of all leaf
    % images were saved
    % data_structure --- the structure variable with fields specifying the
    % names of leaf images and the corresponding labels

    % tr_idx --- the indices of training leaf images
    % num_descriptors --- the number of SIFT descriptors to be collected
    % for dictionary learning
% Output:
    % descriptors --- a matrix with column vectors being the SIFT
    % descriptors randomly extracted from training leaf images
    % xs --- the x coordinates of patches' centers from
    % whcih the SIFT descriptors were extracted
    % ys --- the x coordinates of patches' centers from
    % which the SIFT descriptors were extracted
num_training_images = length(tr_idx);
num_per_img = round(num_descriptors/num_training_images);
num_descriptors = num_per_img*num_training_images;

descriptors = zeros(128, num_descriptors);
xs = zeros(1, num_descriptors);
ys = zeros(1, num_descriptors);
cnt = 0;
for i=1: num_training_images
    fprintf('Extracting training samples for dictionary learning from %d/%d image\n', i,
            num_training_images);
    ind = tr_idx(i);
    [~, fname] = fileparts(data_structure.image_names{ind});
    f_sift_path = fullfile(sift_folder, [fname, '_sift.mat']);
    load(f_sift_path);
    num_fea = size(feaSet.feaArr, 2);
    rndidx = randperm(num_fea);
    num_per_img_actural = min(num_fea, num_per_img);
    descriptors(:, cnt+1:cnt+num_per_img_actural) = feaSet.feaArr(:, rndidx(1:num_per_img_actural));
    xs(:, cnt+1:cnt+num_per_img_actural) = feaSet.x(rndidx(1:num_per_img_actural));
    ys(:, cnt+1:cnt+num_per_img_actural) = feaSet.y(rndidx(1:num_per_img_actural));
    cnt = cnt+num_per_img_actural;
end
descriptors = descriptors(:, 1:cnt);
xs = xs(:, 1:cnt);
ys = ys(:, 1:cnt);