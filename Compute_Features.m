function features = Compute_Features(feaSet, B, pyramid, param)
% Functionality:

    % Given the SIFT descriptors extracted from a leaf image and the
    % sparse coding dictionary, this function returns the computed
    % features from theleaf image by max pooling the sparse codign
    % responses over the dictoinary
% Input:
    % feaSet --- SIFT descriptors extracted from a leaf image
    % B --- Sparse coding dictionary
    % pyramid --- pyramid scale [0, 1, 2]
    % param --- Sparse coding parameters
% Output:
    % features --- the computed features from the input leaf image
dSize = size(B, 2);
img_width = feaSet.width;
img_height = feaSet.height;

sc_codes=mexLasso(double(feaSet.feaArr),B,param);

sc_codes = abs(sc_codes);

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

features = zeros(dSize, tBins);
features_ind = zeros(dSize, tBins);
beta_sum = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    nBins = pBins(iter1);
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);

    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    for iter2 = 1:nBins,
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end
        [features(:, bId), max_ind] = max(sc_codes(:, sidxBin), [], 2);
        features_ind(:, bId) = sidxBin(max_ind);
        beta_sum(:, bId) = sum(sc_codes(:, sidxBin), 2);
    end
end
if bId ~= tBins,
    error('Index number error!');
end
features = features(:);
features = features./sqrt(sum(features.^2));