function K = Compute_RBF_kernel(feaset_1, feaset_2, kparam)
% Functionality:
    % Compute the RBF kernel matrix between two sets of features
% Input:
    % feaset_1 --- feature matrix, each row denotes one sample
    % feaset_2 --- feature matrix, each row denotes one sample
%
% Output:
    % K --- kernel matrix
if (size(feaset_1,2) ~= size(feaset_2,2))
    error('sample1 and sample2 differ in dimensionality!!');
end

[L1, dim] = size(feaset_1);
[L2, dim] = size(feaset_2);
% If sigle parammeter, expand it.
if length(kparam) < dim
    a = sum(feaset_1.*feaset_1,2);
    b = sum(feaset_2.*feaset_2,2);
    dist2 = bsxfun(@plus, a, b' ) - 2*feaset_1*feaset_2';
    K = exp(-kparam*dist2);
else
    kparam = kparam(:);
    a = sum(feaset_1.*feaset_1.*repmat(kparam',L1,1),2);
    b = sum(feaset_2.*feaset_2.*repmat(kparam',L2,1),2);
    dist2 = bsxfun(@plus,a,b') - 2*(feaset_1.*repmat(kparam',L1,1))*feaset_2';
    K = exp(-dist2);
end

end