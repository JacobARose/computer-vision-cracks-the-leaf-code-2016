function C= genConfus(truth,pred,classes,bDisplay)
% Functionality:
    % Given the ground truth labels of the test samples and the predicted
    % labels by the classifier, this function return the confusion matrix
% Input:
    % truth --- The ground truth labels of the test samples
    % pred --- the predicted labels of the test samples
    % classes --- the names of the multiple classes for classification
    % bDisplay --- whether display the confusion matrix
% Output:
    % C --- the confusion matrix
Csize = length(classes);
C = zeros(Csize,Csize);
actionCount = zeros(Csize,1);

for action = 1:Csize
    Tind = find(truth == action);

    Plabel = pred(Tind);
    actionCount(action) = length(Tind);
    Punique = unique(Plabel);
    for i = 1:length(Punique)
        C(action,Punique(i)) = length(find(Plabel == Punique(i))) / length(Tind);
    end
end
d = diag(C); dd = d; dd(find(actionCount == 0 )) = [];
diagAcc = mean(dd);

if bDisplay
    displayConfus(C,actionCount,d,diagAcc,truth,pred,classes);
end

function displayConfus(C,actionCount,d,classes)
    action_truth = cell(1,length(classes));
    action_pred = cell(1,length(classes));
    for i = 1:length(classes)
        action_truth{i} = [classes{i} ' ' num2str(actionCount(i))];
        action_pred{i} = sprintf('%d',round(d(i)*100));
    end
    figure,
    imagesc(C),colorbar, xlabel('prediction'), ylabel('human');
    set(gca,'XTick',[1:length(unique(classes))],'XTickLabel',action_pred,'YTick',...
        [1:length(unique(classes))],'YTickLabel',action_truth)