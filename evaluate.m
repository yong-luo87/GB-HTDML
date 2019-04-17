function [MAP, PRF] = evaluate(fea, labels, numQry, numNbs, mat, isMet)
% -------------------------------------------------------------------------
% Randomly choose some examples for each concept as queries and evaluate
% Input
%   fea - features, nbTest x Dim matrix
%   labels - corresponding labels, nbTrain x nbC array with elements in {0,1}
%   numQry - number of query selected for each concept
%   numNbs - number of nearest neighbors adopted for performance evaluation,
%   such as Prec@k, NDCG@k etc
%   distMet - distance metric, Dim x Dim matrix
% Output
%   mAP - mean average precision over all queries
%   PERF - performance on other evaluation criteria
% -------------------------------------------------------------------------

[labels] = label_convert_mc_ml(labels);

[nbAll, nbC] = size(labels);

[DistAll, IdxAll] = sort_dbs(fea, fea, mat, isMet);

bgn = 2;        % bgn = 1 or 2 : find itself is counted or not
recall = 0.1;   % recall = 0.1 or ... 0.9 1.0
k = 1; Agree = [];
for c = 1:nbC
    idxRel = find(labels(:,c) == 1);
    idxIrr = setdiff([1:nbAll]', idxRel);
    nbRel = length(idxRel); nbIrr = length(idxIrr);
    rand('seed', c); idxRel = idxRel(randperm(nbRel));
    
    if numQry > 0
        nbQry = min(nbRel, numQry);
    else
        nbQry = nbRel;
    end
    
    Dist = DistAll(:, idxRel(1:nbQry)); Idx = IdxAll(:, idxRel(1:nbQry));
    Prob = exp(-normalize_mi(Dist, 1)); Y_dbs = labels(:,c);
    for i = 1:nbQry
        Y_dbs_i = Y_dbs; Prob_i = Prob(:,i);
        Y_dbs_i(idxRel(i)) = []; Prob_i(idxRel(i)) = [];
        if round((nbRel-1)*recall) > 0
            APs(k,1) = map_calc(Y_dbs_i, Prob_i, recall, 1); k = k + 1;
        end
        clear Y_dbs_i Prob_i
        % APs(k,1) = map_calc(Y_dbs, Prob(:,i), recall, 1); k = k + 1;
    end
    
    Y_dbs(Y_dbs == -1) = 0;
    if nbQry > 1 || bgn < 2
        Agree = [Agree Y_dbs(Idx)];
    end
    
    clear idxRel idxIrr nbRel nbIrr
    clear nbQry Dist Idx Prob Y_dbs
end

MAP = mean(APs);

Agree = Agree(bgn:end, :);
PRF = evaluate_PRF(Agree, numNbs);
% MAP = PRF.MAP;

end


function [dists, idces] = sort_dbs(query_example, dbs_examples, mat, isMet)
% -------------------------------------------------------------------------
% Sort the examples in the database according to their distance to query_example
% Input:
%   dbs_examples - database examples, N x Dim matrix
%   test_example - test example, 1 x Dim array
% Output:
%   dists - sorted distances
%   idces - sorted indices, 1 x K array
% -------------------------------------------------------------------------

[N, D] = size(dbs_examples);
[dummy, D2] = size(query_example);

if D ~= D2
    error('invalid data ..');
end

if isMet
%     test_mat = repmat(query_example, N, 1);             % N x D
%     dist_mat = (test_mat - dbs_examples);               % N x D
%     dist_array = diag(dist_mat*mat*dist_mat');     % N x 1
%     dist_array = abs(dist_array); % dist_array = real(dist_array);
%     
%     [dists, idces] = sort(dist_array);
%     % dists = dists(1:K);
%     % idces = idces(1:K);
    
    [dists, idces] = find_top_K_nbs(query_example, dbs_examples, mat);
    dists = dists'; idces =idces';
else
    [dists, idces] = test_distance(query_example', dbs_examples', mat');
end

end


function [dist_mat, neighbors] = find_top_K_nbs(test_data, train_data, dist_met)
% -------------------------------------------------------------------------
% Find top k nearest neighbors in the train_data for test_sample
% Input:
%   train_data - training data, N_trn x Dim matrix
%   test_data - test_sample, N_tst x Dim array
%   K - number of neighbors
% Output:
%   dists - least K distances
%   neighbors - K nearest neighbors, 1 x K array
% -------------------------------------------------------------------------

[N_trn, D_trn] = size(train_data);
[N_tst, D_tst] = size(test_data);

if D_trn ~= D_tst
    error('invalid data ..');
end

test_diag = diag(test_data * dist_met * test_data');
train_diag = diag(train_data * dist_met * train_data');
dist_mat = repmat(test_diag, 1, N_trn) + repmat(train_diag', N_tst, 1) ...
    - 2*(test_data * dist_met * train_data');
dist_mat = abs(dist_mat); % dist_mat = real(dist_mat);

[dists, neighbors] = sort(dist_mat, 2);
% dists = dists(:, 1:K);
% neighbors = neighbors(:, 1:K);

end


function [D, I] = test_distance(Xtest, Xtrain, L)

% CASES:
%   Raw:                        L = []
%   Low rank                    L = d-by-m
%   Linear, diagonal:           L = d-by-1

[d, nTrain, nKernel] = size(Xtrain);
nTest = size(Xtest, 2);

if isempty(L)
    % L = []  => native euclidean distances
    % D = (bsxfun(@plus, dot(Xtest,Xtest)', dot(Xtrain,Xtrain)) - 2*Xtest'*Xtrain)';
    
    sqXtrain = sum(Xtrain.^2, 1);
    sqXtest = sum(Xtest.^2, 1);
    D = bsxfun(@plus, sqXtrain.', bsxfun(@plus, sqXtest, -2*Xtrain.'*Xtest));

elseif size(L,2) == d && (size(L,1) < d || size(L,1) > d)
    % Low rank L!

    Xtrt = L * Xtrain;
    Xtet = L * Xtest;

    % D = (bsxfun(@plus, dot(Xtet,Xtet)', dot(Xtrt,Xtrt)) - 2 * Xtet'*Xtrt)';
    
    sqXtrt = sum(Xtrt.^2, 1);
    sqXtet = sum(Xtet.^2, 1);
    D = bsxfun(@plus, sqXtrt.', bsxfun(@plus, sqXtet, -2*Xtrt.'*Xtet));

elseif size(L,1) == d && size(L,2) == 1
    %diagonal
    Xtrt = bsxfun(@times,Xtrain,L);
    Xtet = bsxfun(@times,Xtest,L);
    D = (bsxfun(@plus, dot(Xtrain,Xtrt)', dot(Xtest,Xtet)) - 2 * Xtrain'*Xtet);

elseif size(L,1) == d && size(L,2) == d
    disp('treating matrix as factored form L')
    Xtrt = L * Xtrain;
    Xtet = L * Xtest;

    % D = (bsxfun(@plus, dot(Xtet,Xtet)', dot(Xtrt,Xtrt)) - 2 * Xtet'*Xtrt)';
    
    sqXtrt = sum(Xtrt.^2, 1);
    sqXtet = sum(Xtet.^2, 1);
    D = bsxfun(@plus, sqXtrt.', bsxfun(@plus, sqXtet, -2*Xtrt.'*Xtet));
    
else
    % Error?
    error('Cannot determine metric mode.');

end

[v, I]   = sort(D, 1);

end

