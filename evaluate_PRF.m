function PRF = evaluate_PRF(agree, numNbs)

% Compute AUC score
PRF.AUC = mlr_test_auc(agree);

% Compute MAP score
PRF.MAP = mlr_test_map(agree);

% Compute MRR score
PRF.MRR = mlr_test_mrr(agree);
PRF.MFR = mlr_test_mfr(agree);

% Compute prec@k
[PRF.PrecAtK, PRF.PrecAtKk] = mlr_test_preck(agree, numNbs);
[PRF.PrecAt1, ~] = mlr_test_preck(agree, 1);
[PRF.PrecAt10, ~] = mlr_test_preck(agree, 10);

% Compute NDCG score
[PRF.NDCG, PRF.NDCGk] = mlr_test_ndcg(agree, numNbs);

end


function [ndcg, ndcgk] = mlr_test_ndcg(Agree_tags,test_k)
trunc = 1;
nTrain = size(Agree_tags,1);

topk_scores = Agree_tags(1:test_k,:);
disc = 1./[log2(2:nTrain+1)];
if trunc == 1
    sort_norel = sort(Agree_tags,1, 'descend');
else
    sort_norel = sort(topk_scores,1, 'descend');
end

dcg = disc(1:test_k)*topk_scores;
nor = disc(1:test_k)*sort_norel(1:test_k,:);
dcgall = disc * Agree_tags;
norall = disc * sort_norel;

ndcgk = mean(dcg./(nor+eps));
ndcg = mean(dcgall./(norall+eps));

end


function [PrecAtK, PrecAtKk] = mlr_test_preck(Agree, test_k)

PrecAtK        = -Inf;
PrecAtKk       = 0;
for k = test_k
    b   = mean( mean( Agree(1:k, :), 1 ) );
    if b > PrecAtK
        PrecAtK = b;
        PrecAtKk = k;
    end
end

end

function MAP = mlr_test_map(Agree)

nTrain      = size(Agree, 1);
MAP         = bsxfun(@ldivide, (1:nTrain)', cumsum(Agree, 1));
MAP         = mean(sum(MAP .* Agree, 1)./ (sum(Agree, 1)+eps));

end

function MRR = mlr_test_mrr(Agree)

nTest = size(Agree, 2);
MRR        = 0;
for i = 1:nTest
    MRR    = MRR  + (1 / find(Agree(:,i), 1));
end
MRR        = MRR / nTest;

end

function MFR = mlr_test_mfr(Agree)

nTest = size(Agree, 2);
MFR        = 0;
for i = 1:nTest
    MFR    = MFR  + (find(Agree(:,i), 1));
end
MFR        = MFR / nTest;

end

function AUC = mlr_test_auc(Agree)

TPR             = cumsum(Agree,     1);
FPR             = cumsum(~Agree,    1);

numPos          = TPR(end,:);
numNeg          = FPR(end,:);

TPR             = mean(bsxfun(@rdivide, TPR, numPos+eps),2);
FPR             = mean(bsxfun(@rdivide, FPR, numNeg+eps),2);
AUC             = diff([0 FPR']) * TPR;

end

