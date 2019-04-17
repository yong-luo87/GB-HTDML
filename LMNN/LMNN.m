function A = LMNN(trainFea, trainLabels, nbKg, nbIterMax, rDim)
% -------------------------------------------------------------------------
% Implementation of the Large Margin Nearest Neighbor Metric Learning
% -------------------------------------------------------------------------

xTr = trainFea';
yTr = trainLabels';

fprintf('\n');
[L, Details] = lmnnCG(xTr, yTr, nbKg, 'maxiter', nbIterMax, 'outdim', rDim, 'quiet', true);

A = L'*L;

end

