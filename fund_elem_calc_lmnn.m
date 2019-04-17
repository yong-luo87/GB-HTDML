function [indv_fund_elems, shar_fund_elems, num_fe] = ...
    fund_elem_calc_lmnn(singleSrcTrnFeaL, singleSrcTrnLabelsL, ori_srcIdxL, set, para, option)
% -------------------------------------------------------------------------
% Calculate the individual and shared fundamental elements by distance 
% metric learning using LMNN
% -------------------------------------------------------------------------

nbKg = para.nbKg; % 3;
nbIterMax = 100; % 1000;

indv_fund_elems = cell(set.nbSrcV, 1);
num_fe.indv = zeros(set.nbSrcV, 1);
for v = 1:set.nbSrcV
    xTr = singleSrcTrnFeaL{v}';
    yTr = singleSrcTrnLabelsL{v}';
    
    feaDim = size(xTr, 1);
    rDim = min(para.rDim, feaDim);
    
    fprintf('\n');
    [L, Details] = lmnnCG(xTr, yTr, nbKg, 'maxiter', nbIterMax, 'outdim', rDim, 'quiet', true);
    
    if para.rDim > feaDim
        L = [L; zeros(para.rDim-feaDim, feaDim)];
    end
    
    if strcmp(option.ker_type, 'gbrt')
        Embd = gb_lmnn(xTr, yTr, nbKg, L, 'ntrees', 200, 'verbose', true, 'XVAL', [], 'YVAL', []);
        indv_fund_elems{v} = Embd; clear Embd
    else
        indv_fund_elems{v} = L';
    end
    num_fe.indv(v) = size(L, 1);
    clear xTr yTr L Details
end

shar_fund_elems = [];
num_fe.shar = 0;

end

