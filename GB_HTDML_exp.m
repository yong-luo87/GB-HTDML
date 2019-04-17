% -------------------------------------------------------------------------
% Experiments of GB-HTDML
% -------------------------------------------------------------------------

clc;
clear all;
setpaths;

option.stopdiffobj = 1;         % use difference of objective value for stopping criterion
para.seuildiffobj = 1e-3;       % stopping criterion for objective value difference
para.nbIterMax = 100;

option.ker_type = 'gbrt';       % 'lin', 'poly', 'rbf' or 'gbrt'
para.ker_para = 0;

para.mu = 0.5;
para.rho = 3;
para.nbKg = 3;
para.rDim = 50;
para.gamma = 1e-1;

% -------------------------------------------------------------------------
% Find the fundamental elements
% -------------------------------------------------------------------------
fprintf('Computing the fundamental elements ... ');
[indv_fund_elems, shar_fund_elems, num_fe] = ...
    fund_elem_calc_lmnn(singleSrcTrnFeaL, singleSrcTrnLabelsL, [], set, para, option);
set.num_indv_fe = num_fe.indv; set.num_shar_fe = num_fe.shar; clear num_fe
disp('Finished!');

% -------------------------------------------------------------------------
% Learn the linear projection matrices
% -------------------------------------------------------------------------
fprintf('Computing the embeddings ... \n');
[tarMatUs, matK] = HTDML(singleTarTrnFeaL, singleTarTrnLabelsL, indv_fund_elems, shar_fund_elems, ...
    singleSrcAuxFea, singleTarAuxFea, set, para, option); clear indv_fund_elems shar_fund_elems
[tarEmbds] = GB_HTDML(tarMatUs, singleTarTrnFeaL, singleTarTrnLabelsL, singleTarAuxFea, matK, set, para); clear tarMatUs matK
save([set.dirTemp 'Embds_s' num2str(para.seed) '_r' num2str(para.rDim) ...
    '_k' num2str(para.nbKg) '_g' num2str(log10(para.gamma)) '.mat'], 'tarEmbds', '-v7.3');
clear singleSrcTrnFeaL singleSrcTrnLabelsL
disp('Finished!');

% -------------------------------------------------------------------------
% Conduct retrieval based on the learned metric and evaluate performance
% -------------------------------------------------------------------------
fprintf('Retrieval and evaluation ... ');
MAPt = zeros(set.nbTarV, 1);
PRFt = cell(set.nbTarV, 1);
for v = 1:set.nbTarV
    tarTstFea = tarEmbds{v}(singleTarTstFea{v}')'; tarTstLabels = singleTarTstLabels{v};
    if ~isempty(tarTstLabels)
        [MAPt(v), PRFt{v}] = evaluate(tarTstFea, tarTstLabels, -1, para.nbK, [], 0);
    else
        MAPt(v) = 0; PRFt{v} = [];
    end
    clear tarTstFea tarTstLabels
end
clear tarEmbds
disp('Finished!');

