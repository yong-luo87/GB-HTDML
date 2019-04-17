function [Embds] = GB_HTDML(matUs, singleTarTrnFeaL, singleTarTrnLabelsL, singleTarAuxFea, matK, set, para)
% -------------------------------------------------------------------------
% Gradient Boosted Heterogeneous Transfer Distance Metric Learning via 
% Knowledge Fragments Transfer
% -------------------------------------------------------------------------

mu = para.mu;
rho = para.rho;

% [un, ~, ~] = unique(singleTarTrnLabelsL{1});
% gb_options.classes = length(un);
gb_options.K = 3;               % number of nearest neighbours
gb_options.tol = 1e-3;          % tolerance for convergence
gb_options.verbose = true;      % screen output
gb_options.depth = 4;           % tree depth
gb_options.ntrees = 200;        % number of boosted trees
gb_options.lr = 1e-3;           % learning rate for gradient boosting
gb_options.no_potential_impo = 50;
gb_options.buildlayer = @buildlayer_sqrimpurity_openmp_multi;
gb_options.Xval = [];
gb_options.Yval = [];

gb_options.K = para.nbKg;
gb_options.no_potential_impo = 10;

Embds = cell(set.nbTarV, 1);
for vt = 1:set.nbTarV
    matL = matUs{vt}';
    
    [un, ~, singleTarTrnLabelsL{vt}] = unique(singleTarTrnLabelsL{vt});
    gb_options.classes = length(un);
    
    % Xtrn = singleTarTrnFeaL{vt}'; Xaux = singleTarAuxFea{vt}';
    % predTrn = matL * Xtrn; predAux = matL * Xaux;
    % pred = [predTrn, predAux];
    
    matX = [singleTarTrnFeaL{vt}', singleTarAuxFea{vt}'];
    pred = matL * matX;
    
    vecYtrn = singleTarTrnLabelsL{vt}'; nbTrn = length(vecYtrn);
    
    if ~isempty(gb_options.Xval), % define validiation criterion for early stopping
        predVal = matL * gb_options.Xval;
        computevalmap = @(predVal) nn_search(predVal', gb_options.Yval, -1, 1, [], 0);
    else
        predVal = [];
        computevalmap = @(predVal) - 1.0;
    end
    
    % find K target neighbors
    targets_ind = lmnnFindTars_copy(singleTarTrnFeaL{vt}', vecYtrn, gb_options.K);
    % targets_ind = findtargetneighbors_copy(singleTarTrnFeaL{vt}', vecYtrn, gb_options);
    
    % sort the training input feature-wise (column-wise)
    nbSam = size(matX, 2);
    [Xs, Xi] = sort(matX');
    
    % initialize ensemble (cell array of trees)
    ensemble{1}=[];
    ensemble{2}={};
    
    % initialize the highest validation error
    highestval = 0;
    embedding = @(xTr) xTr;
    
    % initialize roll-back in case stepsize is too large
    OC = inf;
    Opred = pred;
    OpredVal = predVal;
    
    iter = 0;
    % Perform main learning iterations
    while(length(ensemble{1}) <= gb_options.ntrees)
        % Select potential imposters
        if ~rem(iter, 10)
            active = lmnnFindTopKImps_copy(pred(:,1:nbTrn), vecYtrn, gb_options.no_potential_impo);
            % active = findimpostors_copy(pred(:,1:nbTrn), vecYtrn, gb_options);
            OC = inf; % allow objective to go up
        end
        % [hinge, grad] = lmnnobj(pred, int16(targets_ind'), int16(active));
        [loss, grad] = evaluate_cost(pred, vecYtrn, matK, targets_ind, active, mu, rho, para);
        % [loss, grad] = evaluate_cost(pred, vecYtrn, matK, [], [], mu, rho, para);
        C = sum(loss);
        
        if C > OC, % roll back in case things go wrong
            C = OC;
            pred = Opred;
            predVal = OpredVal;
            % remove from ensemble
            ensemble{1}(end) = [];
            ensemble{2}(end) = [];
            if gb_options.verbose
                fprintf('Learing rate too large (%2.4e) ...\n', gb_options.lr);
            end
            gb_options.lr = gb_options.lr / 2.0;
        else % otherwise increase learning rate a little
            gb_options.lr = gb_options.lr * 1.01;
        end
        
        % Perform gradient boosting: construct trees to minimize loss
        [tree, p] = buildtree(matX', Xs, Xi, -grad', gb_options.depth, gb_options);
        clear grad
        
        % update predictions and ensemble
        Opred = pred;
        OC = C;
        OpredVal = predVal;
        pred = pred + gb_options.lr * p'; % update predictions
        iter = length(ensemble{1}) + 1;
        ensemble{1}(iter) = gb_options.lr; % add learning rate to ensemble
        ensemble{2}{iter} = tree; % add tree to ensemble
        
        % update embeding of validation data
        if ~isempty(gb_options.Xval)
            predVal = predVal + gb_options.lr * evaltree(gb_options.Xval', tree)';
        end
        clear tree p
        
        % Print out progress
        no_slack = sum(loss > 0);
        if (~rem(iter, 5) || iter == 1) && gb_options.verbose
            disp(['Iteration ' num2str(iter) ': loss is ' num2str(C ./ nbSam) ...
                ', violating inputs: ' num2str(no_slack) ', learning rate: ' num2str(gb_options.lr)]);
        end
        
        if mod(iter, 10) == 0 || iter == gb_options.ntrees,
            model.L = matL;
            model.ensemble = ensemble;
            newemb = @(xTr) evalensemble(xTr', model.ensemble, xTr'*model.L')';
            valmap = computevalmap(predVal);
            if valmap >= highestval,
                highestval = valmap;
                embedding = newemb;
                if gb_options.verbose && highestval >= 0.0
                    fprintf('Best validation map: %2.2f%%\n', highestval*100.0);
                end
            end
        end
    end
    Embds{vt} = embedding; clear ensemble embedding
    clear matL Xtrn Xaux vecYtrn predTrn predAux pred predVal
    clear computevalmap matX Xs Xi
end

end


function [obj, grad] = evaluate_cost(Phi, Ytrn, matK, targets_ind, active, mu, rho, para)
% -------------------------------------------------------------------------
% Calculate objective and gradient
% -------------------------------------------------------------------------

[rDim, nbAll] = size(Phi);
nbTrn = length(Ytrn); nbAux = nbAll - nbTrn;
PhiTrn = Phi(:, 1:nbTrn); PhiAux = Phi(:, (nbTrn+1):nbAll);

obj_Trn = []; grad_Trn = zeros(rDim, nbTrn);
% obj_Aux = zeros(1, nbAux); grad_Aux = zeros(rDim, nbAux);

n1sq = sum(PhiTrn.^2, 1); n1 = size(PhiTrn, 2);
DisTrn = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq - 2*(PhiTrn'*PhiTrn);
clear n1sq n1

if isempty(targets_ind) && isempty(active)
    k = 0;
    for i = 1:nbTrn
        for j = (i+1):nbTrn
            k = k + 1;
            if Ytrn(i) == Ytrn(j)
                z_ij = 1 - DisTrn(i,j);
            else
                z_ij = -(1 - DisTrn(i,j));
            end
            [obj_Trn, grad_Trn] = update_obj_grad(obj_Trn, grad_Trn, PhiTrn, z_ij, i, j, rho);
        end
    end
else
    k = 0;
    for i = 1:size(targets_ind,1)
        for j = 1:size(targets_ind,2)
            k = k + 1;
            z_ij = 1 - DisTrn(targets_ind(i,j), j);
            [obj_Trn, grad_Trn] = update_obj_grad(obj_Trn, grad_Trn, PhiTrn, z_ij, targets_ind(i,j), j, rho);
        end
    end
    for j = 1:size(active,2)
        z_ij = -(1 - DisTrn(active(1,j), active(2,j)));
        [obj_Trn, grad_Trn] = update_obj_grad(obj_Trn, grad_Trn, PhiTrn, z_ij, active(1,j), active(2,j), rho);
    end
end

obj_Trn = obj_Trn / (k+size(active,2));
grad_Trn = grad_Trn / (k+size(active,2));


matTemp = PhiAux - matK;
idx1 = find(matTemp < -mu);
idx2 = find(matTemp > mu);
idx3 = setdiff((1:(rDim*nbAux))', union(idx1, idx2));

matQ = zeros(size(matTemp));
matQ(idx1) = -1;
matQ(idx2) = 1;
matQ(idx3) = matTemp(idx3) ./ mu;

obj_Aux_temp = zeros(size(matTemp));
obj_Aux_temp(idx1) = -matTemp(idx1) - 0.5*mu;
obj_Aux_temp(idx2) = matTemp(idx2) - 0.5*mu;
obj_Aux_temp(idx3) = matTemp(idx3).^2 / (2.0*mu);

grad_Aux_temp = matQ;

obj_Aux = para.gamma*sum(obj_Aux_temp, 1);
grad_Aux = para.gamma*grad_Aux_temp;

clear idx1 idx2 idx3 matQ


obj = [obj_Trn, obj_Aux];
grad = [grad_Trn, grad_Aux];

end


function [obj, grad] = update_obj_grad(obj, grad, PhiTrn, z_ij, i, j, rho)

temp_exp = exp(-rho*z_ij);
if isinf(temp_exp)
    obj_k = -z_ij;
else
    obj_k = (1.0/rho)*log(1.0+temp_exp);
end
obj = [obj, obj_k];
clear temp_exp
grad_ij = 2.0*(PhiTrn(:,i)-PhiTrn(:,j)) / (1.0+exp(rho*z_ij));
grad(:,i) = grad(:,i) + grad_ij;
grad(:,j) = grad(:,j) - grad_ij;
clear grad_ij

end


function NN = lmnnFindTars_copy(x, y, Kg)

minclass = min(diff(find(diff([min(y)-1 sort(y) max(y)+1]))));
if minclass <= Kg
    Kg = min(Kg, minclass-1);
    % fprintf('Warning: K too high. Setting K=%i\n',minclass-1);
end

[~, N] = size(x);

un = unique(y);
NN = zeros(Kg, N);
for c = un
    i = find(y == c);
    nn = LSKnn_copy(x(:,i), x(:,i), 2:Kg+1, 0);
    
    NN(:,i) = i(nn);
end

end


function NN = LSKnn_copy(X1, X2, ks, gpu)

B = 750;
[~, N] = size(X2);
NN = zeros(length(ks), N);
DD = zeros(length(ks), N);
for i = 1:B:N
    BB = min(B, N-i);
    Dist = distance(X1, X2(:, i:i+BB));
    % [~,nn] = mink(Dist, max(ks));
    [~,nn] = sort(Dist);
    if gpu
        nn = gather(nn);
    end
    NN(:, i:i+BB) = nn(ks, :);
end

end


function imp = lmnnFindTopKImps_copy(x, y, Ki)

[~, N] = size(x);

un = unique(y);
imp = zeros(2, N*Ki);
i0 = 1;

for c = un
    i = find(y==c);
    
    j = find(y~=c);
    % nn = LSKnn(x(:,j), x(:,i), 2:Ki);
    % Inn(:,i) = j(nn);
    if Ki > 0
        [i1, i2] = TopKImps_copy(x(:,j), x(:,i), Ki*length(i));
        imp(:, i0:i0+length(i1)-1) = [j(i1); i(i2)];
        i0 = i0 + length(i1);
    end
end

end


function [i1, i2] = TopKImps_copy(X1, X2, M)

[~, N1] = size(X1);
[~, N2] = size(X2);
M = min(M, N1*N2);
Dist = distance(X1, X2);
[~, ii] = sort(Dist(:));
[i1, i2] = ind2sub(size(Dist), ii(1:M));

end


% function targets_ind = findtargetneighbors_copy(X, labels, options)
% 
% minclass = min(diff(find(diff([min(labels)-1 sort(labels) max(labels)+1]))));
% if minclass <= options.K
%     options.K = min(options.K, minclass-1);
%     % fprintf('Warning: K too high. Setting K=%i\n',minclass-1);
% end
% 
% [D, N] = size(X);
% targets_ind = zeros(N, options.K);
% for i = 1:options.classes
%     u = i;
%     jj = find(labels==u);
%     Xu = X(:,jj);
%     T = buildmtreemex(Xu,50);
%     targets = usemtreemex(Xu, Xu, T, options.K+1);
%     targets_ind(jj,:) = jj(targets(2:end,:))';
% end
% 
% targets_ind = targets_ind';
% 
% end
%     
%     
% function active = findimpostors_copy(pred, labels, options)
% 
% [~, N] = size(pred);
% active = zeros(options.no_potential_impo, N);
% for i = 1:options.classes
%     ii = find(labels==i);
%     pi = pred(:,ii);
%     jj = find(labels~=i);
%     pj = pred(:,jj);
%     Tj = buildmtreemex(pj,50);
%     active(:,ii) = jj(usemtreemex(pi, pj, Tj, options.no_potential_impo));
% end
% 
% end

