function [matUs, matK] = HTDML(singleTarTrnFeaL, singleTarTrnLabelsL, indv_fund_elems, shar_fund_elems, ...
    singleSrcAuxFea, singleTarAuxFea, set, para, option)
% -------------------------------------------------------------------------
% Heterogeneous Transfer Distance Metric Learning via Knowledge Fragments
% Transfer
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Initialization
% -------------------------------------------------------------------------
% the number of knowledge fragments used for transfer
para.num_know_frag = set.num_indv_fe(1);
if ~isempty(shar_fund_elems)
    para.num_know_frag = para.num_know_frag + set.num_shar_fe;
end
matUs = cell(set.nbTarV, 1);
for vt = 1:set.nbTarV
    tarFeaDim(vt) = size(singleTarTrnFeaL{vt}, 2);
    rand('seed', vt);
    matUs{vt} = rand(tarFeaDim(vt), para.num_know_frag);
end

% ---------------------------------------------------------------------
% Compute the pairwise features and labels
% ---------------------------------------------------------------------
% fprintf('Computing the pairwise features and labels ... ');
[Delta, vecYs] = compute_pair_fea_lbl(singleTarTrnFeaL, singleTarTrnLabelsL, 0, set, para);
% [Delta, vecYs] = compute_pair_fea_lbl(singleTarTrnFeaL, singleTarTrnLabelsL, 1, set, para);
% fprintf('Finished! \n');

% ---------------------------------------------------------------------
% Calculating the fragment knowledges using the unlabeled data and
% fundamental elments
% ---------------------------------------------------------------------
if ~isempty(shar_fund_elems)
    matKS = zeros(set.num_shar_fe, size(singleSrcAuxFea{1},1));
    for vs = 1:set.nbSrcV
        if strcmp(option.ker_type, 'gbrt') == 1
            matKSv = shar_fund_elems(singleSrcAuxFea{vs}');
        elseif strcmp(option.ker_type, 'rbf') == 1 && para.ker_para <= eps
            matKSv = kernel_matrix(shar_fund_elems, singleSrcAuxFea{vs}', option.ker_type);
        else
            matKSv = kernel_matrix(shar_fund_elems, singleSrcAuxFea{vs}', option.ker_type, para.ker_para);
        end
        matKS = matKS + matKSv; clear matKSv
    end
    matKS = matKS / set.nbSrcV;
else
    matKS = [];
end
matKI = zeros(set.num_indv_fe(1), size(singleSrcAuxFea{1},1));
for vs = 1:set.nbSrcV
    if strcmp(option.ker_type, 'gbrt') == 1
        matKIv = indv_fund_elems{vs}(singleSrcAuxFea{vs}');
    elseif strcmp(option.ker_type, 'rbf') == 1 && para.ker_para <= eps
        matKIv = kernel_matrix(indv_fund_elems{vs}, singleSrcAuxFea{vs}', option.ker_type);
    else
        matKIv = kernel_matrix(indv_fund_elems{vs}, singleSrcAuxFea{vs}', option.ker_type, para.ker_para);
    end
    matKI = matKI + matKIv; clear matKIv
end
matKI = matKI / set.nbSrcV;
matK = [matKI; matKS]; clear matKI matKS

% -------------------------------------------------------------------------
% Pre-calculation
% -------------------------------------------------------------------------
fprintf('Pre-calculating ... ');
vecNormDDs = cell(set.nbTarV, 1);
normXXs = zeros(set.nbTarV, 1);
for vt = 1:set.nbTarV
    nbPw = length(vecYs{vt});
    vecNormDDs{vt} = zeros(nbPw, 1);
    for k = 1:nbPw
        vecNormDDs{vt}(k) = norm((Delta{vt}(:,k)'*Delta{vt}(:,k)), 2);
    end
    clear nbPw
    
    nbAuxFea = size(singleTarAuxFea{vt}, 1);
    vecNormXXs = zeros(nbAuxFea, 1);
    for n = 1:nbAuxFea
        vecNormXXs(n) = norm((singleTarAuxFea{vt}(n,:)*singleTarAuxFea{vt}(n,:)'), 2);
    end
    normXXs(vt) = nbAuxFea*max(vecNormXXs);
    clear vecNormXXs nbAuxFea
end
fprintf('Finished! \n');

% -------------------------------------------------------------------------
% Learning by alternating optimization
% -------------------------------------------------------------------------
for vt = 1:set.nbTarV
    pre_calc.vecNormDDv = vecNormDDs{vt};
    pre_calc.normXXv = normXXs(vt);
    matUv = matUs{vt};
    
    obj_v(1,1) = computeObj(Delta{vt}, vecYs{vt}, matUv, singleTarAuxFea{vt}', matK, para);
    loop = 1; iter = 1;
    while loop
        % ------------------------------------------------------
        % Optimize the projection matrices U{v}, v = 1, ..., V
        % ------------------------------------------------------
        [matUv_new, obj_Emp_v, obj_Omg_v] = ...
            optimizeUv_OGM(Delta{vt}, vecYs{vt}, matUv, singleTarAuxFea{vt}', matK, pre_calc, para);
        % [matUv_new, obj_Emp_v, obj_Omg_v] = ...
        %     optimizeUv_NGM(Delta{vt}, vecYs{vt}, matUv, singleTarAuxFea{vt}', matK, pre_calc, para);
        
        iter = iter + 1;
        % ------------------------------------------------------
        % Update the objective value
        % ------------------------------------------------------
        obj_v(iter,1) = computeObj_Pre(obj_Emp_v, matUv_new, singleTarAuxFea{vt}', matK, para);
        % obj_v(iter,1) = computeObj(Delta{v}, vecYs{v}, matUv_new, singleTarAuxFea{vt}', matK, para);
        
        % ------------------------------------------------------
        % Check convergence
        % ------------------------------------------------------
        % obj_diff = abs(obj(iter,1) - obj(iter-1,1)) / abs(obj(iter,1) - obj(1,1));
        % if abs(obj(iter,1) - obj(1,1)) < eps || obj_diff <= epsilon || iter >= maxit
        %     loop = 0;
        % end
        loop = checkConvergence(obj_v(iter,1), obj_v(iter-1,1), obj_v(1,1), iter-1, set, para, option);
        loop = 0;
        
        % ------------------------------------------------------
        % Update variables
        % ------------------------------------------------------
        if loop
            clear matUv
            matUv = matUv_new;
            clear matUv_new
        end
        
        clear obj_Emp_v obj_Omg_v
    end
    matUs{vt} = matUv_new; clear matUv matUv_new
    clear obj_v pre_calc
end

end


function obj = computeObj_Pre(obj_Emp_v, matUv, matXv, matK, para)

mu = para.mu;

[numKF, nbAuxFea] = size(matK);

matPhi_v = matUv' * matXv;

matTemp = matPhi_v - matK; clear matK
idx1 = find(matTemp < -mu);
idx2 = find(matTemp > mu);
idx3 = setdiff((1:(numKF*nbAuxFea))', union(idx1, idx2));

obj_Omega_temp = zeros(size(matTemp));
obj_Omega_temp(idx1) = -matTemp(idx1) - 0.5*mu;
obj_Omega_temp(idx2) = matTemp(idx2) - 0.5*mu;
obj_Omega_temp(idx3) = matTemp(idx3).^2 / (2.0*mu);

clear idx1 idx2 idx3

obj_Omega_v = para.gamma*sum(obj_Omega_temp(:));

obj = obj_Emp_v + obj_Omega_v;

end


function obj = computeObj(Delta_v, vecYv, matUv, matXv, matK, para)

mu = para.mu;
rho = para.rho;

nbPw = length(vecYv);
[numKF, nbAuxFea] = size(matK);

vecUD = cell(nbPw, 1);
vecZv = zeros(nbPw, 1);
for k = 1:nbPw
    vecUD{k} = matUv' * Delta_v(:,k);
    vecZv(k) = vecYv(k) * (1-(vecUD{k}'*vecUD{k}));
end

obj_Emp_temp = zeros(nbPw, 1);
for k = 1:nbPw
    temp_exp = exp(-rho*vecZv(k));
    if isinf(temp_exp)
        obj_Emp_temp(k) = -vecZv(k);
    else
        obj_Emp_temp(k) = (1.0/rho)*log(1.0+temp_exp);
    end
    clear temp_exp
end

clear vecUD vecZv coeff


matTemp = matUv' * matXv - matK; clear matK
idx1 = find(matTemp < -mu);
idx2 = find(matTemp > mu);
idx3 = setdiff((1:(numKF*nbAuxFea))', union(idx1, idx2));

obj_Omega_temp = zeros(size(matTemp));
obj_Omega_temp(idx1) = -matTemp(idx1) - 0.5*mu;
obj_Omega_temp(idx2) = matTemp(idx2) - 0.5*mu;
obj_Omega_temp(idx3) = matTemp(idx3).^2 / (2.0*mu);

clear idx1 idx2 idx3

obj_Emp_v = 1.0/nbPw*sum(obj_Emp_temp(:));
obj_Omega_v = para.gamma*sum(obj_Omega_temp(:));
clear obj_Emp_temp obj_Omega_temp


obj = obj_Emp_v + obj_Omega_v;

end


function [Delta, vecYs] = compute_pair_fea_lbl(singleTarTrnFeaL, singleTarTrnLabelsL, ...
    all_pair, set, para)

Delta = cell(set.nbTarV, 1); vecYs = cell(set.nbTarV, 1);

for vt = 1:set.nbTarV
    tarFeaDim(vt) = size(singleTarTrnFeaL{vt}, 2);
    if all_pair
        set.nbTarPw(vt) = (set.nbTarL(vt)*(set.nbTarL(vt)-1)) / 2;
        Delta{vt} = zeros(set.nbTarPw(vt), tarFeaDim(vt));
        vecYs{vt} = zeros(set.nbTarPw(vt), 1);
        k = 1;
        for i = 1:set.nbTarL(vt)
            for j = (i+1):set.nbTarL(vt)
                Delta{vt}(k,:) = singleTarTrnFeaL{vt}(i,:) - singleTarTrnFeaL{vt}(j,:);
                if singleTarTrnLabelsL{vt}(i) == singleTarTrnLabelsL{vt}(j)
                    vecYs{vt}(k) = 1;
                end
                k = k + 1;
            end
        end
        Delta{vt} = Delta{vt}';
        vecYs{vt}(vecYs{vt} == 0) = -1;
    else
        xTr = singleTarTrnFeaL{vt}'; yTr = singleTarTrnLabelsL{vt}';
        
        Kg = para.nbKg; Ki = 50;
        minclass = min(diff(find(diff([min(yTr)-1 sort(yTr) max(yTr)+1]))));
        if minclass <= Kg
            Kg = min(Kg, minclass-1);
            % fprintf('Warning: K too high. Setting K=%i\n',minclass-1);
        end
        clear minclass
        
        Delta{vt} = []; vecYs{vt} = [];
        [idxNN, idxIM] = lmnnFindTarsTopKImps_copy(xTr, yTr, Kg, Ki);
        if Ki == 0
            clear idxIM
            idxIM = lmnnFindImps_copy(xTr, yTr, idxNN);
        end
        for nnid = 1:Kg
            Delta{vt} = [Delta{vt} (xTr - xTr(:,idxNN(nnid,:)))];
            vecYs{vt} = [vecYs{vt} ones(1, size(idxNN,2))];
        end
        Delta{vt} = [Delta{vt} (xTr(:,idxIM(1,:)) - xTr(:,idxIM(2,:)))];
        vecYs{vt} = [vecYs{vt} -1*ones(1, size(idxIM,2))];
        
        %         Delta{vt} = []; vecYs{vt} = [];
        %         idxNN = lmnnFindTars_copy(xTr, yTr, Kg);
        %         for nnid = 1:Kg
        %             Delta{vt} = [Delta{vt} (xTr - xTr(:,idxNN(nnid,:)))];
        %             vecYs{vt} = [vecYs{vt} ones(1, size(idxNN,2))];
        %         end
        %
        %         if Ki > 0
        %             idxIM = lmnnFindTopKImps_copy(xTr, yTr, Ki);
        %         else
        %             idxIM = lmnnFindImps_copy(xTr, yTr, idxNN);
        %         end
        %         Delta{vt} = [Delta{vt} (xTr(:,idxIM(1,:)) - xTr(:,idxIM(2,:)))];
        %         vecYs{vt} = [vecYs{vt} -1*ones(1, size(idxIM,2))];
        
        vecYs{vt} = vecYs{vt}';
        clear xTr yTr idxNN idxIM
    end
end

end


function [NN, imp] = lmnnFindTarsTopKImps_copy(x, y, Kg, Ki)

[~, N] = size(x);

un = unique(y);
NN = zeros(Kg, N);
imp = zeros(2, N*Ki);
i0 = 1;

for c = un
    i = find(y==c);
    nn = LSKnn_copy(x(:,i), x(:,i), 2:Kg+1, 0);
    
    NN(:,i) = i(nn);
    
    
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


function NN = lmnnFindTars_copy(x, y, Kg)

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


function imp = lmnnFindImps_copy(Lx, y, NN)

pars.subsample = 1; % hidden feature for large data - if <1 allow to subsample constraints
pars.maximp = 500000;

Ni = sum((Lx - Lx(:,NN(end,:))).^2, 1) + 2;
un = unique(y);
imp = [];

for c = un(1:end-1)
    i = find(y==c);
    index = find(y>c);
    % experimental
    ir = randperm(length(i)); ir = ir(1:ceil(length(ir) * pars.subsample));
    ir2 = randperm(length(index)); ir2 = ir2(1:ceil(length(ir2) * pars.subsample));
    index = index(ir2);
    i = i(ir);
    % experimental
    limps = LSImps2_copy(Lx(:,index), Lx(:,i), Ni(index), Ni(i));
    if(size(limps,2) > pars.maximp)
        ip = randperm(size(limps,2));
        ip = ip(1:pars.maximp);
        limps = limps(:,ip);
    end
    imp = [imp [i(limps(2,:));index(limps(1,:))]];
end
%imp=unique(sort(imp)','rows')';

end


function limps = LSImps2_copy(X1, X2, Thresh1, Thresh2)

B = 5000;
N2 = size(X2, 2);
limps = [];
for i = 1:B:N2
    BB = min(B, N2-i);
    newlimps = findimps3Dm(X1, X2(:,i:i+BB), Thresh1, Thresh2(i:i+BB));
    if(~isempty(newlimps) && newlimps(end)==0)
        [~,endpoint] = min(min(newlimps));
        newlimps = newlimps(:, 1:endpoint-1);
    end
    newlimps = unique(newlimps', 'rows')';
    newlimps(2,:) = newlimps(2,:) + i - 1;
    limps = [limps newlimps];
end

end

