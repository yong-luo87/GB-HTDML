function [matUv_opt, obj_Emp_opt, obj_Omega_opt] = ...
    optimizeUv_OGM(Delta_v, vecYv, matUv, matXv, matK, pre_calc, para)
% -------------------------------------------------------------------------
% Optimization of the projection matrix using Nestrov's Optimal Gradient Method
% -------------------------------------------------------------------------

maxit = 200;
mu = para.mu;
rho = para.rho;
epsilon = 1e-3; % 1e-4;

% -------------------------------------------------------------------------
% Optimize Uv (under the non-negative constraints) using the optimal 
% gradient method (OGM) utilized in 'NeNMF, Guan et al., 2012'
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Initialization of the objective, gradient, and Lipschitz constant
% -------------------------------------------------------------------------
[obj(1,1), grad{1,1}, lipsc(1,1), obj_Emp, obj_Omega] = ...
    evaluate_cost(Delta_v, vecYv, matUv, matXv, matK, mu, rho, pre_calc, para);
matUv_guess = matUv;

loop = 1; t = 1;
while loop
    % ------------------------------------------------------
    % Solving the two auxiliary optimization problems
    % ------------------------------------------------------
    Y = matUv - 1.0/lipsc(t)*grad{t};
    tempGrad = zeros(size(grad{1}));
    for i = 1:t
        tempGrad = tempGrad + (i/2.0)*grad{i};
    end
    Z = matUv_guess - 1.0/lipsc(t)*tempGrad;
    clear tempGrad
    
    % ------------------------------------------------------
    % Calculate matUv_new
    % ------------------------------------------------------
    matUv_new = (2.0/(t+2))*Z + (t*1.0/(t+2))*Y; clear Y Z
    
    t = t + 1;
    
    % ------------------------------------------------------
    % Update the objective value, gradient, and Lipschitz
    % ------------------------------------------------------
    [obj(t,1), grad{t,1}, lipsc(t,1), obj_Emp_new, obj_Omega_new] = ...
        evaluate_cost(Delta_v, vecYv, matUv_new, matXv, matK, mu, rho, pre_calc, para);
    
    % ------------------------------------------------------
    % Check convergence
    % ------------------------------------------------------
    % obj_diff = abs(obj(t,1) - obj(t-1,1)) / abs(obj(t,1) - obj(1,1));
    % if abs(obj(t,1) - obj(1,1)) < eps || obj_diff <= epsilon || t >= maxit
    %     loop = 0;
    % end
    option.verbose = 2; option.stopdiffobj = 1;
    para.nbIterMax = maxit; para.seuildiffobj = epsilon;
    loop = checkConvergence(obj(t,1), obj(t-1,1), obj(1,1), t-1, [], para, option);
    
    % ------------------------------------------------------
    % Update variables
    % ------------------------------------------------------
    if loop
        clear matUv obj_Emp obj_Omega
        matUv = matUv_new; obj_Emp = obj_Emp_new; obj_Omega = obj_Omega_new;
        clear matUv_new obj_Emp_new obj_Omega_new
    end
end
matUv_opt = matUv_new;
obj_Emp_opt = obj_Emp_new;
obj_Omega_opt = obj_Omega_new;

end


function [obj, grad, lipsc, obj_Emp, obj_Omega] = ...
    evaluate_cost(Delta_v, vecYv, matUv, matXv, matK, mu, rho, pre_calc, para)
% -------------------------------------------------------------------------
% Compute the objective value, gradient and Lipschitz constant
% -------------------------------------------------------------------------

nbPw = length(vecYv);

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

grad_Emp_temp = zeros(size(matUv));
coeff = zeros(nbPw, 1);
for k = 1:nbPw
    coeff(k) = 2.0*vecYv(k) / (1.0+exp(rho*vecZv(k)));
    grad_Emp_temp = grad_Emp_temp + coeff(k)*(Delta_v(:,k)*vecUD{k}');
end

lipsc_Emp_temp = zeros(nbPw, 1);
for k = 1:nbPw
    lipsc_Emp_temp(k) = coeff(k)*pre_calc.vecNormDDv(k);
end

clear vecUD vecZv coeff


[numKF, nbAuxFea] = size(matK);

matTemp = matUv' * matXv - matK;
idx1 = find(matTemp < -mu);
idx2 = find(matTemp > mu);
idx3 = setdiff((1:(numKF*nbAuxFea))', union(idx1, idx2));

matQ = zeros(size(matTemp));
matQ(idx1) = -1;
matQ(idx2) = 1;
matQ(idx3) = matTemp(idx3) ./ mu;

obj_Omega_temp = zeros(size(matTemp));
obj_Omega_temp(idx1) = -matTemp(idx1) - 0.5*mu;
obj_Omega_temp(idx2) = matTemp(idx2) - 0.5*mu;
obj_Omega_temp(idx3) = matTemp(idx3).^2 / (2.0*mu);

grad_Omega_temp = matXv * matQ';

lipsc_Omega_temp = (1.0 / mu) * pre_calc.normXXv;

clear idx1 idx2 idx3 matQ

% obj_Omega_temp = norm(matUv, 'fro')^2;
% 
% grad_Omega_temp = 2.0 * matUv;
% 
% lipsc_Omega_temp = 2.0;


obj_Emp = 1.0/nbPw*sum(obj_Emp_temp(:));
obj_Omega = para.gamma*sum(obj_Omega_temp(:));
obj = obj_Emp + obj_Omega;

grad = 1.0/nbPw*grad_Emp_temp + para.gamma*grad_Omega_temp;

lipsc = max(lipsc_Emp_temp(:)) + para.gamma*lipsc_Omega_temp;

end

