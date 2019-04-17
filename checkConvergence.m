function loop = checkConvergence(obj_new, obj, obj_ini, iter, set, para, option)
% -------------------------------------------------------------------------
% Check if the stop criterion is reached
% -------------------------------------------------------------------------

loop = 1;
obj_diff = abs(obj_new - obj) / abs(obj);
% obj_diff = abs(obj_new - obj) / abs(obj_new - obj_ini);

% -----------------------------------------------------------
% Verbosity
% -----------------------------------------------------------
if option.verbose >= 1
    if iter == 1 || rem(iter,10) == 0
        fprintf('-------------------------------------------------\n');
        fprintf('Iter | Obj.    | Obj_new  | Obj_diff  |\n');
        fprintf('-------------------------------------------------\n');
    end;
    fprintf('%d    |%8.4f | %8.4f | %8.4f  |\n', [iter obj obj_new obj_diff]);
end
% if option.verbose >= 2
%     fprintf('Theta = ');
%     for is = 1:set.nbSrcV
%         fprintf('%.4f ', Theta_new(is));
%     end
%     fprintf('\n');
% end

% -----------------------------------------------------------
% Check difference of obj. value conditions
% -----------------------------------------------------------
if option.stopdiffobj == 1
    % if abs(obj_new - obj_ini) < eps || obj_diff < para.seuildiffobj
    if abs(obj) < eps || obj_diff < para.seuildiffobj
        loop = 0;
        fprintf(1,'obj. difference convergence criteria reached \n');
    end
end

% -----------------------------------------------------------
% Check number of iteration conditions
% -----------------------------------------------------------
if iter >= para.nbIterMax
    loop = 0;
    fprintf(1,'maximum number of iterations reached \n');
end

end

