%average precision
%input:
%      label_gt: ground truth label, (n, 1) matrix, n is the number of images
%      prob_es: confidence map of being positive, matrix (n, 1), 
%      positive label should be 1, negative -1
%      recall: 0.1 or ... 0.9 1
%      iterwise: 1 over all elemnts, 0 through all unique elements
function ap = map_calc(label_gt, prob_es, recall, iterwise)


[img_num, class_num] = size(prob_es); 

if iterwise
    ap = 0; 
    pos_label = 1; 

    index_pos = find(label_gt==pos_label);                                % find all true positives
    [sorted_prob_pos, index] = sort(prob_es(index_pos), 'descend');
    ap_cc = 0; 
    for i=1:round(length(index_pos)*recall)                                             % over all images of class cc
        index_retrived = find(prob_es>=sorted_prob_pos(i));                % all retrieved image more confident than the ith positive 
        label_retrived = label_gt(index_retrived);
        ap_cc = ap_cc + sum(label_retrived == pos_label)/length(label_retrived); % current precision at this recall point 
    end
    ap = ap + ap_cc/round((length(index_pos)*recall));     

else 
    ap = 0; 
    pos_label = 1; 

    index_pos = find(label_gt==pos_label);                                % find all true positives
    [sorted_prob_pos, index] = sort(prob_es(index_pos), 'descend');
    unique_sort_pro_pos = unique(sorted_prob_pos);
    
    ap_cc = 0; 
    for i=1:round(length(unique_sort_pro_pos)*recall)                                             % over all images of class cc
        index_retrived = find(prob_es>=unique_sort_pro_pos(i));                % all retrieved image more confident than the ith positive 
        label_retrived = label_gt(index_retrived);
        ap_cc = ap_cc + sum(label_retrived == pos_label)/length(label_retrived) * length(find(sorted_prob_pos==unique_sort_pro_pos(i))); % current precision at this recall point 
    end
    ap = ap + ap_cc/round((length(index_pos)*recall));   
end


end


