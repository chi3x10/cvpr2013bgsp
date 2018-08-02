function label = oneofc_inv_top(Y, class_name,topn)



[N,M] = size(Y);

%error( nargchk(1, 2, nargin));
if nargin < 2
    class_name = [];
end
if isempty(class_name)
    class_name = [1:M];
end

% idx = Y*[1:M]';
%[dummy, idx] = max(Y, [], 2);
[dummy idx] = sort(Y,2,'descend');


label = class_name(idx(:,1:topn));
label = label';