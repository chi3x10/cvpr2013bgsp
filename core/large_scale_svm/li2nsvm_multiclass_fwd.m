function [C1 C2 Y] = li2nsvm_multiclass_fwd(X, w, b, class_name)

% function [C Y] = li2nsvm_multiclass_fwd(X, w, b, class_name):
% make multi-class prediction

Y = X*w + +repmat(b,[size(X,1),1]);
C1 = oneofc_inv(Y, class_name);
% the top n candidate
C2 = oneofc_inv_top(Y,class_name,10);
% accuracy = sum(Yte==Cte)/size(Yte,1);
% fprintf('the accuracy is %f \n', accuracy);

