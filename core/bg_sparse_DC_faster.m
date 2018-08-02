function A = bg_sparse_DC_faster(D,X,A,lambda,blocks,nIter)
% This function solve the following problem
%
%   min_C Q(A; D, X) = 1/2 | X -DA |_F^2 + \lambda sum_i |D[i]A[i]|_F 
%
% -----------------------------------------------------------------------
%
% Copyright (2012): Yu-Tseh Chi and Jeffrey Ho
%
% GPSR is distributed under the terms
% of the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
%dictionary D
%data X
%sparse coding A
%p: 1 or 2. Use 1,1 norm or 2,1 norm for the matrix

%number of dictionary elements
nD = size(D,2);

%number of data in this group
nX = size(X,2);

%initialization
dr = zeros(size(D,1),1);

% repetion of d_r
DR = zeros(size(D));

%initialization of MU
MU = zeros(1,nX);

% batch update!
%At = A;

nBlocks = length(unique(blocks));
%for r =nD:-1:1
bLabels = unique(blocks);
epsilon = 0.00000001;

DrX = D'*X;

xzc = 0: 300;
xzc = [0 10.^xzc];

normThreshold = 10000;

SVb={};
DrDi = D'*D;

for b=1:nBlocks
    bidx = (blocks==bLabels(b));
    
    dr = D(:,bidx);
    %DrDi = dr'*D;
    DrDi(bidx,bidx) =0;

    [U S V] = svd(dr,0);
    SVb{b}.S = S;
    SVb{b}.V = V;    
end

clear U S V;

for iii = 1:nIter
    
    % for each block
    for b = 1:nBlocks
        % prevA = A;
        
        bidx = (blocks==bLabels(b));
        
        %totalLambdas = lambda*length(bidx);
        
        %dr = D(:,bidx);
        %DrDi = dr'*D;
        %DrDi(:,bidx) =0;
        
        MU = DrX(bidx,:) - DrDi(bidx,:)*A;

        MU = SVb{b}.V'*MU;
        
        % newton's method to solve kappa
        kappa = 0;
        [kappa fval exitflag] = findzero(kappa, MU, lambda, SVb{b}.S, 100, epsilon);
        
        %does not satisfy  the criterion
        if exitflag==0 && fval >= epsilon
            %rough location of zero-crossing
            yzc=[];
            for i=1:length(xzc)
                yzc = [yzc f(xzc(i),MU,lambda,SVb{b}.S)];
            end
            yzc = sign(yzc);
            dy = diff(yzc);
            [a ydx] = find(dy~=0);
            if ~isempty(ydx)
                kappa = xzc(ydx);
                [kappa fval exitflag] = findzero(kappa, MU, lambda, SVb{b}.S, 200, epsilon);
            else
                kappa = 0;
            end
        end
        
        %
        if isnan(kappa)
            kappa = 0;
        end
        
        if kappa<0
            kappa = 0;
        end
        
        A(bidx, :) = kappa / (kappa + lambda) * SVb{b}.V * (SVb{b}.S) ^-2 * MU;
        
    end
    
end



function [kappa fval exitflag dfv] = findzero(kappa, MU, lambda, S, maxIter, epsilon)
for i=1:maxIter
    fval = f(kappa,MU,lambda,S);
    if fval < epsilon
        %kappa = x0;
        break;
        %else
        %    kappa = 0;
    end
    
    dfv = df(kappa,MU,lambda,S);
    x1= kappa-fval/dfv;
    kappa=x1;
    
    % to prevent divergence
    % when A is large, Mu is large and df ~= 0
    if isnan(kappa)
        %kappa = 0;
        i=maxIter;
        break;
        
    end
end

if i~= maxIter
    exitflag = 1;
else
    exitflag = 0;
end

%function for the norm
function val = f(k,mu,lambda,sig)
val = 0;
for r=1:size(mu,1)
    val = val+norm(mu(r,:),2)^2*((k+lambda)*sig(r,r))^-2;
end
val = (val-1);

function val = df(k,mu,lambda,sig)
val = 0;
for r=1:size(mu,1)
    val = val + -2*norm(mu(r,:),2)^2*(k+lambda)^-3*sig(r,r)^-2;
end
val = val;

