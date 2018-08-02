function [A objV] = bg_sparse_l12_l1(D,X,A,lambda,beta,blocks,varargin)
% 
% This function solve the following problem
%
%   min_C Q(C; D, X) = 1/2 | X -DC |_F^2 + \lambda sum_i |C[i]|_F 
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
%  ===== Required inputs =============
%
%  D: The dictionary or (basis vectors). 
%     
%  A: The initial sparse coefficients. A can be a vector or a matrix
%
%  X: The data samples. 
%
%  lambda: regularization parameter (scalar)
%
%  beta: regularization parameter (scalar)
%
%  blocks: Block labels indicate the block structure in 'D'. If there are 
%          m blocks in D, numbers in 'blocks' cannot contin anything other
%          than [1, 2, ..., m]
%
%  ===== Optional inputs =============
% 
%   maxIter: Number of iterations. Default = 10.
%   
%   epsilon: The threshold value to terminate the zero-finding function.
%            Default = 1E-7.
%
%   termThre:  Termination threshold value. When the difference between 
%              prev. and curr. values of the bjective function is smaller 
%              than this value, porgram terminate. Default = 1E-5;
%
%   verbose:  Optimizing like a ninja (0) or kids in playground (1). 
%             Default = ninja;


% test for number of required parametres
if (nargin-length(varargin)) ~= 6
     error('Wrong number of required parameters');
end
% default values of variables

% Number of iterations
nIter = 10;

%threshold value for the zero-finding function.
epsilon = 1E-7;

%To calculate the value or no
calobj = 0;

% Termination criterion
termThre = 1E-5;

%verbose mode
verbose = 0;

% Read the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'maxiter'
                nIter = varargin{i+1};
            case 'epsilon'
                epsilon = varargin{i+1};
            case 'termthre'
                termThre = varargin{i+1};
            case 'verbose'
                verbose = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end




bLabels = unique(blocks);
if min(bLabels)~=1
    error('Something wrong the block labels.'); 
end
nBlocks = length(bLabels);

if numel(lambda) ==1
    lambda = ones(1,nBlocks)*lambda;
end
%for r =nD:-1:1

% for the zero-finding function
xzc = 0:300;
xzc = [0 10.^xzc];

%value of objective funcitons at each iteration
objV = [];

% precaluclate values do not depends on A
USb   = {};
DJTDR = D' * D;
DtX   = D' * X;

% precalculating the eigen-decompositions and other dict-related values
for b=1:nBlocks
    
    bidx = (blocks==bLabels(b));

    % calcualte the eigen-decomposition of Dr'*Dr
    [U S] = eig(DJTDR(bidx,bidx));
    % %sort eigenvectors, necessary?
%     [sss idx] = sort(diag(S),'descend');
%     U = U(:,idx);
%     S = diag(sss);
    USb{b}.U = U;
    USb{b}.S = S;
    
    % this is     D[r]'*(sum_{j=\=r} D[j])
    DJTDR(bidx,bidx)=0;
end
clear U S;

%prevObj = -10000;
for iter = 1:nIter
    % for each block
    for b = 1:nBlocks
        
        % index of the blocks
        bidx = (blocks==bLabels(b));
        
        % MU defined in the paper. 
        MU = DtX(bidx,:) - DJTDR(:,bidx)'*A;
                
        % First step shrinkage
        midx = (abs(MU)-beta)<0;
        MU = sign(MU).*(abs(MU)-beta);
        MU(midx) = 0;
        
        % project it
        MU = USb{b}.U'*MU;
        
        % newton's method to solve kappa
        kappa = 0;
        [kappa fval exitflag] = findzero(kappa, MU, lambda(b), USb{b}.S, 100, epsilon);
        
        %does not satisfy  the criterion
        if exitflag==0 && fval >= epsilon
            %rough location of zero-crossing
            yzc=[];
            for i=1:length(xzc)
                yzc = [yzc f(xzc(i),MU,lambda,USb{b}.S)];
            end
            yzc = sign(yzc);
            dy = diff(yzc);
            [a ydx] = find(dy~=0);
            if ~isempty(ydx)
                kappa = xzc(ydx);
                [kappa fval exitflag] = findzero(kappa, MU, lambda(b), USb{b}.S, 200, epsilon);
            else
                kappa = 0;
            end
        end
        
        %
        if isnan(kappa)
            kappa = 0;
        end
        
        % shrinkage
        if kappa<=0
            kappa = 0;
            A(bidx,:) = 0;
        else
            vv = (kappa*USb{b}.S+lambda(b)*eye(size(USb{b}.S,1)))^-1*MU;
            cr = kappa*USb{b}.U*vv;
 
            A(bidx,:) = cr;
        end
    end
    %calculating objective funciton
    ov = 1/2*norm(X-D*A,'fro')^2;  % fidelity term
    for b=1:nBlocks   % regularizer term
        ov = ov + lambda(b)*norm(A(blocks==bLabels(b),:),'fro');
    end
    ov = ov+beta*norm(A(:),1);
    % ------- print?
    if verbose ==1
        disp(sprintf('Iteration(%03d): Q(C; D, X) = %.6f',iter,ov))
    end
    %---- save it 
    objV = [objV ov];
    
    % check if it's time to terminate
    if iter >1
        if objV(end-1) - objV(end) < termThre
            if verbose ==1
                disp(sprintf('Terminated at iteration(%03d)!',iter));
            end
            break;
        end
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

%function for 
function val = f(k,mu,lambda,sig)
val = 0;
for r=1:size(mu,1)
    val = val+norm(mu(r,:),2)^2*(k*sig(r,r)+lambda)^-2;
end
val = (val-1);

function val = df(k,mu,lambda,sig)
val = 0;
for r=1:size(mu,1)
    val = val + -2*sig(r,r)*norm(mu(r,:),2)^2*(k*sig(r,r)+lambda)^-3;
end
val = val;
