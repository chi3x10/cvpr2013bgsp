function D = TrainDictCoh(X,C,D,beta,blocks)
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
% X data in columns 
% C: corresponding sparse representation
% D: dictionary to be updated.
% beta: regularizer constant
% blocks: block structure of D

nBlocks = length(unique(blocks));

XCr = X*C';
CiCr = C*C';
%the diagnal of CiCr is cr*cr'
CrCr = diag(CiCr);

%make the diagnal 0
CiCr = CiCr.*(1-eye(size(CiCr,2)));

diCiCr = D*CiCr;

nDim = size(D,1);

for i=1:nBlocks
    idx = find(blocks==i);

    %for each atom in block i
    for r=1:length(idx)
        if CrCr(idx(r)) >0.00000001
            
            diCiCr = D*CiCr;
            djdj=beta*(D(:,idx)*D(:,idx)'-D(:,idx(r))*D(:,idx(r))');
            %tic;
            mu = XCr(:,idx(r)) - diCiCr(:,idx(r));
            %disp(['mu:' num2str(toc)]);
            
            %tic;
            djdj = djdj+eye(nDim)*CrCr(idx(r));
            %disp(['all:' num2str(toc)]);
            
            %tic;
            D(:,idx(r)) = djdj\mu; %inv(djdj)*mu
            %D(:,idx(r)) = djdj^-1*mu;
            %disp(['ddd:' num2str(toc)]);
            D(:,idx(r)) = D(:,idx(r))./norm(D(:,idx(r)),2);
        else
            D(:,idx(r)) = zeros(size(D,1),1);
        end
    end
end
