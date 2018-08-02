clear; close all; clc;

% files and folder name
dataFolder = './data/';
dictName   = 'D_saved_lam0.4_beta400_25b_20a.mat';
svmName    = 'trainedSVMs.mat';
expName    = 'CVPR_DEMO_';  % for saving the trainined dictionary

% set path
path(pathdef);
addpath('./core');
addpath('./core/large_scale_svm');

% Load USPS dataset.
load([dataFolder 'usps.mat']);

% normalized data l2 norm =1
Xt = Xt./repmat(sqrt(sum(Xt.^2)), size(Xt, 1), 1);
% testing data
Xs = Xs./repmat(sqrt(sum(Xs.^2)), size(Xs, 1), 1);


% Following the suggestion in the book Gaussian Processing
% randomly split the data into half and half??
% http://www.gaussianprocess.org/gpml/data/
half_split_data = true;

if half_split_data == true
    X=[Xt Xs];
    Label = [Lt, Ls];
    train_idx = randperm(size(X, 2));
    Xt = X(:, train_idx(1:ceil(size(X, 2)/2)));
    Lt = Label(train_idx(1:ceil(size(X, 2)/2)));
    Xs = X(:, train_idx(ceil(size(X, 2)/2)+1:end));
    Ls = Label(train_idx(ceil(size(X, 2)/2)+1:end));
    clear X Label;
else
    train_idx = [1:(length(Xt)+length(Xs))];
end

% Data group variables
nG = 150;       %Total number of groups. Must be the multiple of 10
nDperG = 50;    %Number of data per group


% Dictionary structure variables
nblocks=20;     % Number of blocks (n_b in the paper)
nBS=25;         % Number of atoms in each block (n_a in the paper)


%% ======================================================
%  Put samples from the same class in the same group
gIDX = {};   % ground truth indices.
C={};
X1={};
tr_label = [];
for b=0:9
    idx = find(b == Lt);
    for g=1:nG/10
        id2 = randperm(length(idx));
        X1 = [X1 Xt(:, idx(id2(1:nDperG)))];
        tr_label = [tr_label; ones(nDperG, 1)*b];
    end
end

%%  Dictionary Learning
try
    % Load dictionary trained by us.
    load([dataFolder dictName]);
    fprintf('Previously saved dictionary: \n\s\nfound!\nLet''s go to the testin phase first.\n', dictName)
catch
    %
    disp('Previously trained dictionary not found! Let''s train one. \n')
    
    % Block label for the dictionary. 
    blocks = kron([1:nblocks], ones(1, nBS));

    % Randomly initializ the dictionary. 
    iniD = rand(size(X1{g}, 1), nBS*nblocks);
    D = iniD;
    D = D./repmat(sqrt(sum(D.^2)), size(D, 1), 1);

    beta   = 400;
    lambda = 0.4;
    lamOld = lambda;          % used for file saving 
    gamma  = 0.0000000000001;
    eta    = 0.008;           % accelerate converge 
    nIter  = 300;              
    
    % Initialize coefficients
    for g=1:nG
        C{g} = zeros(size(iniD, 2), size(X1{g}, 2));
    end    
    
    totaltime = 0;    % total time so far
    for j=1:nIter
        tic;
        fprintf('%s:Iter(%d) Calculating Sparse Coefficients.....\n', expName, j);
        
        % Compute the sparse codings. 
        nzB = 0;  % non-zero blocks
        for g=1:size(X1, 2)
            C{g} = bg_sparse_l12_l1(D, X1{g}, C{g}, lambda, 0, blocks, 'maxIter', 100, 'verbose', 0);
            
            % Computing the # non-zero blocks
            c = C{g}';
            c = reshape(c, size(C{g}, 2)*nBS, nblocks);
            nzB = nzB + sum(sum(c)~=0);
            
            if mod(g, 10)==0
                fprintf('|');
            else
                fprintf('.');
            end
        end
        fprintf('\n        avg. # of non-zero blocks = %.2f\n', nzB/size(X1, 2));

        % Visualize the coefficients
        figure(1);
        subplot(2, 1, 1);
        imagesc(abs([C{1:5:end}]))
        drawnow;
                
        fprintf('  Updating the dictionary.....\n');
        prevD = D;
        % Training dictionary
        for dd = 1: 15
            D = TrainDictCoh([X1{:}], [C{:}], D, beta, blocks);
            
            idx = find(sum(D.^2)==0);
            if ~isempty(idx)
                D(:, idx) = rand(size(D, 1), length(idx));
                D = D ./ repmat(sqrt(sum(D.^2)), size(D, 1), 1);
                disp('Warning!   0-norm dictionary atoms found! replaced with random vectors')
            end
            fprintf('        deltaD=%.6f\n', norm(D-prevD, 'fro'))
            
            % display dictionary
            im22 = Chapter_12_DispDict2(D, nblocks, nBS, 16, 16, 0);
            figure(1);
            subplot(2, 1, 2);
            %%%imwrite(im22, sprintf('./pic/D_GS%03d.jpg', j));
            imagesc((im22));
            drawnow;
        end

        totaltime = totaltime + toc;
        
        fprintf('------ Iteration %d finished. Time remaining:%.1f(min)\n', ...
                 j, totaltime / j / 60 * (nIter - j));
        
        % save the file every 50 iterations  % these files does not have
        % trained SVM parameters
        if mod(j, 50)==0
            save(sprintf('%s_D_lam%.3f_beta%.2f_nIter%03d_D(%d, %d).mat', ...
                expName, lamOld, beta, j, nBS, nblocks), 'D', ...
                'nblocks', 'nBS', 'blocks', 'lambda', 'beta', 'train_idx', ...
                'eta', 'nZ');
        end
        
        lambda = lambda*(1+eta);
        if lambda >1.5
            lambda = 1.5;
        end
    end
    
    fprintf('Dictionary Learning Completed! \n\n Training 10 1-vs-All linear SVMs using the coefficients of the training samples...\n')
    % SVM training ___
    tr_fea = [C{:}];
    eta1 = 0.1;
    [w, b, class_name] = li2nsvm_multiclass_lbfgs(tr_fea', tr_label, eta1);
    rn = sprintf('%s_D_lam%.3f_beta%.2f_nIter%03d_Dsize(%d, %d).mat', ...
                  expName, lamOld, beta, nIter, nBS, nblocks);
    fprintf('Saving the trained dictionary in %s \n', rn);
    save(rn, 'D', 'nblocks', 'nBS', 'blocks', 'lambda', 'beta', 'train_idx', ...
             'w', 'b', 'class_name', 	'eta');
end

%---------------------------------------------------------------------
%% Testing Phase
try
    load([dataFolder svmName]);
    fprintf('Previously trained SVMs found.\n');
catch
    
    fprintf('Previously trained SVMs NOT found. \n  Calculating sparse coefficients of the training samples.....\n');

    %lambda for calculating cs of the training samples.....
    LAM = 0.2;
    
    % calculate sparse coefficients of training data independently.
    nXt = size(Xt, 2);
    Ct = zeros(size(D, 2), size(Xt, 2));
    Ct2 = Ct;
    
    % non-zero blocks
    nZ = zeros(1, size(Ct, 2));
    nZ2 = nZ;
    eta = 0.1;
    totaLsime = 0;
    
    for i=1:nXt
        tic;
        % BGSC
        Ct(:, i) = bg_sparse_l12_l1(D, Xt(:, i), Ct(:, i), LAM, 0, blocks, 'maxIter', 100);
        % R-BGSC
        %Ct2(:, i) = bg_sparse_DC_faster(D, Xt(:, i), Ct2(:, i), LAM, blocks, 50);
        
        % calculate non-zero blocks of the solution
        c = reshape(Ct(:, i), nBS, nblocks);
        nZ(i) = sum(sum(c)~=0);
        
        c = reshape(Ct2(:, i), nBS, nblocks);
        nZ2(i) = sum(sum(c)~=0);
        
        totaLsime =totaLsime+toc;
        if mod(i, 20) ==1
            %fprintf('Sparse coefficients of training samples... Lam=%.3f, Iter %d/%d: # non-zero blocks: %d|DC-%d, TimeLeft=%.1f(min)\n'...
            %    , LAM, i, nXt, nZ(i), nZ2(i), totaLsime/i*(nXt-i)/60);
            fprintf('Sparse coefficients of training samples... Lam=%.3f, Iter %d/%d, TimeLeft=%.1f(min)\n'...
                , LAM, i, nXt, totaLsime/i*(nXt-i)/60);            
        end
    end
    [Wt1, Bt1, clsN1] = li2nsvm_multiclass_lbfgs(Ct', Lt, eta);
    [Wt2, Bt2, clsN2] = li2nsvm_multiclass_lbfgs(Ct2', Lt, eta);

end % end of try and catch

% Calculate 
%lams = [ 0.05 0.1 0.15 0.2 0.25 0.3 0.5];
lams = 0.25;
lams2 = lams;
cRates = [];
for la = 1:length(lams)
    LAM = lams(la);
    LAM2 = lams2(la);
    
    nX = size(Xs, 2);
    Cs = zeros(size(D, 2), size(Xs, 2));
    Cs2 = Cs;
    
    % non-zero blocks
    nZ = zeros(1, size(Cs, 2));
    nZ2 = nZ;
    totaLsime=0;
    
    %lambda = lambda/50;
    
    
    for i=1:nX
        tic;
        
        % BGSC
        Cs(:, i) = bg_sparse_l12_l1(D, Xs(:, i), Cs(:, i), LAM, 0, blocks, 'maxIter', 100);
        % R-BGSC  
        %Cs2(:, i) = bg_sparse_DC_faster(D, Xs(:, i), Cs2(:, i), LAM2, blocks, 50);

        totaLsime = toc+totaLsime;
        
        % calculate non-zero blocks of the solution
        c = reshape(Cs(:, i), nBS, nblocks);
        nZ(i) = sum(sum(c) ~= 0);
        
        c = reshape(Cs2(:, i), nBS, nblocks);
        nZ2(i) = sum(sum(c) ~= 0);
        
        if mod(i, 20) == 1
            fprintf('Compute sparse coef. of test samples...Lam=%.3f, Iter %d/%d, TimeLeft=%.1f(min)\n'...
                , LAM, i, nX, totaLsime/i*(nX-i)/60);
        end
        
    end
    [C1 dummy Y] = li2nsvm_multiclass_fwd(Cs', w, b, class_name);
    [C2 dummy Y] = li2nsvm_multiclass_fwd(Cs2', w, b, class_name);
    
    cr1 = sum((C1-Ls')==0)/length(C1);
    cr2 = sum((C2-Ls')==0)/length(C2);
    
	[C1 dummy Y] = li2nsvm_multiclass_fwd(Cs', Wt1, Bt1, clsN1);
    [C2 dummy Y] = li2nsvm_multiclass_fwd(Cs2', Wt2, Bt2, clsN2);
    [C3 dummy Y] = li2nsvm_multiclass_fwd(Cs2', Wt1, Bt1, clsN1);

    
    cr3 = sum((C1-Ls')==0)/length(C1);
    cr4 = sum((C2-Ls')==0)/length(C2);    
    cr5 = sum((C3-Ls')==0)/length(C3);
    
    cRates = [cRates [cr1;cr2;cr3;cr4;cr5]];
    fprintf('lambda   G|BG|BG      G|BG|R-BG     I|BG|BG     I|R-BG|R-BG   I|BG|R-BG\n');
    fprintf('__________________________________________________________________________\n')
    fprintf('%.2f  ', lams(la))
    fprintf('   %.2f%%     ', cRates(:, la)*100);
    fprintf('\n')
end
% save the SVM and training result
save(sprintf('%strainedSVMs.mat', expName), 'w', 'b', 'class_name', 'Wt1', 'Bt1', ...
             'clsN1', 'Wt2', 'Bt2', 'clsN2', 'D', 'blocks', 'cRates', 'lams');





