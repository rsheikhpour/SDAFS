%% SDAFS(l2,p-norm) algorithm 
% This sript is the code for the SDAFS(l2,p-norm)) algorithm for
% semi-supervised feature selection


% Reference:
% A Graph Laplacian-based Sparse Feature Selection Method using Semi-supervised Discriminant Analysis. 
% Razieh Sheikhpour, Kamal Berahmand, Mehrnoush Mohammadi, Hassan Khosravi

% min
% -trace(W'*X*Lb*X'*W)/trace(W'*((X*(Lt+(alpha*L))*X')+(beta*I))*W)+r*||W||2,p
% La should be positive semi-definite
% Lb should be positive semi-definite
%%
function [W, obj] = SDAFS(X,Lt,Lb,L,alpha,beta,r,k,W0)
% X: d*n training data matrix
% Lb, Lt, L: n*n Laplacian matrices defined as in the paper
% r: regularization parameter
% k: reduced dimensionality
% W: d*m projection matrix
% 

%%%%%%%%%%%%%%
[d,n] = size(X);
I=eye(d);
p=input("enter the amount of p:"); %p is the number in l2,p_norm
if nargin<9
    
    R = inv((X*(Lt+(alpha*L))*X')+(beta*I))*((r*2*(1/p)*eye(d))-(X*Lb*X'));
    %R = max(R,R');
    [evec eval] = eig(R);
    eval = diag(eval);
    [temp idx] = sort(eval);
    W = evec(:,idx(1:k));
    
else
    W = W0;
end;

Di = sqrt(sum(W.*W,2)+eps);

Wi = sqrt(sum(W.*W,2)+eps);         %l2,1_norm
%Wi = (sum(W.*W,2)+eps).^(1/4);     %l2,1/2_norm
%Wi = (sum(W.*W,2)+eps).^(1/8);     %L2,1/4_norm
%Wi = (sum(W.*W,2)+eps).^(3/8);     %l2,3/4_norm
W2p = sum(Wi);

obj_ini =(-trace(W'*X*Lb*X'*W)/trace(W'*((X*(Lt+(alpha*L))*X')+(beta*I))*W))+(r*W2p);
obj_former = obj_ini;
iter = 0;

%%%%%%%%%%%%%%%%%%
while 1
    iter = iter + 1;
    d = 0.5./(Wi);              %l2,1_norm
    %d=0.25./((Di).^(3/2));     %l2,1/2_norm 
    %d=0.125./((Di).^(7/4));    %l2,1/4_norm 
    %d=0.375./((Di).^(5/4));    %l2,3/4_norm 
           
    D = diag(d);
    
    %%%%%%%%%
    R = inv((X*(Lt+(alpha*L))*X')+(beta*I))*((r*2*(1/p)*D)-(X*Lb*X'));
    %R = max(R,R');
    [evec eval] = eig(R);
    eval = diag(eval);
    [temp idx] = sort(eval);
    W = evec(:,idx(1:k));
    
    %%%%%%%%%
    Di = sqrt(sum(W.*W,2)+eps);
    
    Wi = sqrt(sum(W.*W,2)+eps);        %L2,1_norm
    %Wi = (sum(W.*W,2)+eps).^(1/4);    %l2,1/2_norm
    %Wi = (sum(W.*W,2)+eps).^(1/8);    %L2,1/4_norm
    %Wi = (sum(W.*W,2)+eps).^(3/8);    %l2,3/4_norm
 
    W2p = sum(Wi);
    
    obj(iter)=(-trace(W'*X*Lb*X'*W)/trace(W'*((X*(Lt+(alpha*L))*X')+(beta*I))*W))+(r*W2p);

    diff = obj_former - obj(iter);
    obj_former = obj(iter);
    if diff < 10.^-3 || iter ==50
        break;
    end
    
end;
plot(obj)