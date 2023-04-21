clear;

%% For Testing Image Dataset
readImage;
tol = 0.1; 

normAf = norm(A, 'fro'); 
[m, n] = size(A);
b = floor(min(m,n)/100);
b = max(b, 20);

fprintf("Relative Error Tolerance: %.2f\n", tol);
fprintf("Block: %d\n\n", b); 

%% UBV factorization
tic
[U,B,V] = randUBV(A, tol, b); 
t1 = toc;
tic
[Ub,S,Vb] = eigSVD(B); 

%% Searching the smallest rank
s  = diag(S); 
s  = sort(s,'descend');
r  = length(s); 
err= sqrt(1 - cumsum(s.^2)/normAf^2);
rT = find(err<tol,1,'first');

%% Compute the approximate PCA
U = U*Ub; 
V = V*Vb;
t2 = toc;

fprintf("UBV, tStop = %.4f\n",tol)
fprintf("Total time: %.4f\n", t1+t2);
fprintf("Initial rank k: %d\n", r); 
fprintf("Truncated rank r: %d\n\n", rT);


%% farPCA
x = [1,5];

for P = x
    tic
    [U, S, V] = farPCA(A, tol, b, P);
    t1 = toc; 
    tic

    %% Searching the smallest rank
    s  = diag(S); 
    s  = sort(s,'descend');
    r  = length(s);
    errqb = sqrt(1 - cumsum(s.^2)/normAf^2); 
    rT = find(errqb<tol,1,'first'); 
    t2 = toc;

    fprintf("farPCA, P=%d\n", P);
    fprintf("Total time: %.4f\n", t1+t2);
    fprintf("Initial rank k: %d\n", r); 
    fprintf("Truncated rank r: %d\n\n", rT);

end

%% randQB_EI
for P = x
    tic
    [Q,B] = randQB_EI_auto(A, tol, b, P);
    t1 = toc; 
    tic
    [Ub,S,Vb] = eigSVD(B);

    %% Searching the smallest rank
    s  = diag(S); 
    s  = sort(s,'descend');
    r  = length(s);  
    errqb = sqrt(1 - cumsum(s.^2)/normAf^2); 
    rT = find(errqb<tol,1,'first'); 
    
    %% Compute the approximate PCA
    U = Q*Ub;
    t2 = toc;

    fprintf("randQB_EI, P=%d\n", P);
    fprintf("Total time: %.4f\n", t1+t2);
    fprintf("Initial rank k: %d\n", r); 
    fprintf("Truncated rank r: %d\n\n", rT);

end


%% svds
tic;
[U, S, V] = svds(A, 427);
t1 = toc;
tic;
%% Searching the smallest rank
s  = diag(S); 
s  = sort(s,'descend');
r  = length(s);  
errqb = sqrt(1 - cumsum(s.^2)/normAf^2); 
rT = find(errqb<tol,1,'first'); 
t2 = toc;
fprintf("svds\n");
fprintf("Total time: %.4f\n", t1+t2);
fprintf("Initial rank k: %d\n", 427); 
fprintf("Truncated rank r: %d\n\n", rT);

function [U,S,V] = eigSVD(A)
    tflag = false;
    if size(A,1)<size(A,2)
        A = A'; 
        tflag = true; 
    end
    B = A'*A; 
    [V,D] = eig(B,'vector'); 
    S = sqrt(D); 
    U = A*(V./S'); 
    if tflag
        tmp = U; 
        U = V; 
        V = tmp; 
    end
    S = diag(S);
end