function [Y, S, W] = farPCA(A, relerr, b, p)
% [Y, S, W] = farPCA(A, relerr, b, P)
% The fixed-precision randQB_EI algorithm.
% It produces approxiamte PCA of A, whose approximation error fulfills
%     ||A-YSW'||_F <= ||A||_F* relerr.
% b is block size, P is power parameter.
% Output k is the rank.
    [m, n]  = size(A);
    normA = norm(A, 'fro')^2;
    maxiter= ceil(min(m,n)/2/b);
    Z = [];
    Y = zeros(m, 0);
    W = zeros(n, 0);
    WTW = [];
    threshold= relerr^2*normA;
    for i=1:maxiter
        flag = 0;
        w = randn(n, b);
        alpha = 0;
        for j = 1:p
            if i > 1
                w = A'*(A*w) - W*(Z\(W'*w))-alpha*w;
            else
                w = A'*(A*w)-alpha*w;
            end
            [w, ss, ~] =  eigSVD(w);
            if j > 1 && ss(1,1)>alpha
                alpha = (alpha+ss(1, 1))/2;
            end
        end
        y = A*w;
        w = A'*y;
        if i > 1
            ytYtemp = y'*Y;
            Z = [Z, ytYtemp'; ytYtemp, y'*y];
            wtWtemp = w'*W;
            WTW = [WTW, wtWtemp'; wtWtemp, w'*w];
        else
            Z = y'*y;
            WTW = w'*w;
        end
        Y = [Y, y];
        W = [W, w];
        normB = trace(Z\WTW);
        if (normA - normB) < threshold
            flag = 1;
        end
        if i==maxiter
            flag=1;
        end
        if flag == 1
            [V, D] = eig((Z+Z')/2, 'vector');
            d = sqrt(D);
            VS = V./d';
            C = VS'*WTW*VS;
            C = (C+C')/2;
            [V2, D2] = eig(C, 'vector');
            d = sqrt(D2);
            Y = Y*((VS)*V2);
            S = diag(d);
            W = W*(VS*(V2./d'));
            break;
        end
    end
        
end

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