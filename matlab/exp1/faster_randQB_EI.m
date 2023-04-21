function [Y, W] = faster_randQB_EI(A, relerr, b, p, Omega)
% [Y, S, W] = faster_randQB_EI(A, relerr, b, P)
% The fixed-precision randQB_EI algorithm.
% It produces approximate PCA of A, whose approximation error fulfills
%     ||A-YSW'||_F <= ||A||_F* relerr.
% b is block size, P is power parameter.
% Output k is the rank.
    [m, n]  = size(A);
    normA = norm(A, 'fro')^2;
    maxiter = 10;
    Z = [];
    Y = zeros(m, 0);
    W = zeros(n, 0);
    WTW = [];
    threshold= relerr^2*normA;
    for i=1:maxiter
        flag = 0;
        w = Omega(:, (i-1)*b+1:i*b);
        alpha = 0;
        for j = 1:p
            if i > 1
                w = A'*(A*w) - W*(Z\(W'*w));
            else
                w = A'*(A*w)-alpha*w;
            end
            [w, ~] =  qr(w, 0);
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
        C = Z\WTW;
        normB = trace(C);
        if (normA - normB) < threshold
            flag = 1;
        end
        if i==maxiter
            flag=1;
        end
        if flag == 1
            C = (Z+Z')/2;
            [V, D] = eig(C, 'vector');
            d = sqrt(D);
            VS = V./d';
            Y = Y*(VS);
            W = (W*VS)';
            break;
        end
    end
        
end