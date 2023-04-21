function [U,B,V,E] = randUBV(A, relerr, b)
% [U,B,V,E] = randUBV(A, relerr, b)
% The fixed-precision randUBV algorithm.
% It produces a factorization UBV of A that satisfies
%     ||A-UBV||_F <= ||A||_F* relerr.
% b is the block size. 
% E : approximation error estimate
% ---------------------------------
% Code for Algorithm 4.1 
% First commit: December 2020
% Last update : February 2021
% ---------------------------------


    [m,n] = size(A); 
    maxiter = floor(min(m,n)/(2*b)); 
 
    E = norm(A,'fro')^2;
    threshold = relerr^2*E; 
    deflTol = 1e-12*sqrt(norm(A,1)*norm(A,Inf));

    Omega = randn(n,b); 
    [Vk,~] = qr(Omega,0);
    Uk = zeros(m,0); 
    Lt = zeros(b,0);

    V = Vk;
    U = Uk; 
    
    r = 0; % row indexing for B
    c = 0; % column indexing for B
        
    for k = 1:maxiter
        
        % Compute next U, R
        [Uk,R,p] = qr(A*Vk-Uk*Lt',0); 
        dr = find(abs(diag(R))>=deflTol, 1, 'last'); 
        if isempty(dr)
            dr = 0; 
        end
        pt = zeros(size(p)); pt(p) = 1:length(p); 
        R  = R(1:dr,pt);
        Uk = Uk(:,1:dr); 
        U  = [U,Uk]; %#ok<AGROW>
        
        % Update B and error indicator
        B(r+1:r+dr,c+1:c+b) = R;
        c = c + b; 
        E = E - norm(R,'fro')^2;
        
        % Compute next V, L
        Vk = A'*Uk - Vk*R';
        Vk = Vk - V*(V'*Vk); % full reorthogonalization
        [Vk,Lt,p] = qr(Vk,0); 
        pt = zeros(size(p)); pt(p) = 1:length(p); 
        dc = find(abs(diag(Lt))>=deflTol, 1, 'last'); 
        if isempty(dc)
            dc = 0;
        end
        Lt = Lt(1:dc,pt); 
        Vk = Vk(:,1:dc); 
        V  = [V,Vk]; %#ok<AGROW>
        
        % Augmentation 
        if dc < b
            Vaug = randn(n,b-dc); 
            [Vaug,~] = qr(Vaug - V*(V'*Vaug),0); 
            V = [V,Vaug];   %#ok<AGROW>
            Vk = [Vk,Vaug]; %#ok<AGROW>
            Lt(dc+1:b,:) = 0; 
        end
        
        % Update B and error indicator
        B(r+1:r+dr,c+1:c+b) = Lt'; 
        r = r + dr; 
        E = E - norm(Lt,'fro')^2;  
        
        % Check for termination
        if E < threshold
            break
        end
    end
end 



