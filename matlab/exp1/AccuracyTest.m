clear;
load('Dense1.mat');
%load('Dense2.mat');
pmax = 10;

k=200;

[m, n] = size(A);
[U, S, V] = svds(A, k+1);
ss = diag(S(1:k, 1:k)).^2;
s101 = S(k+1, k+1).^2;
A_Ak = A - U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';
Ak_f = norm(A_Ak, 'fro');
Ak_2 = norm(A_Ak, 2);

err1_pve = [];
err1_f = [];
err1_s = [];
err2_pve = [];
err2_f = [];
err2_s = [];
err3_pve = [];
err3_f = [];
err3_s = [];


for p = 0:1:pmax
    [Q, B] = randQB_EI_auto(A, 0, 20, p, Omega);
    [u1, s1, v1] = svd(B, 'econ');
    u1 = Q*u1;
    sst = diag((u1'*A)*A'*u1);
    pvet = max(abs(sst-ss)./s101);
    err1_pve = [err1_pve, pvet];
    err1_f = [err1_f, (norm(A-Q*B, 'fro')-Ak_f)/Ak_f];
    err1_s = [err1_s, (norm(A-Q*B, 2)-Ak_2)/Ak_2];
    
    [Q2, B2] = faster_randQB_EI(A, 0, 20, p, Omega);
    [u2, s2, v2] = svd(B2, 'econ');
    u2 = Q2*u2;
    sst = diag((u2'*A)*A'*u2);
    pvet = max(abs(sst-ss)./s101);
    err2_pve = [err2_pve, pvet];
    err2_f = [err2_f, (norm(A-u2*s2*v2', 'fro')-Ak_f)/Ak_f];
    err2_s = [err2_s, (norm(A-u2*s2*v2', 2)-Ak_2)/Ak_2];
    
    [u3, s3, v3] = farPCA(A, 0, 20, p, Omega);
    sst = diag((u3'*A)*A'*u3);
    sst = flipud(sst);
    pvet = max(abs(sst-ss)./s101);
    err3_pve = [err3_pve, pvet];
    err3_f = [err3_f, (norm(A-u3*s3*v3', 'fro')-Ak_f)/Ak_f];
    err3_s = [err3_s, (norm(A-u3*s3*v3', 2)-Ak_2)/Ak_2];
end

x = 0:pmax;

figure(1);
semilogy(x, err1_pve, 's-', x, err2_pve, 'x-', x, err3_pve, '^-');
xlabel('#power iteration')
ylabel('\epsilon_{PVE}');
legend('Alg. 2', 'Alg. 4', 'Alg. 5');
xmin = 0;
xmax = pmax+1;
ymin = 0.5*min(err3_pve);
ymax = 2*err1_pve(1);
axis([xmin, xmax, ymin, ymax]);
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',25); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'LineWidth',2); 
set( get(gca,'YAxis'),'LineWidth',2); 
set( get(gca,'Legend'),'FontSize',20); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca, 'YTick', [1e-8,1e-6, 1e-4, 1e-2, 1, 100]);

figure(2);
semilogy(x, err1_f, 's-', x, err2_f, 'x-', x, err3_f, '^-');
xlabel('#power iteration')
ylabel('\epsilon_{F}');
legend('Alg. 2', 'Alg. 4', 'Alg. 5');
xmin = 0;
xmax = pmax+1;
ymin = 0.5*min(err3_f);
ymax = 2*err1_f(1);
axis([xmin, xmax, ymin, ymax]);
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',25); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'LineWidth',2); 
set( get(gca,'YAxis'),'LineWidth',2); 
set( get(gca,'Legend'),'FontSize',20); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca, 'YTick', [1e-8,1e-6, 1e-4, 1e-2, 1, 100]);

figure(3);
semilogy(x, err1_s, 's-', x, err2_s, 'x-', x, err3_s, '^-');
xlabel('#power iteration')
ylabel('\epsilon_{s}');
legend('Alg. 2', 'Alg. 4', 'Alg. 5');
xmin = 0;
xmax = pmax+1;
ymin = 0.5*min(err3_s);
ymax = 2*err1_s(1);
axis([xmin, xmax, ymin, ymax]);
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',25); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'LineWidth',2); 
set( get(gca,'YAxis'),'LineWidth',2); 
set( get(gca,'Legend'),'FontSize',20); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca, 'YTick', [1e-8,1e-6, 1e-4, 1e-2, 1, 100]);