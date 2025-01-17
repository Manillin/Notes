clear; close all;

%punti di osservazione (presi come vettori colonna)
t = [1 2 4 5 8]';
y = [3 4 6 11 20]';

% rappresentiamo i punti di osservazione sul grafico
plot(t,y,'*r');
hold on;


z = linspace(0,12,200);
% x1=2.5 x2=0.21 -> tentativo iniziale
vz = 2.5 * exp(0.21 .* z);
% rappresentiamo la funzione esponenziale con i parametri di prova  
plot(z,vz,'k');

pause 

tic 
%[ris,info] = steepest('sofer','grad_sofer',[2.5,0.21],1e-4,20000);
[ris,info] = newton_sofer('sofer','grad_sofer','hess_sofer',[2.5,0.21],1e-4,20000);
toc 

if info == 0
    sol = ris(1:2,end);
    fprintf('\niter. = %d, soluzione: [%e %e]\n',length(ris)-1,sol);
else
    fprintf('\nsuperato maxit \n');
    return;
end
vz = sol(1) * exp(sol(2)*z);
plot(z,vz,'r');
legend('dati da approssimare', 'app. con pesi [2.5 0.21]', 'app con pesi ottenuti'); 