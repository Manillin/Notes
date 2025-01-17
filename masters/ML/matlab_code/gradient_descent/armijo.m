function [alpha, info] = armijo(fun,gradx,x,d,s,beta,sigma)

%*
% fun -> funzione 
% gradx -> il gradiente 
% x -> punto dal quale compiere il passo 
% d -> direzione lungo la quale compiere il passo 
% s -> lunghezza iniziale del passo
% beta -> fattore di riduzione
% sigma -> costante di scaling  
% *%

alpha = s;
fx = feval(fun,x);
gf = gradx' *d;

%* Condizione di controllo:
% 
% se lo steplength è stato ridotto cosi tante volte da assumere un valore 
% che è inferiore alla precisione di macchina moltiplicata per la norma
% euclidea della direzione. 
% 
% aka: alpha eccessivamente piccolo 
% *%
while alpha > eps*norm(d)
    if fx - feval(fun,x+alpha*d) < -sigma * alpha * gf
        alpha = alpha * beta;
    else
        info = 0;
        return
    end
end
info = 1; 






