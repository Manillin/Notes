function [ris,info] = BB1_m_sofer(fun,grad,x,tol,maxit)

x=x(:);
ris = x;
rho = 1;
g = feval(grad,x);
gold = g;
for i = 1 : maxit
    % in g ho il gradiente, e la mia direzione di discesa sarà: -(rho * g)
    d = -rho*g;
    [alpha,inf] = armijo(fun,g,x,d,1,0.5,1e-4); 
    if inf > 0
        fprint('\nProblemi con Armijo: alpha<=eps*s \n');
    end
    % alpha contiene il valore che garantisce la suff. decrescita della
    % funzione obiettivo secondo Armijo

    % la variabile s mi tiene traccia delle due iterate successive
    s = alpha*d;
    x = x + s;
    g = feval(grad,x);
    % la variabile y tiene traccia della differenza tra il nuovo gradiente
    % e quello vecchio
    y = g-gold; gold = g; 

    %rho = (s'*s)/(s'*y);    % prima regola BB
    rho = (s'*y)/(y'*y);   % seconda regola BB

    ris = [ris x];
    if norm(g) < tol
        info = 0; % success 
        return 
    end 
end 
info = 1;
