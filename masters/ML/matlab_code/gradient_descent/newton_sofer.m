function[ris,info] = newton_sofer(fun,grad,hes,x,tol,maxit)

x= x(:);
ris=x;

g = feval(grad,x);

for i = 1 : maxit
    % Cambiamento rispetto a steepest:
    % -> usiamo l'hessiana
    % l'operatore '\' mi risolve il sistema lineare che ha come matrice dei
    % coefficienti feval(hes,x) e come termine noto g.
    % in questo modo troviamo la direzione di discesa usando newton 
    d = - feval(hes,x) \ g;
    [alpha,inf] = armijo(fun,g,x,d,1,0.5,1e-4);
    if inf > 0
        fprintf('\nProblemi in Armijo: alpha <=eps*s \n');
    end
    x = x + alpha * d;
    ris=[ris x];
    g = feval(grad,x);
    if norm(g) < tol
        info = 0; % 0 -> success 
        return 
    end
end
info = 1;