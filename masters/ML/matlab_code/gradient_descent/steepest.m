function[ris,info] = steepest(fun,grad,x,tol,maxit)

% info mi dice se ho soddisfatto il test di arresto 

% x è l'iterata iniziale del processo iterativo 
x= x(:);

% ris è una matrice dove ogni colonna contiene sia l'approssimazione
% corrente che (in col3) il valore della funzion obiettivo su quella approssimazione
% ricordiamo che x è composto da x1,x2 -> per questo ris ha 3 colonne
ris=[x;feval(fun,x)];

% feval mi permette di valutare funzioni prese in ingresso all funzione
% attuale -> valuto gradiente su x (x^(0) iniziale)
g = feval(grad,x);

for i = 1 : maxit
    d = -g;
    %ottengo valore di alpha seguendo la regola di armijo e un parametro
    %inf di salvaguardia in caso ci siano errori 
    [alpha,inf] = armijo(fun,g,x,d,1,0.5,1e-4);
    if inf > 0
        fprintf('\nProblemi in Armijo: alpha <=eps*s \n');
    end
    
    %aggiorno approssimazione corrente con la formula
    x = x + alpha * d;
    
    %a questo punto ho fatto un passo in discesa rispetto alla funzione
    %obiettivo 

    %tengo traccia delle quantità nuove (sia le x che la funzione
    %calcolata nel nuovo punto)
    ris=[ris [x;feval(fun,x)]];

    % calcolo nuovo gradiente
    g = feval(grad,x);

    % valuto condizione di arresto con nuovo gradiente 
    if norm(g) < tol
        info = 0; % 0 -> success 
        return 
    end
end
info = 1;

 
