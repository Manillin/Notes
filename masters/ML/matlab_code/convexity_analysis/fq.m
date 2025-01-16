function z = fq(x1,x2)

% x1 e x2 possono essere sia punti che matrici per come abbiamo 
% strutturat il programma

% x vettore colonna, b vettore riga, A matrice diagonale (hessiana)

%
% f:R^2 --> R
% f(x) = 1/2 * x^T * A * x + b^T * x + c --> f quadratica
% = 0.5 *[x1,x2] * [A11, A12; A21, A22] * [x1;x2] + [b1,b2] * [x1;x2] + c

% assumiamo A come hessiana diagonale e otteniamo
% f(x) = 0.5* (A11*x1^2 + A22*x2^2) + [b1,b2] * [x1;x2] + c 
% = 0.5*(A11x1^2 + A22x2^2) + b1*x1 + b2*x2 + c 


% esempio1: A = [1 0 ; 0 3], b=[-1;0], c=50
% --> funzione strettamente convessa: Hessiana definita positiva
% ha ununico punto di minimo
%z = 0.5 * (x1.^2 + 3 * x2.^2) - x1 + 50;

%notare che l'elevamento a potenza è stato forzato ad operare per
%componenti ('.'), in questo modo funziona nel caso in cui l'input 
%siano vettori o matrici.  

% esempio2: A = [1 0 ; 0 0], b = [-1;0], c = 50
% --> funziona convessa (non strettamente), ha autovalori >= 0
% in questo caso A22 = 0 -> A è semidefinita positiva
%z = 0.5*(x1.^2) - x1 + 50;

% esempio3: [0 0; 0 3], b=[-1;0], c=50
% autovalore A11 uguale a zero
%z = 0.5*( 3 * x2.^2) - x1 + 50;

%esempio4: A = [-1 0 ; 0 3], b=[-1;0], c=50
% la matrice A è indefinita, ha autovalori di segno discorde
% siamo in presenza di un punto di sella!
%z = 0.5*(-x1.^2 + 3*x2.^2) - x1 + 50;


% esempio5: A = [1 0 ; 0 3], b=[-1;0], c=50
% --> funzione strettamente convessa e aggiunta del piano gradiente
% si utilizza il gradiente dal function file grad_fq.m

z = 0.5 * (x1.^2 + 3 * x2.^2) - x1 + 50;


