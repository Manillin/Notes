clear; close all;

% domain discretization 
x = linspace(-10,11,100); % 100 punti equispaziati tra -10 e 11
y = linspace(-10,10,100);

% evaluate the function on the grid x,y
[X,Y] = meshgrid(x,y);

% valuto con un unica istruzione la funzione quadratica fq(x1,x2)
% su tutti i punti della griglia 
Z = fq(X,Y);
%plot the function con comando surf | mesh 
surf(x,y,Z);

figure(2);
contour(x,y,Z,60); % 60 sono le curve di livello scelte
 
% return -> se commentato attiva la seconda parte dello script

figure(1); hold on;
% inserisco le coordinate in un vettore 
xs = input('punto in cui disegnare il piano tangente: ');
% accedo alle coordinate con xs(1) e xs(2)
Z = piano_tg(X,Y,xs(1),xs(2));
surf(x,y,Z);
hold off;



