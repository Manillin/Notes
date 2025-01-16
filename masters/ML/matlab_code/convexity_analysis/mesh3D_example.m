clear all;

x = linspace(-pi,pi);
y = linspace(-pi/2,pi/2);
[X,Y] = meshgrid(x,y);

% con una istruzione esprimo la funzione che deve essere valutata
% su tutti i punti della griglia 

Z = sin(X.*Y);

mesh(X,Y,Z);
title('Grafico delle funzione sin(xy)');