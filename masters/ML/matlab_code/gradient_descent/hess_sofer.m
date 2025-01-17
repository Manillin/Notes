function h=hess_sofer(x)

t = [1 2 4 5 8]';
y = [3 4 6 11 20]';

jacob = [ exp(x(2)*t(1))   x(1)*t(1)*exp(x(2)*t(1));
          exp(x(2)*t(2))   x(1)*t(2)*exp(x(2)*t(2));
          exp(x(2)*t(3))   x(1)*t(3)*exp(x(2)*t(3));
          exp(x(2)*t(4))   x(1)*t(4)*exp(x(2)*t(4));
          exp(x(2)*t(5))   x(1)*t(5)*exp(x(2)*t(5));


        ];

h = jacob' * jacob;

%* Calcolo della Jacobiana 
%  La sommatoria è stata implementata da un ciclo for
% *%

for i = 1 : length(t)
    f_i = x(1) * exp( x(2)*t(i) ) - y(i);
    d2f_i = [       0,      t(i)*exp( x(2)*t(i) );
       t(i)*exp( x(2)*t(i) ),  x(1)*t(i)^2*exp( x(2)*t(i) )];
    h = h + f_i * d2f_i;
end