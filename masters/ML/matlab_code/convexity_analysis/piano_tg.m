function z = piano_tg(x1,x2,xs1,xs2)

%x1 e x2 sono i punti in cui valutare il piano tangente - punti di
%discretizzazione del dominio
% xs1, xs2 sono le coordinate del punto in cui il piano deve essere
% tangente alla mia funzione 

vg = grad_fq(xs1,xs2);
z = fq(xs1,xs2) + vg(1)*(x1-xs1) + vg(2)*(x2-xs2);