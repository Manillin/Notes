function f=sofer(x)

% il parametro di ingresso x contiene un vettore di due componenti in
% corrispondenza dei quali devo calcolare il valore della funzione
% obiettivo 

% i parametri t e y sono i punti di osservazione, la mia funzione obiettivo
% dipende da questi parametri 
t = [1 2 4 5 8];
y = [3 4 5 11 20];


% sotto i calcoli necessari ad ottenere i valori della funzione obiettivo
f = 0.5 * sum( (x(1) * exp( x(2)*t ) - y).^2);

%nota: con questa espressione ottengo un vettore di 5 componenti
% questo perchè la t e la y sono vettori di len(5) e le espressioni 
% scritte operano in forma vettoriale. notare anche l'elevamento a potenza
% per componenti

