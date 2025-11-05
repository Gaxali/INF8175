% BROCHU (2095018)
% LESOURD (2488094)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

constraint chiffres[4] mod 2 = 0;
constraint forall(i in 1..4) (chiffres[i] != 0);
constraint alldifferent(chiffres);
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint chiffres[2] > chiffres[4];
constraint sum(chiffres) > 15;

constraint p = chiffres[2] * chiffres[3] * chiffres[4];
constraint nombre = chiffres[1]*1000 + chiffres[2]*100 + chiffres[3]*10 + chiffres[4];

solve minimize p;

%-----------------------------------------------------------------------------%

output [
  show(nombre)
];

