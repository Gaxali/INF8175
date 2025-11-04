% BROCHU (2095018)
% LESOURD (2488094)


include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Chaque acteur doit avoir un rôle différent
constraint all_different(assignation_role);

% Pas de rôles adjacents pour des acteurs voisins
constraint
  forall(i in 1..n-1) (
    abs(assignation_role[i] - assignation_role[i+1]) != 1
   );

% Deux acteurs différents ne doivent pas porter le même costume
constraint
  forall(i in 1..n, j in i+1..n) (
    choix_costume[assignation_role[i], i] != choix_costume[assignation_role[j], j]
  );

% Solution valide
solve satisfy;

%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];
