% BROCHU (2095018)
% LESOURD (2488094)


include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de tâches
int: n;
% Durée maximal du projet
int: m;
% Limite de ressources pouvant être partagées entre les tâches en cours
int: limite;

% Durée des tâches
array[1..n] of 1..m: duree_tache;
% Ressources requises par les tâches
array[1..n] of 1..m: ressource_tache;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

array[1..n] of var 1..m: debut_tache; % Début des tâches
array[1..n] of var 1..m: fin_tache; % Fin des tâches

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Fin du projet
var int: fin_projet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Chaque fin de tâche est égale au début de la tache plus sa durée
constraint forall(i in 1..n)(
  fin_tache[i] = debut_tache[i] + duree_tache[i]
);

% Contrainte de borne pour restreindre les domaines des débuts (fin <= m)
constraint forall(i in 1..n)(
  debut_tache[i] <= m - duree_tache[i] + 1
);

% Chaque tâche "consomme" 1 unité de démarrage à son instant de début, limite 2 (manière plus efficace de faire en sorte que plus 2 tâches ne démarrent pas en même temps)
array[1..n] of int: start_duration = [1 | i in 1..n];
array[1..n] of int: start_need = [1 | i in 1..n];

constraint cumulative(debut_tache, start_duration, start_need, 2);

% Utilisation de la contrainte globale cumulative pour gérer la capacité des ressources
constraint cumulative(debut_tache, duree_tache, ressource_tache, limite);

% La fin du projet est le maximum des fins des tâches (bornée)
constraint 
  fin_projet = max(i in 1..n)(fin_tache[i]) /\ fin_projet <= m;

solve minimize fin_projet;
%-----------------------------------------------------------------------------%

output [
  "Envergure du projet  : ", show(fin_projet), "\n",
  "Début des tâches     : ", show(debut_tache), "\n",
  "Fin des tâches       : ", show(fin_tache), "\n",
];