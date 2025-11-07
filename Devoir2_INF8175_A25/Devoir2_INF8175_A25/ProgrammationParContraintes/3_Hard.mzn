% BROCHU (2095018)
% LESOURD (2488094)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de villes à visiter + point de départ de la tournée
int: n;
 % Distance entre chaque villes (symmétrique)
array[1..n,1..n] of int: distance;
 % Distance minimale possible entre deux villes différentes
int: min_val = min([distance[i,j] | i,j in 1..n where distance[i,j] > 0]);
% Distance maximale possible entre deux villes différentes
int: max_val = max([distance[i,j] | i,j in 1..n]);

% Fenêtre de temps (1 = début, 2 = fin) durant laquelle il est possible de jouer pour chaque ville
array[1..n,1..2] of int: fenetre_arrivee;
% Temps maximum pour tout le trajet
int: max_trajet = max([fenetre_arrivee[i,2] | i in 1..n]);
% Fenêtre de temps la plus grande du trajet
int: max_fenetre = max([fenetre_arrivee[i,2] - fenetre_arrivee[i,1]| i in 2..n]);

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Ordre des villes visitées par la tournée
array[1..n] of var 1..n: chemin_tournee;
% Circuit de la tournée, circuit_tournee[i] = j => j est le successeur de i
array[1..n] of var 1..n: circuit_tournee;
% Temps cumulé lors de l'arrivé au point i de la tournée
array[1..n] of var 0..max_trajet: cumul_temps;
% Temps d'attente avant le spectacle en ville i
array[1..n] of var 0..max_fenetre: temps_attente;
% Temps d'attente total 
var 0..max_trajet: total_temps_attentes = sum(temps_attente);

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Temps de trajet à minimiser
var int: temps_trajet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Toutes les villes visitées une seule fois (sauf la dernière = ville 1)
constraint all_different(chemin_tournee);

% La dernière ville visitée est la ville 1
constraint chemin_tournee[n] = 1;

% Construction du circuit : successeur exact pour chaque ville
constraint forall(i in 1..n-1) (
    circuit_tournee[chemin_tournee[i]] = chemin_tournee[i+1]
);
constraint circuit_tournee[chemin_tournee[n]] = 1;

% Calcul du temps cumulé et temps d'attente 
% Pour la première ville visitée (index 1 du chemin)
constraint cumul_temps[1] = distance[1, chemin_tournee[1]];
constraint temps_attente[chemin_tournee[1]] = max(0, fenetre_arrivee[chemin_tournee[1],1] - cumul_temps[1]);
constraint cumul_temps[1] + temps_attente[chemin_tournee[1]] <= fenetre_arrivee[chemin_tournee[1],2];

% Pour les villes suivantes
constraint forall(i in 2..n-1) (
    let {
        var int: ville_prev = chemin_tournee[i-1],
        var int: ville_curr = chemin_tournee[i],
        % CORRECTION : Le temps de départ = arrivée précédente + attente précédente
        var int: depart_prev = cumul_temps[i-1] + temps_attente[ville_prev],
        var int: arrivee_brute = depart_prev + distance[ville_prev, ville_curr]
    } in
    temps_attente[ville_curr] = max(0, fenetre_arrivee[ville_curr,1] - arrivee_brute) /\
    cumul_temps[i] = arrivee_brute + temps_attente[ville_curr] /\
    cumul_temps[i] + temps_attente[ville_curr] <= fenetre_arrivee[ville_curr,2]
);

% Pour le retour à la ville 1 (dernière étape)
constraint let {
        var int: derniere_ville = chemin_tournee[n-1],
        var int: depart_derniere = cumul_temps[n-1] + temps_attente[derniere_ville],
        var int: arrivee_finale = depart_derniere + distance[derniere_ville, 1]
    } in (
        temps_attente[1] = 0 /\  % Pas d'attente pour le retour
        cumul_temps[n] = arrivee_finale /\
        temps_trajet = cumul_temps[n]
    );

% Temps d'attente total
constraint total_temps_attentes = sum(i in 1..n)(temps_attente[i]);

solve minimize temps_trajet;

%-----------------------------------------------------------------------------%

output [
  "Chemin de la tournée                   : \(chemin_tournee)\n",
  "Circuit de la tournée                  : \(circuit_tournee)\n",
  "Temps cumulés avant d'entrer en ville  : \(cumul_temps)\n",
  "Temps d'attente avant l'entrée en ville: \(temps_attente)\n",
  "Temps d'attente total                  : \(total_temps_attentes)\n",
  "Temps de trajet total                  : \(temps_trajet)\n",
];
