# Fonction de coût
def cost(schedule, solution):
 
    #Retourne le nombre de créneaux distincts utilisés dans une solution.
    return schedule.get_n_creneaux(solution)

# Construction de la solution initiale
def initial_solution(schedule):
    """
    Génère une solution initiale en utilisant la stratégie "Largest Degree First" :
    - On priorise les cours avec le plus de conflits (degré élevé)
    - Chaque cours reçoit le plus petit créneau possible qui ne
      crée pas de conflit avec ses voisins déjà assignés.
    """
    # Calculer le degré de chaque cours (nombre de conflits)
    degrees = {c: len(schedule.get_node_conflicts(c)) for c in schedule.course_list}

    # Trier les cours par degré décroissant
    sorted_courses = sorted(schedule.course_list, key=lambda c: -degrees[c])

    # Initialisation de la solution
    solution = {}

    # Assigner un créneau à chaque cours
    for course in sorted_courses:
        # Obtenir les créneaux déjà utilisés par les cours en conflit
        neighbor_slots = {solution[conflict] for conflict in schedule.get_node_conflicts(course) 
                          if conflict in solution}

        # Choisir le plus petit créneau libre
        slot = 0
        while slot in neighbor_slots:
            slot += 1

        solution[course] = slot  # Attribuer le créneau au cours

    return solution

# Recherche locale
def local_search(schedule, solution):
    """
    Améliore une solution existante en déplaçant les cours
    vers d'autres créneaux afin de réduire le nombre total
    de créneaux utilisés.

    Stratégie :
    - Pour chaque cours, on teste tous les créneaux existants
      pour voir s'il est possible de le déplacer sans créer
      de conflit.
    - Si un déplacement réduit le nombre de créneaux, on l'applique.
    - On répète jusqu'à ce qu'aucune amélioration ne soit possible.
    """
    improved = True
    current_solution = solution.copy()

    # Boucle jusqu'à ce qu'aucune amélioration ne soit trouvée
    while improved:
        improved = False

        # Parcourir tous les cours
        for course in schedule.course_list:
            current_slot = current_solution[course]

            # Tester tous les créneaux existants
            for new_slot in range(max(current_solution.values()) + 1):
                if new_slot == current_slot:
                    continue  # Ignorer le créneau actuel

                # Vérifier qu'il n'y a pas de conflit dans ce créneau
                conflict = False
                for conflict_node in schedule.get_node_conflicts(course):
                    if current_solution[conflict_node] == new_slot:
                        conflict = True
                        break  # On peut arrêter dès qu'un conflit est trouvé
                if conflict:
                    continue

                # Créer une solution temporaire avec le nouveau créneau
                temp_solution = current_solution.copy()
                temp_solution[course] = new_slot

                # Si la solution est meilleure, l'appliquer
                if cost(schedule, temp_solution) < cost(schedule, current_solution):
                    current_solution = temp_solution
                    improved = True
                    break  # Sortir de la boucle des créneaux pour recommencer

            if improved:
                break  # Recommencer la boucle sur tous les cours

    return current_solution



# Solveur 
def solve(schedule):
    # Étape 1 : solution initiale
    solution = initial_solution(schedule)

    # Étape 2 : amélioration via recherche locale
    solution = local_search(schedule, solution)

    # Étape 3 : retourner la solution finale
    return solution
