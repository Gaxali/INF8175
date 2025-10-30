def cost(schedule, solution):
    return schedule.get_n_creneaux(solution)

def local_search(schedule, solution):
    best_solution = solution.copy()
    best_cost = cost(schedule, solution)
    nb_timeslots = best_cost

    for node in solution:
        for new_timeslot in solution.values():
            if (solution[node] == new_timeslot):
                continue
            
            conflict_found = False

            for conflict in schedule.get_node_conflicts(node):
                if solution[conflict] == new_timeslot:
                    conflict_found = True

            if (conflict_found):
                continue
            
            new_solution = solution.copy()
            new_solution[node] = new_timeslot
            new_cost = cost(schedule, new_solution)

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

    return best_solution
            


def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    solution = dict()

    timeslot_nb = 0

    for c in schedule.course_list:
        solution[c] = timeslot_nb
        timeslot_nb += 1

    max_iter = 100

    for it in range(max_iter):
        solution = local_search(schedule, solution)
        
    return solution