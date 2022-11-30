import heapq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    distance = 0

    for i in range(1, 8):
        from_i = from_state.index(i)
        to_i = to_state.index(i)
        from_row, from_col = divmod(from_i, 3)
        to_row, to_col = divmod(to_i, 3)

        distance += (abs(from_row - to_row) + abs(from_col - to_col))
    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    succ_states = []
    indices = [i for i, x in enumerate(state) if x == 0]

    for i in indices:
        manh_r, manh_c = divmod(i, 3)
        const = [-1, -1, 1, 1]
        row_col = [manh_r, manh_c]
        state_const = [-3, -1, 3, 1]

        for j in range(4):
            if (row_col[j % 2] + const[j] in range(3) and state[i + state_const[j]]):
                temp_state = state.copy()
                temp_state[i] = state[i + state_const[j]]
                temp_state[i + state_const[j]] = 0
                succ_states.append(temp_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    output = []
    deck = []
    path = []
    move = 0
    parent_index = -1
    curr_index = 0

    h = get_manhattan_distance(state)
    heapq.heappush(
        deck, (move + h, state, (move, h, curr_index, parent_index)))
    max_length = 1

    while deck:
        max_length = max(max_length, len(deck))
        n = heapq.heappop(deck)
        output.append(n)

        # Goal
        if (n[2][1] == 0):
            break

        move = n[2][0] + 1
        parent_index = n[2][2]
        succ_states = get_succ(n[1])

        for n_succ in succ_states:
            is_contain = False
            for output_elem in output:
                if (n_succ in output_elem):
                    is_contain = True
                    break
            if (is_contain == False):
                curr_index += 1
                h = get_manhattan_distance(n_succ)
                cost = move + h
                heapq.heappush(
                    deck, (cost, n_succ, (move, h, curr_index, parent_index)))

    last_index = len(output) - 1
    i = output[last_index][2][3]
    path.append(output[last_index])
    while (i > -1):
        for elem in output:
            if (elem[2][2] == i):
                n = elem
                break
        path.insert(0, n)
        i = n[2][3]

    for succ_state_heap in path:
        print(succ_state_heap[1], "h={}".format(
            succ_state_heap[2][1]), "moves: {}".format(succ_state_heap[2][0]))
    print("Max queue length: {}".format(max_length))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()

    print(get_manhattan_distance(
        [2, 5, 1, 4, 0, 6, 7, 0, 3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()
