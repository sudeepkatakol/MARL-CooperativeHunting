import numpy as np

def get_max_indices(br):
    if br[0] > br[1]:
        br = set([0])
    elif br[0] < br[1]:
        br = set([1])
    else:
        br = set([0, 1])
    return br


def nash_equilibrium(bimatrix, return_all=False, choice="value"):
    assert(bimatrix.shape == (2, 2, 2))
    br = [[0, 0], [0, 0]]
    br[0][0] = get_max_indices(bimatrix[:, 0, 0])
    br[0][1] = get_max_indices(bimatrix[:, 1, 0])
    br[1][0] = get_max_indices(bimatrix[0, :, 1])
    br[1][1] = get_max_indices(bimatrix[1, :, 1])
    ne = []
    for s in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        if s[0] in br[0][s[1]] and s[1] in br[1][s[0]]:
            ne.append(s)
    
    ## NE exists
    if len(ne) > 0:
        ## Multiple NE
        if len(ne) > 1:
            if not return_all:
                if choice == "value":
                    idx = -1
                    value = -10000000
                    for i in range(len(ne)):
                        if sum(bimatrix[ne[i][0], ne[i][1]]) > value:
                            value = sum(bimatrix[ne[i][0], ne[i][1]])
                            idx = i
                    return [ne[idx][0], ne[idx][1]], bimatrix[ne[idx][0], ne[idx][1]]
                ## Random choice
                else:
                    idx = np.random.randint(low=0, high=len(ne))
                    return [ne[idx][0], ne[idx][1]], bimatrix[ne[idx][0], ne[idx][1]]
            if return_all:
                _all = []
                for i in range(len(ne)):
                     _all.append([[ne[i][0], ne[i][1]], bimatrix[ne[i][0], ne[i][1]]])
                return _all
        else:
            idx = 0
            return [ne[idx][0], ne[idx][1]], bimatrix[ne[idx][0], ne[idx][1]]
    return None
