from gym.spaces import discrete

def triplePerms(t):
    ret = {}
    perms = []
    (aes,bes,ces) = t.spaces
    for a in range(aes.n):
        for b in range(bes.n):
            for c in range(ces.n):
                perms.append([a+1,b+1,c+1])
    for (i, perm) in zip(range(len(perms)),perms):
        ret[i] = perm
    return ret

def tupleSize(t):
    return discrete.Discrete(len(triplePerms(t)))
