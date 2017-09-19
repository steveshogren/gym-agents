from gym.spaces import discrete

def triplePerms(t):
    ret = []
    (aes,bes,ces) = t.spaces
    for a in range(aes.n):
        for b in range(bes.n):
            for c in range(ces.n):
                ret.append([a+1,b+1,c+1])
    return ret

def tupleSize(t):
    return len(triplePerms(t))
