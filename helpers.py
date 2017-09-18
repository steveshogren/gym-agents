from gym.spaces import discrete

def tupleSize(t):
    c = 1
    for s in  t.spaces:
        c = c * s.n
    return c

def tuplePerms(t):
    ret = ()
    #for s in  t.spaces:
    #    c = c * s.n
    return ret
