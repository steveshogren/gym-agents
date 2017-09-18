from gym.spaces import discrete,Tuple,Discrete
import helpers as h

import unittest

class TestStringMethods(unittest.TestCase):

    def test_tupleCount(self):
        MOVEMENTS = ['left', 'right']
        BASE = 11
        d = Tuple(
            [Discrete(len(MOVEMENTS)), Discrete(2), Discrete(BASE)]
        )
        self.assertEqual(h.tupleSize(d), (10+1) *2*2)

    def test_tuplePerm(self):
        d = Tuple([Discrete(1), Discrete(2), Discrete(1)])
        self.assertEqual(h.tuplePerms(d), ([1,1,1], [1,2,1]))


if __name__ == '__main__':
    unittest.main()
