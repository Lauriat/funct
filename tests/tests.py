from funct import Array
import numpy as np
import unittest


# Ugly


class TestArray(unittest.TestCase):
    def test(self):
        l = Array(1, 2, 3)
        a = Array(range(10))
        b = Array(range(10, 20))
        c = Array(range(20, 30, 2))
        d = Array(-1, -2, -4)
        e = Array(5, 4, 99, 1)
        m = Array([[1, 2], [3, 4]])
        g = Array([0])
        gg = lambda: iter(range(2))

        self.assertEqual(a.max(), 9)
        self.assertEqual(a.min(), 0)
        self.assertEqual(a.sum(), 45)
        self.assertEqual(a.head, 0)
        self.assertTrue(Array(True, True).all())
        self.assertFalse(Array(True, True, False).all())
        self.assertTrue(Array(True, False).any())
        self.assertFalse(Array(False, False).any())
        self.assertEqual(a.headOption(), 0)
        self.assertTrue(Array().headOption() is None)
        self.assertEqual(a.last, 9)
        self.assertEqual(a.lastOption(), 9)
        self.assertTrue(Array().lastOption("foo") == "foo")
        self.assertTrue(a.tail.eq(range(1, 10)).all())
        self.assertTrue(a.nonEmpty)
        self.assertTrue(Array().isEmpty)
        self.assertTrue(l.map(testfun).equal((1, 2, 0)))
        self.assertTrue(l.parmap(testfun).equal((1, 2, 0)))
        self.assertTrue(l.asyncmap(testfun).result().equal((1, 2, 0)))
        ee = Array(3, 3, 33)
        ee.clear()
        self.assertTrue(ee.isEmpty)
        self.assertEqual(ee.size, 0)
        self.assertEqual(a.unique().size, a.size)
        self.assertEqual(len(Array(1, 1, 2, 2).unique().toSet() - {1, 2}), 0)
        g[0] = [1, 2]
        self.assertTrue(isinstance(next(g.toIter()), Array))
        m[0][0] = 555
        self.assertEqual(m[0][0], 555)
        f = Array(range(10))
        f[:3] = 999
        self.assertTrue(f[:3].eq(999).all())
        f[[1, 2]] = 123
        self.assertTrue(f[:3].eq((999, 123, 123)).all())
        self.assertTrue(Array(1.1, 2.9).toInt().eq((1, 2)).all())
        self.assertTrue(Array("ab").toInt().eq((97, 98)).all())
        self.assertTrue(Array(97, 98).toChar().eq(["a", "b"]).all())
        self.assertTrue(Array.arange(3).eq([0, 1, 2]).all())
        self.assertTrue(Array.arange(1, 3).eq([1, 2]).all())
        self.assertTrue(Array.arange(1, 3, 0.5).eq([1, 1.5, 2, 2.5]).all())
        self.assertTrue(Array.arange(5.1, 10.1).eq(np.arange(5.1, 10.1)).all())
        self.assertTrue(
            Array.arange(5.1, 0.75, 10.1).eq(np.arange(5.1, 0.75, 10.1)).all()
        )
        self.assertTrue(
            Array.arange(5.1, 0.75, -10.1).eq(np.arange(5.1, 0.75, -10.1)).all()
        )
        self.assertTrue(Array.arange(10.1).eq(np.arange(10.1)).all())
        self.assertTrue(Array.linspace(1, 3, 2).eq(np.linspace(1, 3, 2)).all())
        self.assertTrue(Array.linspace(1.2, 2.4, 20).eq(np.linspace(1.2, 2.4, 20)).all())
        self.assertTrue(
            Array.linspace(1.2, 2.4, 11, endpoint=False).equal(
                np.linspace(1.2, 2.4, 11, endpoint=False)
            )
        )
        self.assertTrue(Array.logspace(1, 3, 2).eq(np.logspace(1, 3, 2)).all())
        self.assertTrue(
            Array.logspace(1.2, 5.4, 11, base=3.14, endpoint=False).equal(
                np.logspace(1.2, 5.4, 11, base=3.14, endpoint=False)
            )
        )
        self.assertEqual(Array((1, 2), (4, 5)).index((4, 5)), 1)
        self.assertTrue(e.sort().equal((1, 4, 5, 99)))
        e.sort_()
        self.assertTrue(e.argsort().equal(e.range))
        wq = Array((0, 5), (1, 3), (2, 4))
        self.assertTrue(wq.argsortBy(lambda e: e[1]).equal((1, 2, 0)))
        qq = Array(-1, 2, -3)
        qq.abs_()
        self.assertTrue(qq.equal((1, 2, 3)))
        self.assertTrue(e.equal((1, 4, 5, 99)))
        self.assertTrue(a.reverse().equal(range(9, -1, -1)))
        self.assertTrue(e.reverse_().equal((99, 5, 4, 1)))
        self.assertEqual(d.toStr(), "-1-2-4")
        self.assertTrue(Array(1, "foo").equal((1, "foo")))
        self.assertTrue(Array("foo").equal(("f", "o", "o")))
        self.assertTrue(isinstance(next(Array([[1], [2]]).toIter()), Array))
        self.assertEqual(Array([1]).unsqueeze, [[1]])
        self.assertEqual(Array([[1]]).squeeze, [1])
        self.assertTrue(Array((1, 2), 3).flatten.equal((1, 2, 3)))
        f = a.copy()
        f[0] += 1
        self.assertNotEqual(f[0], a[0])
        self.assertTrue(a.range.equal(range(a.size)))
        self.assertTrue(d.enumerate.map(lambda k: k[0]).equal(range(d.size)))
        self.assertTrue(d.enumerate.map(lambda k: k[1]).equal(d))
        self.assertTrue(d.abs().equal([1, 2, 4]))
        self.assertTrue(d.add(1).equal((0, -1, -3)))
        d.add_(1)
        self.assertTrue(d.eq((0, -1, -3)).all())
        self.assertTrue(d.add([3, 2, 1]).eq((3, 1, -2)).all())
        d.add_([3, 2, 1])
        self.assertTrue(d.eq((3, 1, -2)).all())

        self.assertTrue(d.mul(-1).eq((-3, -1, 2)).all())
        d.mul_(-1)
        self.assertTrue(d.eq((-3, -1, 2)).all())
        self.assertTrue(d.mul([-1, 2, 3]).eq((3, -2, 6)).all())
        d.mul_([-1, 2, 3])
        self.assertTrue(d.eq((3, -2, 6)).all())

        self.assertTrue(d.pow(2).eq((9, 4, 36)).all())
        d.pow_(2)
        self.assertTrue(d.eq((9, 4, 36)).all())
        self.assertTrue(d.pow([1, 2, 0.5]).eq((9, 16, 6)).all())
        d.pow_([1, 2, 0.5])
        self.assertTrue(d.eq((9, 16, 6)).all())
        hh = Array(1, 2, 3, 4, 6, 10)
        self.assertEqual(Array(np.diff(hh)), hh.diff())
        self.assertEqual(Array(np.diff(hh, 2)), hh.diff(2))

        self.assertEqual(a.difference(b).size, a.size)
        self.assertTrue(l.difference([2, 3, 4]).eq([1]).all())
        self.assertEqual(a.intersect(b).size, 0)
        self.assertTrue(Array(1, 2).append(3).eq([1, 2, 3]).all())
        self.assertTrue(l.intersect([2, 2, 4, 4]).eq([2]).all())
        f = a.copy()
        self.assertTrue(a.remove(0).eq(range(1, 10)).all())
        self.assertTrue(a.eq(f).all())
        d.remove_(9)
        self.assertTrue(d.eq((16, 6)).all())
        self.assertTrue(d.removeByIndex(1).eq((16,)).all())
        d.removeByIndex_(1)
        self.assertTrue(d.eq((16,)).all())
        self.assertTrue(d.get(22) is None)
        self.assertEqual(c.get(0), 20)
        self.assertEqual(c.get(22, "foo"), "foo")
        self.assertTrue(l.map(lambda k: k ** 2).eq((1, 4, 9)).all())
        self.assertTrue(l.forall(lambda k: k < 5))
        self.assertFalse(l.forall(lambda k: k < 2))
        self.assertTrue(l.forany(lambda k: k < 2))
        self.assertFalse(l.forany(lambda k: k < -1))
        self.assertEqual(l.reduce(lambda e, b: e * b), 6)
        self.assertEqual(l.reduce(lambda e, b: e - b), -4)
        self.assertEqual(b.indexWhere(lambda k: k == 19), b.size - 1)
        self.assertEqual(b.indicesWhere(lambda k: k == 999).size, 0)
        self.assertTrue(Array(1, 0, 2, 1).indices(1).eq((0, 3)).all())
        self.assertEqual(a.indices(-1).size, 0)
        self.assertTrue(
            a.splitAt(5)
            .zip([range(5), range(5, 10)])
            .map(lambda k: k[0].equal(k[1]))
            .all()
        )
        self.assertTrue(a.takeWhile(lambda k: k < 3).eq([0, 1, 2]).all())
        self.assertTrue(a.dropWhile(lambda k: k < 8).eq([8, 9]).all())
        gr = l.groupBy(lambda k: k % 2)
        self.assertTrue(gr[gr.map(lambda e: e[0]).index(0)][1].eq(2).all())
        self.assertTrue(gr[gr.map(lambda e: e[0]).index(1)][1].eq([1, 3]).all())
        self.assertTrue(Array((0, 10), (3, -1)).maxBy(lambda k: k[1]).eq((0, 10)).all())
        self.assertTrue(Array((4, 2), (-1, 3)).minBy(lambda k: k[1]).eq((4, 2)).all())
        y = Array(range(5))
        y[[0, 1, 2]] = 0
        self.assertTrue(y.eq([0, 0, 0, 3, 4]).all())
        y[[0, 1, 2]] = [-1, -2, -3]
        self.assertTrue(y.eq([-1, -2, -3, 3, 4]).all())
        self.assertEqual(Array(1, 2, 3).minBy(lambda k: k), 1)
        self.assertEqual(Array(1, 2, 3).maxBy(lambda k: k), 3)
        self.assertTrue(
            Array((4, 2), (-1, 3), (5, 1))
            .sortBy(lambda k: k[1])
            .equal([(5, 1), (4, 2), (-1, 3)])
        )
        self.assertTrue(Array(6, 5, 2, 1).sortBy_(lambda k: k).eq((1, 2, 5, 6)).all())
        self.assertTrue(l.asType(str).eq(["1", "2", "3"]).all())
        self.assertTrue(l.join("-") == "1-2-3")
        self.assertTrue(l.prepend(0).eq([0, 1, 2, 3]).all())
        l.prepend(0)
        self.assertTrue(l.eq([0, 0, 1, 2, 3]).all())
        l = l.dropWhile(lambda k: k == 0)
        self.assertTrue(l.insert(1, 55).eq((1, 55, 2, 3)).all())
        self.assertTrue(l.eq((1, 55, 2, 3)).all())
        l.remove_(55)
        l.insert((0, 0), [99, 98])
        self.assertTrue(l.eq((98, 99, 1, 2, 3)).all())
        l.insert(0, (0, 0))
        self.assertTrue(l.eq(((0, 0), 98, 99, 1, 2, 3)).all())
        self.assertTrue(l.fill(123).eq([123] * l.size).all())
        l.fill_(123)
        self.assertTrue(l.eq([123] * l.size).all())
        self.assertTrue(Array(1, 2, 3).eq(Array(1, 2, 3)).all())
        self.assertTrue((Array(1, 2, 3) > Array(-1, -2, 0)).all())
        self.assertTrue(Array(1, 2, 3).ge((-1, -2, 0)).all())
        self.assertTrue((Array(1, 2, 3) >= Array(0, 2, 5)).eq([True, True, False]).all())
        self.assertTrue(Array(1, 2, 3).ge((0, 2, 5)).eq([True, True, False]).all())
        self.assertTrue((Array(-1, -2, 0) < Array(1, 2, 3)).all())
        self.assertTrue((Array(0, 2, 5) <= Array(1, 2, 3)).eq([True, True, False]).all())
        self.assertTrue(Array(5, 6, 7, 8, 9)[[0, 2, 4]].eq([5, 7, 9]).all())
        self.assertTrue(Array(5, 6, 7, 8, 9)[:3].eq([5, 6, 7]).all())
        self.assertTrue(Array(1.1, 1.2, 1.3).round().equal((1, 1, 1)))
        self.assertTrue(Array(1.112, 1.2312, 1.2644).round(1).equal((1.1, 1.2, 1.3)))
        self.assertTrue(Array(0.1, 0.9, 2.0).round().equal(range(3)))
        asd = Array(1.1, 2, 2.9)
        asd.round_()
        self.assertTrue(asd.equal((1, 2, 3)))
        h = Array(range(5))
        self.assertTrue(h.clip(1, 3).equal((1, 1, 2, 3, 3)))
        h.clip_(0, 2)
        _h = (0, 1, 2, 2, 2)
        self.assertTrue(h.equal(_h))
        self.assertTrue(h.roll(2).equal((2, 2, 0, 1, 2)))
        h.roll_(1)
        self.assertTrue(h.equal((2, 0, 1, 2, 2)))
        self.assertTrue(Array(1, 2, 3).roll(100).equal(np.roll((1, 2, 3), 100)))
        h = Array(0)
        h.extend([5, 5])
        self.assertTrue(h.equal((0, 5, 5)))
        h.extendLeft([5])
        self.assertTrue(h.equal((5, 0, 5, 5)))
        self.assertTrue(h.pad(1, -1).equal((5, 0, 5, 5, -1)))
        self.assertTrue(h.padLeft(1, -1).equal((-1, 5, 0, 5, 5)))
        self.assertTrue(h.padTo(6).eq((5, 0, 5, 5, 0, 0)))
        self.assertTrue(h.padLeftTo(6).eq((0, 0, 5, 0, 5, 5)))
        q = Array(1, 2, float("nan"))
        w = Array(1, 2, float("inf"))
        self.assertFalse(q.isFinite())
        self.assertFalse(w.isFinite())
        self.assertTrue(h.isFinite())
        self.assertTrue(Array([1, 2, 3])[[True, False, False]].equal([1]))
        self.assertTrue(isinstance(Array([[[1], 0]]).head.head, Array))
        self.assertTrue(isinstance(Array([[np.zeros(1), 0]]).head.head, np.ndarray))
        self.assertTrue(isinstance(Array([[np.zeros(1), 0]]).toArray().head.head, Array))
        self.assertTrue(isinstance(Array([gg()]).head, type(gg())))
        self.assertTrue(isinstance(Array([gg()]).toArray().head, Array))
        self.assertEqual(Array(1, 2, 3).mean(), 2)
        self.assertEqual(Array(1, 2, 3).average(), 2)
        self.assertEqual(Array(1, 2, 3).product(), 6)
        self.assertTrue(([1] + Array(2)).equal((1, 2)))
        self.assertTrue(((1, 2) + Array(3)).equal((1, 2, 3)))
        self.assertTrue((Array(1) + [2]).equal((1, 2)))
        self.assertTrue((Array(1) + [2, 3]).equal((1, 2, 3)))
        self.assertTrue((Array(0, 1, 2) + range(3)).equal((0, 1, 2, 0, 1, 2)))
        self.assertTrue((range(3) + Array(0, 1, 2)).equal((0, 1, 2, 0, 1, 2)))
        self.assertEqual(Array(1, 1, 4, 3).count(1), 2)
        md = Array((1, 2), (3, 4), (5, 6))
        qd = Array(((1, 2), (3, 4)), ((5, 6), (7, 8)))
        mdn = np.array(md)
        qdn = np.array(qd)
        self.assertTrue(md[:, 0].equal(mdn[:, 0]))
        self.assertTrue(md[:, 1].equal(mdn[:, 1]))
        self.assertTrue(md[0, :].equal(mdn[0, :]))
        self.assertTrue(md[-1, :].equal(mdn[-1, :]))
        self.assertEqual(md[-1, 1], mdn[-1, 1])
        self.assertTrue(md[:, :].equal(mdn[:, :]))

        self.assertTrue(qd[:, 0].equal(qdn[:, 0]))
        self.assertTrue(qd[:, :, 1].equal(qdn[:, :, 1]))
        self.assertTrue(qd[0, :].equal(qdn[0, :]))
        self.assertTrue(qd[:, 1, :].equal(qdn[:, 1, :]))
        self.assertTrue(qd[1:, 1, :].equal(qdn[1:, 1, :]))
        self.assertEqual(qd[1, 1, 0], qdn[1, 1, 0])
        self.assertTrue(qd[:, :, :].equal(qdn[:, :, :]))

        md[:, 0] = [11, 12, 13]
        mdn[:, 0] = [11, 12, 13]
        self.assertTrue(md.equal(mdn))
        md[1, :] = [0, 1]
        mdn[1, :] = [0, 1]
        self.assertTrue(md.equal(mdn))

        qd[:, 0, :] = 99, 99
        qdn[:, 0, :] = 99, 99
        self.assertTrue(md.equal(mdn))
        qd[1, :, 1] = 11, 12
        qdn[1, :, 1] = 11, 12
        self.assertTrue(md.equal(mdn))
        qd[1, 0, 1] = 0
        qdn[1, 0, 1] = 0
        self.assertTrue(md.equal(mdn))
        qd[:, :, -1] = 33
        qdn[:, :, -1] = 33
        self.assertTrue(md.equal(mdn))
        qd = qd.flatten
        qdn = qdn.ravel()
        self.assertTrue(md[:3].equal(mdn[:3]))
        self.assertTrue(md[3:].equal(mdn[3:]))

    def test_errs(self):
        with self.assertRaises(ValueError):
            Array(4, 4, 4, 4).remove(5)
        with self.assertRaises(ValueError):
            Array(4, 4).indexWhere(lambda k: k == 5)
        with self.assertRaises(ValueError):
            Array(4, 4, 4).add_([1, 5])
        with self.assertRaises(ValueError):
            Array(4, 4).mul_([1, 5, 5])
        with self.assertRaises(ValueError):
            Array(4, 4, 4).pow_([1, 5])
        with self.assertRaises(IndexError):
            Array(4, 4).removeByIndex(2)
        with self.assertRaises(IndexError):
            Array(4, 4, 4)[[0, 1, 5]]
        with self.assertRaises(TypeError):
            Array.zeros(1.1)
        with self.assertRaises(TypeError):
            Array().pad("a")
        with self.assertRaises(TypeError):
            Array().pad(4.2)
        with self.assertRaises(TypeError):
            Array().get("a")
        with self.assertRaises(TypeError):
            1 + Array(1, 2, 3)
        with self.assertRaises(TypeError):
            "a" + Array(1, 2, 3)
        with self.assertRaises(TypeError):
            iter([1]) + Array(1, 2, 3)
        with self.assertRaises(TypeError):
            Array(1, 2, 3) + 1
        with self.assertRaises(TypeError):
            "a" + Array(1, 2, 3) + "a"
        with self.assertRaises(TypeError):
            Array(1, 2, 3) + iter([22])
        with self.assertRaises(IndexError):
            Array(1, 2, 3)[:, 0]
        with self.assertRaises(IndexError):
            Array(1, 2, 3)[:, 0] = 5


def testfun(x):
    return x % 3


if __name__ == "__main__":
    unittest.main()
