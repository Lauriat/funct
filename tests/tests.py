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
        self.assertEqual(a.head_option(), 0)
        self.assertTrue(Array().head_option() is None)
        self.assertEqual(a.last, 9)
        self.assertEqual(a.last_option(), 9)
        self.assertTrue(Array().last_option("foo") == "foo")
        self.assertTrue(a.tail.eq(range(1, 10)).all())
        self.assertTrue(a.nonempty)
        self.assertTrue(Array().isempty)
        self.assertTrue(l.map(testfun).equal((1, 2, 0)))
        self.assertTrue(l.parmap(testfun).equal((1, 2, 0)))
        self.assertTrue(l.asyncmap(testfun).result().equal((1, 2, 0)))
        ee = Array(3, 3, 33)
        ee.clear()
        self.assertTrue(ee.isempty)
        self.assertEqual(ee.size, 0)
        self.assertEqual(a.unique().size, a.size)
        self.assertEqual(len(Array(1, 1, 2, 2).unique().to_set() - {1, 2}), 0)
        g[0] = [1, 2]
        self.assertTrue(isinstance(next(g.to_iter()), Array))
        self.assertTrue(m.mul(1).equal(m))
        m[0][0] = 555
        self.assertEqual(m[0][0], 555)
        f = Array(range(10))
        f[:3] = 999
        self.assertTrue(f[:3].eq(999).all())
        f[[1, 2]] = 123
        self.assertTrue(f[:3].eq((999, 123, 123)).all())
        self.assertTrue(Array(1.1, 2.9).int().eq((1, 2)).all())
        self.assertTrue(Array("ab").int().eq((97, 98)).all())
        self.assertTrue(Array(97, 98).char().eq(["a", "b"]).all())
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
        self.assertTrue(
            Array.linspace(1.2, 2.4, 20).eq(np.linspace(1.2, 2.4, 20)).all()
        )
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
        e.sort(inplace=True)
        self.assertTrue(e.argsort().equal(e.range))
        wq = Array((0, 5), (1, 3), (2, 4))
        self.assertTrue(wq.argsortby(lambda e: e[1]).equal((1, 2, 0)))
        qq = Array(-1, 2, -3)
        qq.abs(inplace=True)
        self.assertTrue(qq.equal((1, 2, 3)))
        self.assertTrue(e.equal((1, 4, 5, 99)))
        self.assertTrue(a.reverse().equal(range(9, -1, -1)))
        self.assertTrue(e.reverse(inplace=True).equal((99, 5, 4, 1)))
        self.assertEqual(d.to_str(), "-1-2-4")
        self.assertTrue(Array(1, "foo").equal((1, "foo")))
        self.assertTrue(Array("foo").equal(("f", "o", "o")))
        self.assertTrue(isinstance(next(Array([[1], [2]]).to_iter()), Array))
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
        d.add(1, inplace=True)
        self.assertTrue(d.eq((0, -1, -3)).all())
        self.assertTrue(d.add([3, 2, 1]).eq((3, 1, -2)).all())
        d.add([3, 2, 1], inplace=True)
        self.assertTrue(d.eq((3, 1, -2)).all())

        self.assertTrue(d.mul(-1).eq((-3, -1, 2)).all())
        d.mul(-1, inplace=True)
        self.assertTrue(d.eq((-3, -1, 2)).all())
        self.assertTrue(d.mul([-1, 2, 3]).eq((3, -2, 6)).all())
        d.mul([-1, 2, 3], inplace=True)
        self.assertTrue(d.eq((3, -2, 6)).all())

        self.assertTrue(d.pow(2).eq((9, 4, 36)).all())
        d.pow(2, inplace=True)
        self.assertTrue(d.eq((9, 4, 36)).all())
        self.assertTrue(d.pow([1, 2, 0.5]).eq((9, 16, 6)).all())
        d.pow([1, 2, 0.5], inplace=True)
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
        d.remove(9, inplace=True)
        self.assertTrue(d.eq((16, 6)).all())
        self.assertTrue(d.remove_by_index(1).eq((16,)).all())
        d.remove_by_index(1, inplace=True)
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
        self.assertEqual(b.index_where(lambda k: k == 19), b.size - 1)
        self.assertEqual(b.indices_where(lambda k: k == 999).size, 0)
        self.assertTrue(Array(1, 0, 2, 1).indices(1).eq((0, 3)).all())
        self.assertEqual(a.indices(-1).size, 0)
        self.assertTrue(
            a.split_at(5)
            .zip([range(5), range(5, 10)])
            .map(lambda k: k[0].equal(k[1]))
            .all()
        )
        self.assertTrue(a.takewhile(lambda k: k < 3).eq([0, 1, 2]).all())
        self.assertTrue(a.dropwhile(lambda k: k < 8).eq([8, 9]).all())
        gr = l.groupby(lambda k: k % 2)
        self.assertTrue(gr[gr.map(lambda e: e[0]).index(0)][1].eq(2).all())
        self.assertTrue(gr[gr.map(lambda e: e[0]).index(1)][1].eq([1, 3]).all())
        self.assertTrue(Array((0, 10), (3, -1)).maxby(lambda k: k[1]).eq((0, 10)).all())
        self.assertTrue(Array((4, 2), (-1, 3)).minby(lambda k: k[1]).eq((4, 2)).all())
        y = Array(range(5))
        y[[0, 1, 2]] = 0
        self.assertTrue(y.eq([0, 0, 0, 3, 4]).all())
        y[[0, 1, 2]] = [-1, -2, -3]
        self.assertTrue(y.eq([-1, -2, -3, 3, 4]).all())
        self.assertEqual(Array(1, 2, 3).minby(lambda k: k), 1)
        self.assertEqual(Array(1, 2, 3).maxby(lambda k: k), 3)
        self.assertTrue(
            Array((4, 2), (-1, 3), (5, 1))
            .sortby(lambda k: k[1])
            .equal([(5, 1), (4, 2), (-1, 3)])
        )
        self.assertTrue(
            Array(6, 5, 2, 1).sortby(lambda k: k, inplace=True).eq((1, 2, 5, 6)).all()
        )
        self.assertTrue(l.astype(str).eq(["1", "2", "3"]).all())
        self.assertTrue(l.join("-") == "1-2-3")
        self.assertTrue(l.prepend(0).eq([0, 1, 2, 3]).all())
        l.prepend(0)
        self.assertTrue(l.eq([0, 0, 1, 2, 3]).all())
        l = l.dropwhile(lambda k: k == 0)
        self.assertTrue(l.insert(1, 55).eq((1, 55, 2, 3)).all())
        self.assertTrue(l.eq((1, 55, 2, 3)).all())
        l.remove(55, inplace=True)
        l.insert((0, 0), [99, 98])
        self.assertTrue(l.eq((98, 99, 1, 2, 3)).all())
        l.insert(0, (0, 0))
        self.assertTrue(l.eq(((0, 0), 98, 99, 1, 2, 3)).all())
        self.assertTrue(l.fill(123).eq([123] * l.size).all())
        l.fill(123, inplace=True)
        self.assertTrue(l.eq([123] * l.size).all())
        self.assertTrue(Array(1, 2, 3).eq(Array(1, 2, 3)).all())
        self.assertTrue((Array(1, 2, 3) > Array(-1, -2, 0)).all())
        self.assertTrue(Array(1, 2, 3).ge((-1, -2, 0)).all())
        self.assertTrue(
            (Array(1, 2, 3) >= Array(0, 2, 5)).eq([True, True, False]).all()
        )
        self.assertTrue(Array(1, 2, 3).ge((0, 2, 5)).eq([True, True, False]).all())
        self.assertTrue((Array(-1, -2, 0) < Array(1, 2, 3)).all())
        self.assertTrue(
            (Array(0, 2, 5) <= Array(1, 2, 3)).eq([True, True, False]).all()
        )
        self.assertTrue(Array(5, 6, 7, 8, 9)[[0, 2, 4]].eq([5, 7, 9]).all())
        self.assertTrue(Array(5, 6, 7, 8, 9)[:3].eq([5, 6, 7]).all())
        self.assertTrue(Array(1.1, 1.2, 1.3).round().equal((1, 1, 1)))
        self.assertTrue(Array(1.112, 1.2312, 1.2644).round(1).equal((1.1, 1.2, 1.3)))
        self.assertTrue(Array(0.1, 0.9, 2.0).round().equal(range(3)))
        asd = Array(1.1, 2, 2.9)
        asd.round(inplace=True)
        self.assertTrue(asd.equal((1, 2, 3)))
        h = Array(range(5))
        self.assertTrue(h.clip(1, 3).equal((1, 1, 2, 3, 3)))
        h.clip(0, 2, inplace=True)
        _h = (0, 1, 2, 2, 2)
        self.assertTrue(h.equal(_h))
        self.assertTrue(h.roll(2).equal((2, 2, 0, 1, 2)))
        h.roll(1, inplace=True)
        self.assertTrue(h.equal((2, 0, 1, 2, 2)))
        self.assertTrue(Array(1, 2, 3).roll(100).equal(np.roll((1, 2, 3), 100)))
        h = Array(0)
        h.extend([5, 5])
        self.assertTrue(h.equal((0, 5, 5)))
        h.extendleft([5])
        self.assertTrue(h.equal((5, 0, 5, 5)))
        self.assertTrue(h.pad(1, -1).equal((5, 0, 5, 5, -1)))
        self.assertTrue(h.padleft(1, -1).equal((-1, 5, 0, 5, 5)))
        self.assertTrue(h.pad_to(6).eq((5, 0, 5, 5, 0, 0)))
        self.assertTrue(h.padleft_to(6).eq((0, 0, 5, 0, 5, 5)))
        q = Array(1, 2, float("nan"))
        w = Array(1, 2, float("inf"))
        self.assertFalse(q.isfinite())
        self.assertFalse(w.isfinite())
        self.assertTrue(h.isfinite())
        self.assertTrue(Array([1, 2, 3])[[True, False, False]].equal([1]))
        self.assertTrue(isinstance(Array([[[1], 0]]).head.head, Array))
        self.assertTrue(isinstance(Array([[np.zeros(1), 0]]).head.head, np.ndarray))
        self.assertTrue(
            isinstance(Array([[np.zeros(1), 0]]).to_Array().head.head, Array)
        )
        self.assertTrue(isinstance(Array([gg()]).head, type(gg())))
        self.assertTrue(isinstance(Array([gg()]).to_Array().head, Array))
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

    def testLazy(self):
        l = Array(1, 2, 3)
        a = Array.arange(10)
        d = Array(-1, -2, -4)
        m = Array((1, 2), (3, 4))

        self.assertEqual(a.max_(), 9)
        self.assertEqual(a.min_(), 0)
        self.assertEqual(a.sum_(), 45)
        self.assertEqual(l.argmax_(), 2)
        self.assertEqual(l.argmin_(), 0)
        self.assertTrue(l.map_(testfun).result().equal((1, 2, 0)))
        self.assertTrue(Array(1.1, 2.9).int_().eq_((1, 2)).all_())
        self.assertTrue(Array("ab").int_().eq_((97, 98)).all_())
        self.assertTrue(Array(97, 98).char_().eq_(["a", "b"]).all_())
        self.assertTrue(m.add_(0).result().equal(m))
        self.assertFalse(Array([1], [2]).equal((1, 2)))
        self.assertTrue(Array([1], [2], 3).equal([(1,), (2,), 3]))
        qq = Array(-1, 2, -3)
        self.assertTrue(qq.abs_().result().equal((1, 2, 3)))
        a = Array.arange(10, 20)
        self.assertTrue(d.enumerate_.map_(lambda k: k[0]).result().equal(range(d.size)))
        d = Array(-1, -2, -4)
        self.assertTrue(d.enumerate_.map_(lambda k: k[1]).result().equal(d))
        self.assertTrue(d.abs().equal([1, 2, 4]))
        self.assertTrue(d.add(1).equal((0, -1, -3)))
        d.add(1, inplace=True)
        self.assertTrue(d.eq((0, -1, -3)).all)
        self.assertTrue(d.add([3, 2, 1]).eq((3, 1, -2)).all)
        d.add([3, 2, 1], inplace=True)
        self.assertTrue(d.eq((3, 1, -2)).all)

        self.assertTrue(d.mul(-1).eq((-3, -1, 2)).all)
        d.mul(-1, inplace=True)
        self.assertTrue(d.eq((-3, -1, 2)).all)
        self.assertTrue(d.mul([-1, 2, 3]).eq((3, -2, 6)).all)
        d.mul([-1, 2, 3], inplace=True)
        self.assertTrue(d.eq((3, -2, 6)).all)

        self.assertTrue(d.pow(2).eq((9, 4, 36)).all)
        d.pow(2, inplace=True)
        self.assertTrue(d.eq((9, 4, 36)).all)
        self.assertTrue(d.pow([1, 2, 0.5]).eq((9, 16, 6)).all)
        d.pow([1, 2, 0.5], inplace=True)
        self.assertTrue(d.eq((9, 16, 6)).all)
        hh = Array(1, 2, 3, 4, 6, 10)
        self.assertEqual(Array(np.diff(hh)), hh.diff())
        self.assertEqual(Array(np.diff(hh, 2)), hh.diff(2))

        l = Array(1, 2, 3)
        self.assertTrue(l.map_(lambda k: k ** 2).result().eq((1, 4, 9)).all)
        self.assertTrue(l.forall_(lambda k: k < 5))
        self.assertFalse(l.forall_(lambda k: k < 2))
        self.assertTrue(l.forany_(lambda k: k < 2))
        self.assertFalse(l.forany_(lambda k: k < -1))
        self.assertEqual(l.reduce_(lambda e, b: e * b), 6)
        self.assertEqual(l.reduce_(lambda e, b: e - b), -4)
        a = Array.arange(10)
        self.assertTrue(
            a.split_at(5)
            .zip_([range(5), range(5, 10)])
            .map_(lambda k: k[0].equal(k[1]))
            .all_
        )
        self.assertTrue(a.takewhile_(lambda k: k < 3).result().eq([0, 1, 2]).all)
        self.assertTrue(a.dropwhile_(lambda k: k < 8).result().eq([8, 9]).all)
        self.assertTrue(Array((0, 10), (3, -1)).maxby_(lambda k: k[1]).eq((0, 10)).all)
        self.assertTrue(Array((4, 2), (-1, 3)).minby_(lambda k: k[1]).eq((4, 2)).all)

        self.assertTrue(l.astype_(str).result().eq(["1", "2", "3"]).all)
        self.assertTrue(Array(1, 2, 3).gt_((-1, -2, 0)).result().all)
        self.assertTrue(
            Array(1, 2, 3).ge_((0, 2, 5)).result().eq([True, True, False]).all
        )
        self.assertTrue(Array(1.1, 1.2, 1.3).round_().result().equal((1, 1, 1)))
        self.assertTrue(Array(0.1, 0.9, 2.0).round_().result().equal(range(3)))
        h = Array(range(5))
        self.assertTrue(h.clip_(1, 3).result().equal((1, 1, 2, 3, 3)))
        q = Array(1, 2, float("nan"))
        w = Array(1, 2, float("inf"))
        self.assertFalse(q.isfinite_().all_())
        self.assertFalse(w.isfinite_().all_())
        self.assertTrue(h.isfinite_().all_())
        self.assertEqual(Array(1, 2, 3).mul_(1).product_(), 6)
        self.assertEqual(Array(1, 2).add_(10).next_(), 11)

    def test_errs(self):
        with self.assertRaises(ValueError):
            Array(4, 4, 4, 4).remove(5)
        with self.assertRaises(ValueError):
            Array(4, 4).index_where(lambda k: k == 5)
        with self.assertRaises(IndexError):
            Array(4, 4).remove_by_index(2)
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
