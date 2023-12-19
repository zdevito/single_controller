from supervisor import Context, as_completed, Future
import unittest
from unittest.mock import patch, Mock
from contextlib import contextmanager
import supervisor
import time

@contextmanager
def context():
    try:
        ctx = Context()
        yield ctx
    finally:
        ctx.shutdown()

class TestSupervisor(unittest.TestCase):
    def test_future(self):
        with context() as ctx:
            def future(hostname=None):
                return Future(ctx, "test", hostname)

            f = future()
            self.assertFalse(f.done())
            fstr = str(f)
            self.assertIn("incomplete", fstr)
            ctx._schedule(lambda: f.set_result("finished"))
            self.assertEqual(f.result(timeout=1), "finished")
            fstr = str(f)
            self.assertIn("[complete", fstr)
            f2 = future(hostname=f)
            fstr = str(f2)
            self.assertIn("finished", fstr)
            with self.assertRaises(TimeoutError):
                f2.result(timeout=.001)
            with self.assertRaises(TimeoutError):
                f2.exception(timeout=.001)
            ctx._schedule(lambda: f2.set_exception(ValueError('foo')))
            self.assertIsInstance(f2.exception(timeout=1), ValueError)
            with self.assertRaises(ValueError):
                f2.result()

            m = Mock()
            with patch.object(supervisor.logger, 'exception', m):
                f3 = future()
                l = []
                f3.add_done_callback(lambda f: l.append(True))
                def err(v):
                    raise ValueError(v)
                f3.add_done_callback(lambda f: err("done0"))
                self.assertFalse(l)
                ctx._schedule(lambda: f3.set_result("finished"))
                f3.result(timeout=1)
                self.assertEqual(l[0], True)
                f3.add_done_callback(lambda f: l.append(4))
                f3.add_done_callback(lambda f: err("done1"))
                self.assertEqual(l[1], 4)
            self.assertEqual(len(m.call_args_list), 2)
            f4 = future(hostname=future())
            self.assertIn("unconnected", str(f4))
            ctx._schedule(lambda: f4.set_result("finished"))
            time.sleep(0.1)
            self.assertEqual(f4.result(timeout=0), "finished")
            f5 = future()
            ctx._schedule(lambda: f5.set_result("finished"))
            self.assertEqual(f5.result(), "finished")


    def test_as_completed(self):
        with context() as ctx:
            futures = [Future(ctx, f"future_{i}", None) for i in range(10)]
            ctx._schedule(lambda: futures[1].set_result(1))
            for f in as_completed(futures, timeout=1):
                self.assertTrue(f is futures[1] and f.result() == 1)
                break
            ctx._schedule(lambda: futures[2].set_result(2))
            a = as_completed(futures, timeout=1)
            for f in a:
                if f is futures[1]:
                    pass
                elif f is futures[2]:
                    self.assertTrue(f.result() == 2)
                    nf = Future(ctx, "new_future", None)
                    a.add(nf)
                    ctx._schedule(lambda: nf.set_result(3))
                else:
                    self.assertIs(f, nf)
                    break
            with self.assertRaises(TimeoutError):
                for x in as_completed(futures[3:], timeout=.001):
                    self.fail("should have timed out")
            m = Mock()
            with patch.object(Context._process_futures, '__defaults__', (0,)), patch.object(supervisor.logger, 'info', m):
                with self.assertRaises(TimeoutError):
                    for x in as_completed(futures[3:], timeout=.001):
                        pass
            self.assertIn("Waiting for {Future[", m.call_args[0][0])

            for _ in as_completed([]):
                pass

            seen = False
            for x in as_completed([futures[1]]):
                seen = True
            self.assertTrue(seen)

            x = supervisor.wait(futures, return_when=supervisor.FIRST_COMPLETED)
            self.assertTrue(len(x.done))
            self.assertTrue(len(x.not_done))

            x = supervisor.wait(futures[1:3], supervisor.ALL_COMPLETED)
            self.assertEqual(len(x.done), 2)
            self.assertEqual(len(x.not_done), 0)
            ctx._schedule(lambda: futures[4].set_exception(ValueError()))
            x = supervisor.wait(futures[4:6], return_when=supervisor.FIRST_EXCEPTION)
            self.assertTrue(futures[4] in x.done)





if __name__ == '__main__':
    unittest.main()
