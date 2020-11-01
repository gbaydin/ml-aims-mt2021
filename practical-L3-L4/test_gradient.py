from engine import Value
import random
import operator as op
import functools
import math

RTOL = 1e-3

def test_add():
    x = Value(4.343)
    y = Value(1.098)

    z = x + y
    z.backward()

    assert x.grad == 1
    assert y.grad == 1


ops = ['__add__',
      '__mul__',
      '__pow__',
      '__radd__',
      '__sub__',
      '__rsub__',
      '__rmul__',
      '__truediv__',
      '__rtruediv__']

def make_test_binop(opname):
    for i in range(100):
        value = random.random() * 10 - 5
        other = random.random() * 10 - 5
        
        dt = 1e-6
        v = Value(value)
        r = getattr(v, opname)(other)
        r.backward()
        g_auto = v.grad
        r = getattr(value, opname)(other)
        r_ = getattr(value + dt, opname)(other)
        g_emp = (r_ - r) / dt

        rtol = abs((g_emp - g_auto) / g_emp)
        assert rtol < RTOL

def test_add():
     return make_test_binop('__add__')

def test_mul():
     return make_test_binop('__mul__')

def test_pow():
     return make_test_binop('__pow__')

def test_radd():
     return make_test_binop('__radd__')

def test_sub():
     return make_test_binop('__sub__')

def test_rsub():
     return make_test_binop('__rsub__')

def test_rmul():
     return make_test_binop('__rmul__')

def test_truediv():
     return make_test_binop('__truediv__')

def test_rtruediv():
     return make_test_binop('__rtruediv__')

def make_test_unary(opname):
    for i in range(100):
        value = random.random() * 2 - 1
        
        dt = 1e-6
        v = Value(value)
        r = getattr(v, opname)()
        r.backward()
        g_auto = v.grad
        r = getattr(value, opname)()
        r_ = getattr(value + dt, opname)()
        g_emp = (r_ - r) / dt

        rtol = abs((g_emp - g_auto) / (1e-8 + g_emp))
        assert rtol < RTOL

def test_cos():

    for i in range(100):
        value = random.random() * 2 - 1
        
        dt = 1e-6
        v = Value(value)
        r = v.cos()
        r.backward()
        g_auto = v.grad
        r = math.cos(value)
        r_ = math.cos(value + dt)
        g_emp = (r_ - r) / dt

        rtol = abs((g_emp - g_auto) / (1e-8 + g_emp))
        assert rtol < RTOL

def test_sin():

    for i in range(100):
        value = random.random() * 2 - 1
        
        dt = 1e-6
        v = Value(value)
        r = v.sin()
        r.backward()
        g_auto = v.grad
        r = math.sin(value)
        r_ = math.sin(value + dt)
        g_emp = (r_ - r) / dt

        rtol = abs((g_emp - g_auto) / (1e-8 + g_emp))
        assert rtol < RTOL


def test_neg():
    return make_test_unary('__neg__')

def test_relu():
    for value in [random.random(), -random.random()]:
        
        dt = 1e-6
        v = Value(value)
        r = v.sigmoid()
        r.backward()
        g_auto = v.grad
        r = 1 / (1 + math.exp(-value))
        r_ = 1 / (1 + math.exp(-(value + dt)))
        g_emp = (r_ - r) / dt
        
        rtol = abs((g_emp - g_auto) / (1e-6 + g_emp))
        assert rtol < RTOL


def test_sigmoid():
    for value in [random.random(), -random.random()]:
        
        dt = 1e-6
        v = Value(value)
        r = v.relu()
        r.backward()
        g_auto = v.grad
        r = max(value, 0)
        r_ = max(value + dt, 0)
        g_emp = (r_ - r) / dt
        
        rtol = abs((g_emp - g_auto) / (1e-6 + g_emp))
        assert rtol < RTOL

def test_random_poly():
    coeffs = [random.random() for _ in range(10)]

    value = Value(2.718)

    poly = sum([c * value ** p for (c, p) in zip(coeffs, range(len(coeffs)))])

    dt = 1e-6

    poly_dt = sum([c * (value + dt) ** p for (c, p) in zip(coeffs, range(len(coeffs)))]) 

    # finite difference gradient
    g = (poly_dt.data - poly.data) / dt

    poly.backward()
    rtol = abs((value.grad - g) / g)
    assert rtol < RTOL

def test_rosenbrock():
    a = 1
    b = 100

    for i in range(-10, 11):
        for j in range(-10, 11):
            vx = i / 10
            vy = i / 10
            x = Value(vx)
            y = Value(vy)

            f = (a - x) ** 2 + b * (y - x ** 2) ** 2
            f.backward()
            g_auto = [x.grad, y.grad]

            # finite differences
            dt = 1e-8
            
            f = (a - vx) ** 2 + b * (vy - vx ** 2) ** 2
            f_dx = (a - (vx + dt)) ** 2 + b * (vy - (vx + dt) ** 2) ** 2
            f_dy = (a - vx) ** 2 + b * ((vy + dt) - vx ** 2) ** 2
            g_emp = [(f_dx - f) / dt, (f_dy - f) / dt]

            for g, j in zip(g_auto, g_emp):
                rtol = abs((g - j) / (j + 1e-6))
                # for very small values of the gradient, this relative tolerance can blow up.
                if abs(j) < 1:
                    assert abs(g - j) < RTOL
                else:
                    assert rtol < RTOL
            
