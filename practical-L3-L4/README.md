## Practical L3-L4

This practical is going to review the implementation of the minimal automatic differentiation (differentiable programming) system.
It's based on one written by Andrej Karpathy, called [micrograd](
https://github.com/karpathy/micrograd) - but please don't look at it unless you are really stuck!

We've created a version of this library with much of the functionality removed, but with stubs remaining.

The idea is that, using what you learned in the lecture, you can implement this functionality yourselves.

The file you are mainly going to be working on is `engine.py`.
It implements a single 'Value' class, which represents a floating point value that tracks it's gradient.

You can look at the attributes of the 'Value' class in this file - it has a 'data' field that tracks the underlying value, a 'grad' field that is used to store the gradient with respect to a scalar function.

Importantly, it also has a backward hook, which is called when backward is called. This is used to find the gradient of a scalar function $g(f(v))$ with respect to the value $v$, using the chain rule dg/dv = dg/df * df/dv, assuming that dg/df has already been calculated.
We've left an example of how to do this for the add method.

You then need to fill out the backward method, which uses the fact that we can do this for a set of primitive operations to recursively calulate $df/dx$ for any node x in the computation graph.

Then, there are a bunch of simple arithemetic operations (python's built in numerical operators) and a few special functions - relu, sigmoid, cos and sin - which you need to fill in the autograd logic for. This should be a nice refresher of your calculus skills.

In order to help you check your work, we've provided a simple suite of tests, which you can run using pytest.
These tests check that your gradients agree with gradients calulated using finite differences, to avoid giving you the answers.
However, as it evaluates this at random points, occasionally these tests will fail the tolerance check randomly (for example, if the function is near a singular point, like 1/x for small x, then a finite difference approximation is not very accurate).
So if you fail the tests but can't reproduce it, it's probably just numerical error - but if you are failing them more than once in a row you probably got something wrong.

Once you have implemented the value class correctly, you should be able to run the notebook demonstration.ipynb, which imports the value class, and has a couple of small things for you to play with.
Even though the Value class can only differentiate a small number of primitive operations, most interesting numerical programs are just very big compositions of these primitives, so it's enough to, for example, build a very inefficient neural network library.
This should hopefully demonstrate how powerful this simple machinery can be!
