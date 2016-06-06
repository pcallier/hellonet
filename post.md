# Hello, Net: Recurrent Nets in 3 Popular Frameworks

In the spirit of Andrej Karpathy's great-and-great-for-you post
on recurrent neural networks, I thought I'd lay out how to use 
builtins from three different machine learning frameworks
to do a simple recurrent net.

The problem: character-level language modeling. Can a recurrent net
reproduce the works of Shakespeare? Or maybe just three sentences, 
which I expose it to (with noise) over and over again?

In proper "Hello, World" fashion, the prompts are:

```
hello
goodbye
talk to me
```

The completions are, respectively:

```
patrick
little kitty
(nothing)
```

The training data the net sees are sentences like 
```
hello patrick
goodbye little kitty
talk to me
```

although it will sometimes see examples like `hello little kitty` and 
`talk to me patrick`.

The data are converted to one-hot vectors, with 1 through 26 representing the
letters of the English alphabet and 0 telling the net to stop the sentence.

## Meet the frameworks

*TensorFlow* is the newest arrival on the machine learning library
scene and it has been attracting a lot of attention. It is forged 
in the fires of Google and while its initial release was notably
rough around the edges, the current 0.8.0 release sits on top of
the fairly up-to-date cuDNN v4 and has support for distributed
compute across GPUs.

TensorFlow (TF) works at a fairly low level of abstraction compared
to some popular alternatives, though it is comparable to 
Theano---it exposes a good deal of basic linear algebra functionality and
has a bunch of high-level building block-type functions as well.

Like most machine learning frameworks, TF is _declarative_; the programmer 
specifies a chain of computations in the creation of the model but
the computations themselves are deferred until a later stage. 

*Chainer*, by contrast, is only halfway-declarative---the maintainers call 
their execution model "define-by-run." Unlike TF, where tensor 
operations are specified and linked together before any actual data
is fed to them, in Chainer, the graphs produced by Chainer
function calls contain the results of actual computation
on data. 

This initially helped Chainer differentiate itself
from contenders such as Theano, which have clunky and 
hard-to-understand stand-ins for looping and conditionals
such as `scan`.

But TensorFlow was able to avoid the perils of the dreaded
`scan` and remain fully declarative, however,
so another area where Chainer has tried to distinguish itself
is in its generally Pythonic look-and-feel and high degree
of compatibility with the well-known NumPy array API.

*Neon* has made speed its claim to fame; while in terms of 
interface it has made the choice, similar to its cousin Keras,
to follow in the footsteps of Facebook's Lua-based ML library
Torch. Torch (and by extension neon and Keras) uses container 
constructs to keep track of the computations specified in a model.
Combining the container of computations with some mechanism for
feeding the computations with data allows the library to 
construct the forward and backward computational passes.

In deep learning, the container construct can help you 
think more directly in terms of network layers, rather
than composed functions. For more new or more complex 
architectures, however, it can be an awkward context shift to add
functionality in the backend API. Neon's ML Operational Layer (MOP)
API for specifying tensor computations is, however,
quite analogous to Theano or TensorFlow, so if extensibility
is at issue, there is no need to fret.
