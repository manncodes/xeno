.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg  
   :target: https://GitHub.com/manncodes/xeno/graphs/commit-activity  
.. image:: https://img.shields.io/badge/license-MIT-blue.svg  
    :target: https://github.com/manncodes/xeno/blob/master/LICENSE   
.. image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg    
   :target: mailto:manncodes@gmail.com   
.. image:: https://img.shields.io/github/issues/manncodes/xeno.svg  
   :target: https://GitHub.com/manncodes/xeno/issues/   
.. image:: https://hitsofcode.com/github/manncodes/xeno
   :target: https://hitsofcode.com/view/github/manncodes/xeno  

  
    
Xeno:Deep learning library from Scratch.
============

Scalable deep learning library built from scratch. Purely based on NumPy.

*Currently in build phase*

Descriptions
============

``Xeno`` is:

1. Based on pure Numpy/Python.
2. Just a midnight curiosity forming real shape.

Features
============
``Xeno`` contains following dee learning features, currently:

* Activations
    * Sigmoid
    * Tanh
    * ReLU
    * Linear
    * Softmax
    * Elliot
    * SymmetricElliot
    * SoftPlus
    * SoftSign     
* Initializations
    * Zero
    * One
    * Uniform
    * Normal
    * LecunUniform
    * GlorotUniform
    * GlorotNormal
    * HeNormal
    * HeUniform
    * Orthogonal
* Layers
    * Linear
    * Dense
    * Convolution
    * Softmax
    * Dropout
    * Embedding
    * BatchNormal
    * MeanPooling
    * MaxPooling
    * SimpleRNN
    * GRU
    * LSTM
    * Flatten
    * DimShuffle 
* Objectives
    * Objectives
    * MeanSquaredError
    * HellingerDistance
    * BinaryCrossEntropy
    * SoftmaxCategoricalCrossEntropy 
* Optimizers
    * SGD
    * Momentum
    * NesterovMomentum
    * Adagrad
    * RMSprop
    * Adadelta
    * Adam
    * Adamax 



One simple code example: 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/manncodes/xeno/blob/master/XenoDemoInColab.ipynb

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_digits
    import xeno

    # prepare
    xeno.utils.random.set_seed(1234)

    # data
    digits = load_digits()
    X_train = digits.data
    X_train /= np.max(X_train)
    Y_train = digits.target
    n_classes = np.unique(Y_train).size

    # model
    model = xeno.model.Model()
    model.add(xeno.layers.Dense(n_out=500, n_in=64, activation=xeno.activations.ReLU()))
    model.add(xeno.layers.Dense(n_out=n_classes, activation=xeno.activations.Softmax()))
    model.compile(loss=xeno.objectives.SCCE(), optimizer=xeno.optimizers.SGD(lr=0.005))

    # train
    model.fit(X_train, xeno.utils.data.one_hot(Y_train), max_iter=150, validation_split=0.1)
