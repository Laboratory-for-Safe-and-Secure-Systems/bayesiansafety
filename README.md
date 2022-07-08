[![bayesiansafety](https://github.com/othr-las3/bayesiansafety/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/othr-las3/bayesiansafety/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/othr-las3/bayesiansafety/branch/main/graph/badge.svg?token=PQZDQLLBQO)](https://codecov.io/gh/othr-las3/bayesiansafety)

# Repo for our Bayesian Network based Fault Tree and Event Tree Analysis tool.
## Getting started:
The easiest way of learning about bayesiansafey's functionality and starting your own causal analyses is to check out the [examples](https://github.com/othr-las3/bayesiansafety/tree/main/examples).

## Contributing:
As this is an ongoing project, constructive feedback is very welcome.
If you find bugs, feel free to open an issue or fix them right away (pull-request)!

## Installation:
Since there is still a lot going on and the project is rather new, installing bayesiansafety is only supported directly 
via the source code. 

The easiest way for this is simply running setup.py via pip in the main directory

```
pip install .
```

This should install all relevant third-party libraries via the specified requirement files.
A common complication might be encountered while installing [pygraphviz](https://pygraphviz.github.io/documentation/stable/index.html). 
If so - please follow the [official instruction](https://pygraphviz.github.io/documentation/stable/install.html) to fix it.


## Documentation:
Code is the best documentation. 
If you want something more fancy looking simply build the docs!
The current documentation environment is based on [sphinx](https://www.sphinx-doc.org/en/master/).
You first need to setup sphinx including the [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/) extension.

```
pip install -U Sphinx
pip install sphinx-rtd-theme
```

Then go to the "docs" directory and build the documentation via

```
make html
```

Go to ./docs/documentation/html and open index.html in a browser of your choice.

## Citation:
If you are using bayesiansafety in your work, please cite as:
(Details will be available September 2022)

Maier R., Mottok J. (2022) BayesianSafety - an Open-Source package for Causality-Guided, Multi-Model Safety Analysis. In:  xxx (eds) Computer Safety, Reliability, and Security. SAFECOMP 2022. Lecture Notes in Computer Science(), vol xxx. Springer, Cham. https://doi.org/xxxx


## Implemented features:
### core: 
- Class 'ConditionalProbabilityTable' representing CPTs.
- Class 'DiscreteFactor' representing factors.
- Class 'BayesianNetwork' managing a graph and associated probability tables. Support for associational, interventional and counterfactual (via twin-nets) inference.
- Module 'Inference' providing infernce engines and methods to run queries on Bayesian Networks based on different backends (pgmpy, pyagrum).

### faulttree:
- Calculation based on Bayesian Networks.
- Easy definition of a (binary) FT containing AND and OR gates as well as basis events with a fixed or an exponential prob. of failure (given a /lambda)
- Cutset analysis with MOCUS or FATRAM algorithm (at any given time)
- Birnbaum-Importance and Fussell-Vessely-Importance (at any given time)
- Risk Reduction Worth (RRW) and Risk Achievement Worth (RAW) (at any given time, for any given node)
- Plot or save the time evolution of the prob. of failure (for any selected member of the FT)
- Plot FT at any given time showing the current prob. of failure for every member of the FT.
- Modification of time behaviour for any basis event after instantiation. I.e. replace the default behaviour (const. or exp.) for a specific node with a custom function.
- Load FT from OpenPSA file

### synthesis:
- Management of multiple Bayesian Networks and multiple Bayesian Fault Trees
- Management and creation of 'hybrid' networks consisting of a Fault Tree and associated BNs. Shared nodes are specified for BNs as well as mounting points (logic gates) in associated
  Fault Trees where the BN node will be treated as basis event with a const. probability of failure taken from a dedicated state.
- Hybrid networks can consist of multiple BNs - PBFs can be calculated by providing additional associational evidence for individual BNs.
- Full, individually parameterized time evaluation including support of plots for each extended FT.

### eventtree:
- Load Event Trees from OpenPSA file
- Calculation of Consequence (outcome events) likelihoods

### bowtie:
- Load Bow-Tie model (FT + ET)
- Instantiate Bow-Tie model from FT and ET with custom pivot node (node that links both models)
- Calculation of Consequence (outcome events) likelihoods

## Remarks:
- OpenPSA model exchange format files (https://open-psa.github.io/mef/index.html) are only partially supported. 
- FT capabilities include AND/OR gates only.
- Time dependency has to specified via exponential-tags.
- Transfere/housing gates and templates are not supported.
- Custom stochastic functions or conditional parameters are not supported.

- ET capabilities include an abritrary number of branches per functional event. 
- Branching probabilites need to be included as float-tags. 
- Collect-formula-tags, conditionals or parameters are not supported.









