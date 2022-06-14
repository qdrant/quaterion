.. Quaterion documentation master file, created by
   sphinx-quickstart on Thu Feb 17 16:24:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Quaterion's documentation!
============================================

Quaterion is a framework for fine-tuning similarity learning models.
The framework closes the "last mile" problem in training models for semantic search, recommendations, anomaly detection, extreme classification, matching engines, e.t.c.

It is designed to combine the performance of pre-trained models with specialization for the custom task while avoiding slow and costly training.

Faatures
---------------

* 🌀 **Warp-speed fast**: With the built-in caching mechanism, Quaterion enables you to train thousands of epochs with huge batch sizes even on *laptop GPU*.
* 🐈‍ **Small data compatible**: Pre-trained models with specially designed head layers allow you to benefit even from a dataset you can label *in one day*.
* 🏗️ **Customizable**: Quaterion allows you to re-define any part of the framework, making it flexible even for large-scale and sophisticated training pipelines.


Install
=======

TL;DR:

For training:

.. code-block:: bash

   pip install quaterion

For inference service:

.. code-block:: bash

   pip install quaterion-models

Quaterion framework consists of two packages - `quaterion` and `quaterion-models <https://github.com/qdrant/quaterion-models>`_.

Since it is not always possible or convenient to represent a model in ONNX format (also, it **is supported**), the Quaterion keeps a very minimal collection of model classes, which might be required for model inference, in a `separate package <https://github.com/qdrant/quaterion-models>`_.

It allows avoiding installing heavy training dependencies into inference infrastructure: `pip install quaterion-models`

At the same time, once you need to have a full arsenal of tools for training and debugging models, it is available in one package: `pip install quaterion`


Next Steps
==========

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">
         <div class="col-md-6">
            <a href="/getting_started/why_quaterion.html">
               <div class="text-container">
                  <h3>Motivation</h3>
                  <p class="body-paragraph">When and Why to use Similarity Learning</p>
               </div>
            </a>
         </div>
         <div class="col-md-6">
            <a href="/getting_started/quick_start.html">
               <div class="text-container">
                  <h3>Quick Start</h3>
                  <p class="body-paragraph">Quaterion overview</p>
               </div>
            </a>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6">
            <a href="/api/index.html">
               <div class="text-container">
                  <h3>API References</h3>
                  <p class="body-paragraph">Detailed list of Quaterion package</p>
               </div>
            </a>
         </div>
         <div class="col-md-6">
            <a href="/tutorials/tutorials.html">
               <div class="text-container">
                  <h3>Tutorials</h3>
                  <p class="body-paragraph">Deep Dive in Similarity Learning</p>
               </div>
            </a>
         </div>
      </div>
   </div>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/why_quaterion
   getting_started/quick_start

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/tutorials
   tutorials/cache_tutorial
   tutorials/nlp_tutorial
   tutorials/cars-tutorial

.. toctree::
   :maxdepth: 1
   :caption: Quaterion API

   api/index
   quaterion

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
