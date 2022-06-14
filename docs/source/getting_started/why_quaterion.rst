Why Similarity Learning?
++++++++++++++++++++++++

There is a number of issues you are likely to face, if you are attempting to solve machine learning problems using traditional approaches like classification:

* **Cold start problem** - you may not have marked-up data before the product is launched. And to mark them up, you have to launch the product.
* **Dataset compatibility** - you have found several datasets suitable for your task, but they have slightly different labeling, which makes it impossible to use them together.
* **Requirement changes** - you have trained the model, but after launching it you found out that there is a new class that is not included in the markup.
* **Impossible tweaking** - once the model is deployed, it is impossible to alter it's behavior in some specific corner cases without manual condition checking.
* **Lack of Explanability** - explaining predictions of the classification model is a form of art. Only the simplest models have some form of explainability available.

Similarity Learning offers an alternative that eliminates these disadvantages.

Unlike traditional classification models it does not rely on a predefined set of labels, but instead learns the similarities between objects.
And turns out that this approach gives a number of benefits on data collection, modelling, and deployment stages.

It is much easier to collect datasets for similarity learning, because any classification dataset is also a similarity dataset. The information about classes can be directly used to determine similarity between objects.

And therefore, multiple classification datasets could be combined into a single similarity dataset even if initial labels were not compatible.
And in some cases you won't even need manual markup at all. If the data consists of several components, such as multimodal pairs, you can use a self-supervised approach.

On the modelling stage similarity learning is also much flexible - it does not depend on a fixed number of classes, so the new categories could be introduced by just a simple extension of the reference set.
Same is also true for the deployment stage, where you can introduce new examples without any downtime and redeployment.

And finally, examining the similarities between objects resulting from the model can provide insights into what the model is guided by when making predictions.

When to apply Similarity Learning?
==================================

Of course, this approach can not solve every problem.
In some cases, binary classification is more than enough, but there is a set of common patterns for tasks compatible with similarity learning.

First, similarity learning can help in tasks where it is difficult to define class boundaries. These are all kinds of recommendation and matching problems. Those also include all tasks in which the number of classes is large or may increase.

In general, if you are going to train a classifier with more than 50 classes or, heaven forbid, thinking about hierarchical classification - we encourage you to consider similarity learning as an alternative.

This also includes binary classification problems, where many structurally different subclasses are combined under one class. An example of such a problem is anomaly detection - a set of anomaly categories united by a single label. They can be processed much better by allowing the model to find anomalies similar to those already seen.

Check out our `collection <https://qdrant.tech/solutions/>`_ of solutions you can build with Similarity Learning.

Why Quaterion?
++++++++++++++

Many general-purpose frameworks allow you to train Computer Vision or NLP tasks quickly.
However, Similarity Learning has peculiarities, which usually require a significant additional layer on top of the usual pipelines.

So, for example, the batch size in the training of similarity models has a much greater role than in other models.
Labels either do not exist or are handled in a completely different way. In many cases, the model is already pre-trained, which also adjusts the process.

By focusing on Similarity Learning, Quaterion can better support such specialized requirements while simplifying development and accelerating the training process.

In addition, Quaterion is made with the intention that its primary mission will be fine-tuning the models.
It makes it compatible even with tasks with only a tiny amount of marked-up data.

Quaterion uses PyTorch Lightning as a backend. It lets you be productive instantly without getting bogged down in the boilerplate. You can sneak a peek at what Quaterion tastes like in the `Quick Start guide <./quick_start.html>`_.
