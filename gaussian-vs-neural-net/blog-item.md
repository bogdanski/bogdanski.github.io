---
title: 'Gaussian vs. Neural Net'
published: true
date: '11-01-2016 14:20'
visible: true
author: 'Felix Bogdanski'
authorimage: felix-new2.jpg
---

<link rel="stylesheet" href="/user/themes/zen/js/highlight-2/styles/monokai.css">
<script src="/user/themes/zen/js/highlight-2/highlight.pack.js"></script>
<script>hljs.initHighlightingOnLoad();</script>

<!-- KaTex -->
<link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0-alpha2/katex.min.css">
<script src="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0-alpha2/katex.min.js"></script>
<script>
  window.onload = function() {
      var tex = document.getElementsByClassName("tex");
      Array.prototype.forEach.call(tex, function(el) {
          var inline = el.getAttribute("inline") == "true";
          katex.render(el.getAttribute("data-expr"), el, { displayMode: !inline });
      });
  };
</script>
<!-- /Katex -->

In the article [Artificial Neural Networks](/blog/artificial-neural-networks) we studied the core concepts of these powerful, universal approximators. In this article, we research how a neural net can be used instead of a gaussian for basic predictive analysis. You can check the code and data on [GitHub.](https://github.com/zenecture/neuroflow)

[This data set](http://archive.ics.uci.edu/ml/datasets/Adult) from the University of California aggregates US census income data from 1994. The data set counts 32461 rows, which is roughly 4MB. Being a multivariate data set, we could pick arbitrary combinations of features for our model. To keep things simple, we work with only age and a boolean, indicating if a person makes more than 50K a year. In the end, we want a probability of earning more than 50K a year with respect to age.

```bash
26, Private, 94936, Assoc-acdm, 12, Never-married, Sales, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
38, Private, 296478, Assoc-voc, 11, Married-civ-spouse, Craft-repair, Husband, White, Male, 7298, 0, 40, United-States, >50K
36, State-gov, 119272, HS-grad, 9, Married-civ-spouse, Protective-serv, Husband, White, Male, 7298, 0, 40, United-States, >50K
33, Private, 85043, HS-grad, 9, Never-married, Farming-fishing, Not-in-family, White, Male, 0, 0, 20, United-States, <=50K
22, State-gov, 293364, Some-college, 10, Never-married, Protective-serv, Own-child, Black, Female, 0, 0, 40, United-States, <=50K
43, Self-emp-not-inc, 241895, Bachelors, 13, Never-married, Sales, Not-in-family, White, Male, 0, 0, 42, United-States, <=50K
...
```

Since census data means people and the law of large numbers, things often turn out to be under a normal distribution. Let's for now forget about the neural net and build a predictive model with the good, old gaussian:

<p class="tex" data-expr="\mathcal{N}(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}"></p>

While being a clever way of drawing a bell curve, the gaussian is inherently unimodal and, because of the square within the functional dependency, symmetric. Indeed, a very perfect function &mdash; but maybe _too perfect for nature?_

<p class="gentle gentle-bottom text-center">
	<img src="/blog/gaussian-vs-neural-net/normage.png" style="width: 50%;" />
</p>

The plot has mean <span class="tex" inline="true" data-expr="\mu = 44.24"></span>, it is the average age of people earning more than 50k. Chances are higher for this person to be 44 than 18 years. Sounds pretty reasonable. But my gut tells me there is something wrong about the symmetry. Is it true, that 18 and 70 agers earn the same? Intuitively, I would say no, if a person is 70 years old, he or she earns more money than the average kid of 18 years. Think of pension income, interest income, or even a regular job. Maybe, the gaussian is just too perfect to explain nature.

Let's build a more natural model using our beloved neural nets. We choose layout <span class="tex" inline="true" data-expr="L = [1,20,1]"></span>, so 1 neuron for the input age, 20 neurons for the dense layer and one neuron for the output boolean encoded as <span class="tex" inline="true" data-expr="true=1, false=0"></span>:

```scala
val src = scala.io.Source.fromFile(getResourceFile("file/income.txt")).getLines.map(_.split(",")).flatMap { k =>
  (if (k.length > 14) Some(k(14)) else None).map { over50k => (k(0).toDouble, if (over50k.equals(" >50K")) 1.0 else 0.0) }
}.toArray

val f = Sigmoid

val network = Network(Vector(1) :: Dense(20, f) :: Dense(1, f) :: SquaredMeanError())

val maxAge = train.map(_._1).sorted.reverse.head

val xs = train.map(a => Seq(a._1 / maxAge))
val ys = train.map(a => Seq(a._2)) // Boolean > 50k

network.train(xs, ys)
```

Furthermore, we need to map the age to domain <span class="tex" inline="true" data-expr="[0, 1]"></span>, because this is where the sigmoid operates on. So, we divide by maximum age and start training. After a couple of seconds, and a cup of sencha, training with 2000 samples succeeds, so we can normalize and plot both models:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/gaussian-vs-neural-net/gaussiannet.png" style="width: 50%;" />
</p>

Comparing the gaussian with the net, we notice a slight difference between them. While the gaussian is not able to capture the asymmetry, our net can capture this shape in a natural way. The gaussian works with mean and variance to fit a training set. However, the gaussian shape will always _dominate,_ because of its functional form. A neural net, on the other hand, _learns_ the underlying data. If, for instance, our data set was multimodal, i. e. with two peaks, the gaussian would give poor results, whereas the neural net would be able to capture it.

Thanks for reading.