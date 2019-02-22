---
title: 'Natural Trend Detection'
published: true
date: '12-01-2016 11:34'
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

In the [last article](/blog/gaussian-vs-neural-net) we compared the gaussian with a neural net. Now we research how a neural net can be applied for the detection of trends in 2d spaces. You can check the code and data on [GitHub.](https://github.com/zenecture/neuroflow)

A trend indicates a lot of people like a certain thing. Possible use cases for detecting trends are stock market, clicks on headlines, social and video platforms, et cetera. When I think of a trend, I see a graph, a steep curve to the top, or bottom if negative. To me, the definition of a trend is a graphical one, thus a good candidate for a neural net, since it learns graphical shapes in a _natural way_, e. g. neural receptors similar to the human retina.

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/trendflat.png" style="width: 50%;" />
</p>

Looking at the candle bars, we can clearly see an uptrend, supported by the linear function drawn in purple. The green line is a constant, flat function and is the opposite of a trend. How can we train a neural net to differentiate between these two? 

## Abstract Time Steps

A trend is time, but time is not always a trend. For instance, in the trading scene, a daytrader or scalper is more interested in riding with short term trends, whereas a long-term investor focuses on a much wider timeframe. A long-term downtrend may accumulate a lot of short-term uptrends. Our net should be able grasp the difference bewteen a trend and the complete opposite. We work on abstract time steps, so we can use the net for any time frames.

We map the training data to domain <span class="tex" inline="true" data-expr="[0,1]"></span>, which is for the sigmoid activator. So, as long as input data will be mapped to domain <span class="tex" inline="true" data-expr="[0,1]"></span>, we can safely use our trained net to detect a trend in time series data of arbitrary length.

Let's generate our training data using Scalas _Range:_

```scala
val trend = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i))
val flat = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.3))
```

Here, the _trend_ data is drawn from the linear function from 0.0 until 1.0 with step size 0.01, whereas _flat_ is a constant discrete line. The constant 0.3 is picked arbitrarily and could be marginalized out through a richer training set. Note that choosing, for instance, 0.5 instead of 0.3 leads to similar results. The important thing is that both graphical shapes are clearly separable.

Next, we need to spawn a neural network:

```scala
val f = Sigmoid
val net = Network(Vector(trend.size) :: Dense(25, f) :: Dense(1, f) :: SquaredMeanError())

net.train(Seq(trend, flat), Seq(->(1.0), ->(0.0)))
```

We choose a <span class="tex" inline="true" data-expr="[N,25,1]"></span> net architecture, with N counting our discrete training values. Every discrete step of our training set will get its dedicated neuron. Plus, a step size of 0.01 means that the range produces <span class="tex" inline="true" data-expr="N = 200"></span> neural receptors, so our model lives in a 5025 dimensional space. The output neuron will answer with a number close to 1 if it's a trend, and 0 if it's not.

The training succeeds after 118 seconds:

```bash
[INFO] [12.01.2016 12:52:33:853] [run-main-0] Took 61 iterations of 10000 with error 9.965761971223065E-5  
Weights: 5025
[success] Total time: 118 s, completed 12.01.2016 12:52:33
```

Let's see what kind of answers we get for various inputs. Feeding it with training input, a linear and a flat function, we can check if they get classified correctly:

```bash
Flat Result: DenseVector(0.010301838081712023)
Linear Trend Result: DenseVector(0.9962517960703637)
```

Yup, good.

### Square trend

A pure, linear trend will be rare, so let's try it with a square trend:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/squaretrend.png" style="width: 50%;" />
</p>

```scala
val squareTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i * i))
```

```bash
Square Trend Result: DenseVector(0.958457769839082)
```

Our net is pretty confident that this is a trend. Check!

### Linear downtrend

Let's feed it with a linear downtrend and see what happens:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/downtrend.png" style="width: 50%;" />
</p>

```scala
val declineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 1.0 - i))
```

```bash
Linear Decline Trend Result: DenseVector(0.0032519862410505525)
```

Our net is very confident that a linear downtrend is _not_ an uptrend. Check!

### Square downtrend

This time, we try a square downtrend:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/squaredowntrend.png" style="width: 50%;" />
</p>

```scala
val squareDeclineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (-1 * i * i) + 1.0))
```

```bash
Square Decline Trend Result: DenseVector(0.011391593430466094)
```

Our net is very confident that a square downtrend is _not_ an uptrend. Check!

### Jamming trend

Let's make it a bit harder and simulate a _jamming_ trend:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/jammingtrend.png" style="width: 50%;" />
</p>

```scala
val jammingTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5*Math.sin(3*i)))
```

```bash
Jamming Result: DenseVector(0.03840459974525514)
```

Again, our net is confident that this is _not_ an uptrend. Check!

### Hero to zero to hero

What about a curve first being a downtrend, but then recovering to the original level:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/herozero.png" style="width: 50%;" />
</p>

```scala
val heroZeroTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5*Math.cos(6*i) + 0.5))
```

```bash
HeroZero Result: DenseVector(0.024507592248881733)
```

Indeed, our net is pretty confident that this is _not_ an uptrend. Check!

### Oscillating sideways

Now, we want this oscillating sideways movement to be evaluated:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/oscillating.png" style="width: 50%;" />
</p>

```scala
val oscillating = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (Math.sin(100*i) / 3) + 0.5))
```

```bash
Oscillating Result: DenseVector(0.0332458093016362)
```

Strike, our net is confident that this is _not_ an uptrend. Check! Let's take a truly random sideways movement:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/random.png" style="width: 50%;" />
</p>

```scala
val random = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, Random.nextDouble))
```

```bash
Random Result: DenseVector(0.3636381886772248)
```

What's interesting here is that our net still says it is rather not a trend - which is correct - but with 0.36 it is not as certain as in the example before. If we examine the range <span class="tex" inline="true" data-expr="[0.0,0.55]"></span> we see a slight uptrend. I guess this is the reason for the increased uncertainty. However, check!

### Oscillating uptrend

I want to try an oscillating uptrend:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/oscillatingup.png" style="width: 50%;" />
</p>

```scala
val oscillatingUp = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100*i) / 3)))
```

```bash
Oscillating Up Result: DenseVector(0.9441733303039243)
```

Our net is pretty confident that this is an uptrend. Check! But what if we slighty randomize this trend, to make it seem more natural:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/realworld.png" style="width: 50%;" />
</p>

```scala
val realWorld = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100*i) / 3) * Random.nextDouble))
```

```bash
Real World Result: DenseVector(0.9900808890038473)
```

It's also correctly classified, check!

### Oscillating downtrend

Finally, let's try the oscillating downtrend:

<p class="gentle gentle-bottom text-center">
	<img src="/blog/natural-trend-detection/oscillatingdown.png" style="width: 50%;" />
</p>

```scala
val oscillatingDown = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, -i + (Math.sin(100*i) / 3) + 1))
```

```bash
Oscillating Down Result: DenseVector(0.0030360075614561453)
```

Again, our net is really confident that this is _not_ an uptrend. Check!

## Final thoughts

That's it. We trained a net to detect trends in a timeframe agnostic manner. The extension to more training data to force a stronger generalization is straight forward from here. Also, a multi-output layer is conceivable, so we could detect more states, e. g. uptrends, downtrends and no trends. 