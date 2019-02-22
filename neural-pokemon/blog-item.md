---
title: 'Neural Pokémon Cluster'
media_order: 'a.png,b.png,c.png,d.png,e.png,f.png,felix-new2.jpg,g.png,h.png,h2.png,h3.png,International_Pokémon_logo.svg.png,Rectifier_and_softplus_functions.svg,types.png,generalbycomp.png'
published: true
date: '24-04-2017 20:14'
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

Whenever I see Pokémons, they remind me of my younger days. Just recently there was this virtual reality revival. I never really got obsessive about the card games, I'd rather collect and trade them on the schoolyard for the social aspect. The video games were not bad actually. I spent quite some time gaining XP points there on my gameboy, he he.

<div style="text-align:center; width: 100%; padding: 4em;">
    <img src="/user/pages/blog/neural-pokemon/International_Pokémon_logo.svg.png" width="50%" />
</div>

So, on Kaggle I found this data set <a href="https://www.kaggle.com/abcsds/pokemon">Pokémon with stats</a>, and I couldn't resist to feed it into <a href="https://github.com/zenecture/neuroflow">NeuroFlow</a>. Surprisingly, there are 721 Pokémon out there, and the best thing is, they all come with stats, like attack and health points. Code is on <a href="https://github.com/zenecture/neuroflow/blob/master/playground/src/main/scala/neuroflow/playground/PokeMonCluster.scala">GitHub</a>, as usual.

# Learn yourself!

I want to find similar Pokémons, and I want help from a neural network. How can a net detect similarities between these monsters? By using compression. For instance, after training the identity function <span class="tex" inline="true" data-expr="f(x) \rightarrow x"></span>, the hidden layers will reveal interesting properties about how <span class="tex" inline="true" data-expr="x"></span>, our monsters, are structually composed.

<div style="text-align:center; width: 100%; margin-bottom: 2em;">
        <img src="/user/pages/blog/neural-pokemon/generalbycomp.png" width="33%" /><br>
        <small>(AutoEncoder)</small>
</div>

The identity function implies that the input and output dimensions are equal, <span class="tex" inline="true" data-expr="dim_{in} = dim_{out}"></span>. Further, if there is at least one hidden layer with a dimension <span class="tex" inline="true" data-expr="dim_{in} > dim_c < dim_{out}"></span>, this implies a certain degree of (de-)compression, since the net loses information by reducing to <span class="tex" inline="true" data-expr="dim_c"></span>, whereas it restores the information using the weights of the remaining layers. Training under this _bottleneck_ forces the net to generalize similar inputs. Then, to find similar Pokémons, we would have to query our nets bottleneck layer and look for close matches.

# Feature selection

Before we can learn the identity, we have to define it. The set comes with 721 Pokémon, each of them characterized by a certain set of attributes. Let's reduce the feature set to the most striking ones. We only care for _type1_ (like fire, ice, ...), _total_, _hp_, _attack_, and _defense_ of a Pokémon. All features are numbers (continuous variables), except for _type1_, which is a multi-class vector.

```scala
def toVector(p: Pokemon): Vector[Double] = p match {
  case Pokemon(_, t1, t2, tot, hp, att, defe, spAtk, spDef, speed, gen, leg) =>
    ζ(types.size).updated(t1, 1.0) ++ ->(tot / maximums._1) ++ ->(hp / maximums._2) ++ 
      ->(att / maximums._3) ++ ->(defe / maximums._4) // attributes normalized to be <= 1.0
}
val xs = pokemons.map(p => toVector(p))
val dim = xs.head.size // == 23
```

This maps a CSV line representing a Pokémon to a 23-dimensional vector, yielding our training data _xs_. Next, we need to carefully design our network. Let's import a dense network:

```scala
import neuroflow.core._
import neuroflow.core.Activator._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._
```

We want to start with random weights and <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">rectified linear units (ReLU)</a>. Compared to the <a href="https://en.wikipedia.org/wiki/Logistic_function">Sigmoid</a> activator, the ReLU is a bit more simple. And simplicity is good, it can be the basis for something complex as a neural net. 

<div style="text-align:center; width: 100%; padding: 2em;" id="relu">
        <img src="/user/pages/blog/neural-pokemon/relu.png" width="33%" /><br>
</div>

The ReLU can give faster training than Sigmoid or Tanh, because the gradient is constant and not vanishing; no matter where the positive weights stand, the gradient is always 1. And sparsity comes for free, the weights make cells act like on-off switches quickly and this is where the non-linearity comes from.

```scala
val net =
Network(
  Vector(dim)                 ::
  Dense(3, Linear)            ::
  Dense(dim / 2, ReLU)        ::
  Dense(dim, ReLU)            ::   SquaredError(),
  Settings[Double](
    iterations = 5000, 
    prettyPrint = true, 
    learningRate = { case (_, _) => 1E-5 }
  )
)

net.train(xs, xs)
xs.map(x => net(x))
```

The input and output layers have the same dimension _dim_. The actual output of the model is the central cluster layer, which is 3-dimensional. This is a good thing, because we can visualize _where_ and _how_ the net stores information. Further, we want to use a summed up linear combination of the original input to compress, so the central layer gets the linear (or identity) function. The layer after the cluster layer is rather intuitive, to pre-group a little using ReLUs before reaching the output layer. We wait for 500 iterations and for the sake of beauty, we want to _prettyPrint_ our net on console.

```bash




             _   __                      ________
            / | / /__  __  ___________  / ____/ /___ _      __
           /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
          / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
         /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


            Version : 1.3.3

            Network : neuroflow.nets.cpu.DenseNetwork
               Loss : neuroflow.core.SquaredError
             Update : neuroflow.core.Vanilla

             Layout : 23 Vector
                      3 Dense(x)
                      11 Dense (R)
                      23 Dense (R)
                    
            Weights : 355 (≈ 0,00270844 MB)
          Precision : Double



    
         O                 O
         O                 O
         O           O     O
         O           O     O
         O     O     O     O
         O     O     O     O
         O           O     O
         O           O     O
         O                 O
         O                 O



neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:26:825] Training with 800 samples, batch size = 800, batches = 1 ...
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:462] Iteration 1 - Loss 0,984109 - Loss Vector 0.4342741321580766  0.12354959298863942  4.767386946017951  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:508] Iteration 2 - Loss 0,523299 - Loss Vector 0.24434263228748687  0.06230241858166424  2.6454113282008813  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:535] Iteration 3 - Loss 0,371047 - Loss Vector 0.17849810378111158  0.04153676504335314  1.9031709188244275  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:560] Iteration 4 - Loss 0,291979 - Loss Vector 0.14599273484390507  0.0311696796636946  1.500094654118244  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:580] Iteration 5 - Loss 0,242986 - Loss Vector 0.12756149089729327  0.025110515177868865  1.2417398168753926  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:594] Iteration 6 - Loss 0,209563 - Loss Vector 0.11621659294818004  0.021296496834591873  1.0609521279456204  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:604] Iteration 7 - Loss 0,185329 - Loss Vector 0.10858822124655221  0.018846647283148665  0.9273539773987669  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:619] Iteration 8 - Loss 0,167051 - Loss Vector 0.1031596563714349  0.017185307369569098  0.824409293358061  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:632] Iteration 9 - Loss 0,152741 - Loss Vector 0.09916708918647826  0.016068084657543582  0.7421994416221095  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:643] Iteration 10 - Loss 0,141258 - Loss Vector 0.09614992073964508  0.015286892120987623  0.6750032695224172  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:27:656] Iteration 11 - Loss 0,131864 - Loss Vector 0.09366913014757446  0.014742771731776485  0.619233569973472  ... (23 total)
...
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:31:023] Iteration 499 - Loss 0,0318495 - Loss Vector 0.06269992174775436  0.017762885408949533  0.034436778622842265  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:31:026] Iteration 500 - Loss 0,0318402 - Loss Vector 0.0626990056751711  0.01776232163859681  0.03444182413724106  ... (23 total)
neuroflow.nets.cpu.DenseNetworkDouble - [12.12.2017 22:25:31:026] Took 500 iterations of 500 with Loss = 0,0318402
[success] Total time: 7 s, completed 12.12.2017 22:25:31
```

# Visualization and Dimensions > 3

After the training is done, we normalize our vectors to make sure that we have 3d-coordinates within range [0:1]. Using beloved <a href="http://www.gnuplot.info">gnuplot</a>, we can fly through the three-dimensional plot. If you want to have the same flight, you should do the training on your machine. However, let's deep dive into the result space...

<div style="text-align:center; width: 100%; padding: 2em;">
        <a href="/user/pages/blog/neural-pokemon/e.png"><img src="/user/pages/blog/neural-pokemon/e.png" width="80%" /></a><br>
        <p>... we find a couple of clusters here. Remember we used five features (type, total, hp, attack, defense), so the net would have to compromise between all 721 monsters.</p>

        <a href="/user/pages/blog/neural-pokemon/a.png"><img src="/user/pages/blog/neural-pokemon/a.png" width="80%" /></a><br>
        <p>Some groups are far away from each other, where the attribute <em>type</em> is the dominating clustering key. This comes naturally, because the type is a multi-class vector and contributes to the input vector with 19 (of 23) dimensions. Also, there is a big chunk concentrating at the bottom that is a bit hard to read, since the labels overlap. Eventually, we will discover some kind of numerical ordering in here...</p>
        
        <a href="/user/pages/blog/neural-pokemon/b.png"><img src="/user/pages/blog/neural-pokemon/b.png" width="80%" /></a><br>
        <p>... it seems like the net thoroughly orders the <em>type</em> groups, in such a way, that monsters with higher skills are placed at the top, whereas the ones with lower skills at the bottom (or vice versa, depending on the direction).</p>
        
        <a href="/user/pages/blog/neural-pokemon/c.png"><img src="/user/pages/blog/neural-pokemon/c.png" width="80%" /></a><br>
		<a href="/user/pages/blog/neural-pokemon/f.png"><img src="/user/pages/blog/neural-pokemon/f.png" width="80%" /></a><br><br><br>
        <p>What happens if we remove the dominating feature <em>type</em>, which contributes most to the input dimension, thus giving the net freedom to give up the strict separation between Pokémon types?</p>
        
        <div class="row gentle-l">
	        <div class="col-xs-6"><a href="/user/pages/blog/neural-pokemon/g.png"><img src="/user/pages/blog/neural-pokemon/g.png" width="100%" /></a></div>
			<div class="col-xs-6"><a href="/user/pages/blog/neural-pokemon/h2.png"><img src="/user/pages/blog/neural-pokemon/h2.png" width="100%" /></a></div>
        </div>
        
        <p class="gentle-l">... let's see. Mh, it pretty much does the same thing. It places similar Pokémons close to each other. So basically, what we can see is Pokémon2Vec. Note that here we used Sigmoids instead of ReLUs, to add more fuzzyness.</p>
</div>

Since we used the identity function for training, we can observe how the net represents all 23-dimensional Pokémon vectors in 3-dimensional space, using these coordinates to de-compress to the identity. I mean, sometimes I am baffled by these nets. I ask, what would a human do? What would I do? I'd be trying to cluster to group and generalize, to reduce the total amount of information to be stored, just as the net would do.

Now, if someone asks: "Hey, I have this _Pikachu_ card, can you give me monsters of the same type that would be a fair trade?". Sure, we can ask our net for Pokémons, which are close to _Pikachu_. We could use our three-dimensional plot to find these, but not always we face such situations. Sometimes, compressing to the third (or even below) dimension is infeasible due to the loss of information. But visualizing high-dimensional spaces is difficult. So we need something else to measure similarity, if we can't do it visually. Usually this is done using either the cosine similarity or the euclidean distance. So, for _P=Pikachu_, one would search monsters _X_ in some n-dimensional space that satisfy:

<p class="tex" data-expr="cosine(P,X) = \frac {P \cdot X}{|| X|| \cdot || Y||} \longrightarrow 1"></p>
<p class="tex" data-expr="euclidean(P, X) = \sqrt{\sum_{i=1}^n (P_i-X_i)^2} \longrightarrow 0"></p>  

Some prefer the euclidean over the cosine, because it cares for the actual distance, not the direction of two vectors. Two planets may point into the same direction from some origin, but they can be light years away from each other. Are they similar? Cosine would say so, euclidean wouldn't. On the other hand, if two planets stood close pointing into the opposite direction, cosine would say not similar, whereas euclidean would say similar.

Thanks for reading.