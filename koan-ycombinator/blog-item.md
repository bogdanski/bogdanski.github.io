---
title: 'Koan: YCombinator'
published: true
date: '01-06-2016 14:54'
visible: true
author: 'Felix Bogdanski'
authorimage: felix-new2.jpg
---

<link rel="stylesheet" href="/user/themes/zen/js/highlight-2/styles/monokai.css">
<script src="/user/themes/zen/js/highlight-2/highlight.pack.js"></script>
<script>hljs.initHighlightingOnLoad();</script>

We all work within some industry and often it's about time and money. But essentially, coding to me is art and we should have fun practicing our art. So, this time, let's talk about the art of coding. For this, I picked a very beautiful and elegant piece of code &ndash; the <em><a target="_blank" href="https://en.wikipedia.org/wiki/Fixed-point_combinator#Fixed_point_combinators_in_lambda_calculus">YCombinator</a></em> written in Scala<a href="#note-1"><sup>1</sup></a>:

```scala
object YCombinator {
  def apply[A, B](f: (A => B) => (A => B)): A => B = f(apply(f))(_)
}
```

You might have a rough idea about what this _YCombinator_ does, but can't grasp the whole concept of it, e. g. in terms of drawing a call stack in your head. What can we do with this baffling thing?

<p class="gentle gentle-bottom text-center">
  <img src="/blog/koan-ycombinator/recursionvoid.png" style="width: 50%;" />
</p>

Let's start with a recursive implementation of the faculty function _n! = n \* (n - 1) \* (n - 2) \* ... \* 1_:

```scala
object Factorial {
  def apply(n: Int): Int = if (n > 1) n * apply(n - 1) else 1
}

val a = Factorial(5) // is 120
```

This is a straightforward thing. Since _defs_ are method calls, thus functions with a this-pointer and a well-known memory location, we can express this recursive pattern by safely referring to _apply_ within _apply_. Now, what if we can't refer to our own logic within our logic, because we simply don't have a name to refer to? Can we still express this faculty function in a recursive manner?

```scala
object FactorialTailrec {
  def apply(n: Int): Int = {
    @tailrec def fac(i: Int, k: Int): Int = if (i > 1) fac(i - 1, k * (i - 1)) else k
    fac(n, n)
  }
}

val b = FactorialTailrec(5)
```

If we try it with a nested tail-recursive version, we could express _apply_ without using _apply_, but still we need to call _fac_ within _fac_, and since this is the very same pattern, just on a different level, all we can really do to express the faculty function without a reference to its own name is to do what the compiler would do with _@tailrec_: eliminate recursion through an imperative while-loop!

```scala
object FactorialIter {
  def apply(n: Int): Int = {
    var k = n
    var i = n - 1
    while (i > 1) {
      k = k * i
      i = i - 1
    }
    k
  }
}

val c = FactorialIter(5)
```

What do we get? Well, a long but working faculty function that does not refer to itself!<a href="#note-2"><sup>2</sup></a> And since even the most elegant functional wizardish code at one point has to be translated to imperative assembler instructions, we could argue that this is the only way to express recursive functions without a name.

Is it, really? Let's go back to our _YCombinator_ and try to define the faculty function with it:

```scala
object YCombinator {
  def apply[A, B](f: (A => B) => (A => B)): A => B = f(apply(f))(_)
}

val d = YCombinator[Int, Int](a => b => if (b > 1) b * a(b - 1) else 1)(5) // immediately applying 5! = 120
```

This variant of the faculty function is recursive, but it doesn't have a name to refer to within its definition. All it has are parameters _a_ and _b_. Apparently, we can use the _YCombinator_ to express nameless, anonymous, recursive functions. Granted. But, how does this work? Let's step into the beauty of it.

The function _f: (A => B) => (A => B)_ can be considered as a template. The first function _A => B_ is the continuation of the second function _A => B_. Using the _YCombinator_, we can define the recursive factorial in terms of _(a: Int => Int) => ((b: Int) => ...)_ where _a_ is the same logic as _(b => ...)_, but never the same instance. The job of the _YCombinator_ is to _pre-set_ this logic _a_ for all subsequent calls from inside _(b => ...)_. If we write:

```scala
object Overflominator {
  def apply[A, B](f: (A => B) => (A => B)): A => B = f(apply(f))
}
```

This would compile, because it is the same on the type-level. But there will be a stack overflow error as soon as our factorial function is passed at run-time. The heart of the _YCombinator_ is the partial application _(\_)_ in tail position, so let's rewrite it to explicitly show the ad-hoc application, which _tames_ the recursion:

```scala
object ExplicitYCombinator {
  def apply[A, B](f: (A => B) => (A => B)): A => B = a => f(apply(f))(a) // instead of (_)
}
```

So calling the _YCombinator_ with our factorial function will give us a function object. When we call this function object with a number, what will happen? Before it computes something with this number, it will create its own logic as function object _X_, which can be used by factorial function _A_ to continue, if need be, or terminate. When the current factorial _A_ needs to continue, it calls this function object _X_. When _X_ is called, it will do the same thing before handing out the desired next factorial function _B_: it will prepare the factorial function _B_ with its own logic as function object _Y_ and then return this next factorial function _B_, which, hence, when called from current factorial function _A_ via _X_, can continue the computation through the very same recursive mechanism _Y_ offers, or terminate. When it terminates, the call stack will unwind and give the desired result.

Thanks for reading.

------

<small id="note-1">1: The _YCombinator_ can be defined in a way which is <a target="_blank" href="https://github.com/zenecture/techtalk/blob/master/ycombinator/src/main/scala/ycombinator/Run/Run.scala#L62">closer to Curry's original formulation.</a> In this implementation, the recursion is realized by deftly interweaving a combinator case class instead of using native recursion. I find that this version is a little less intuitive to explain in plain english. So instead, I picked the native, recursive version to illustrate the concept of it. Although they are different, both implementations carry out the same idea.</small>

<small id="note-2">2: A true zen master would question this, of course.</small>