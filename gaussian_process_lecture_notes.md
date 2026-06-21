# Lecture Notes: Gaussian Process (GP) Regression

## 1. Introduction to Gaussian Processes

### 1.1 Parametric vs. Non-Parametric Models
In traditional **parametric** regression (e.g., linear regression, polynomial regression), we assume a specific functional form for the relationship between inputs $X$ and outputs $y$. For example:
$y = \beta_0 + \beta_1 x + \epsilon$

The goal is to find the best parameters ($\beta_0, \beta_1$) given the data. However, if the true relationship is not a straight line, the model will underfit.

A **non-parametric** model, like a Gaussian Process, does not assume a fixed functional shape. Instead of placing a prior over *parameters*, a GP places a prior over the *space of functions*. The complexity of the model grows with the size of the data.

### 1.2 What is a Gaussian Process?
**Definition:** A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian (Normal) distribution.

Think of it as an infinite-dimensional multivariate normal distribution. Just as a Gaussian distribution is fully specified by its mean vector and covariance matrix, a Gaussian Process is fully specified by a **mean function** $m(x)$ and a **covariance function (or kernel)** $k(x, x')$.

$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$

---

## 2. Mathematical Foundations

### 2.1 The Multivariate Normal (MVN) Distribution
To understand GPs, you must understand two key properties of MVNs:
1.  **Marginalization:** If you have a joint Gaussian distribution over two sets of variables, the marginal distribution of either set is also Gaussian.
2.  **Conditioning:** If you observe the values of one set of variables, the conditional distribution of the remaining variables is *still* Gaussian.

### 2.2 From MVN to GP
Imagine an MVN with 10 dimensions. We can plot a single sample from this MVN as 10 discrete points. If we increase the dimensions to 100, 1000, and eventually $\infty$, the discrete points merge into a continuous curve. This is what it means to sample a function from a GP prior.

---

## 3. The Mechanics of GP Regression

### 3.1 The Prior
Before seeing any data, we define our belief about what the function looks like. Usually, we assume the prior mean is zero: $m(x) = 0$. The shape (smoothness, amplitude, periodicity) is defined by the Kernel $k(x, x')$.

### 3.2 Adding Observations (The Posterior)
Suppose we have some training data (observations): $X_{train}$ and $y_{train}$. We want to predict the function values $f_{test}$ at some new test points $X_{test}$.

Because of the GP definition, the joint distribution of the training data and the test data is a massive Multivariate Normal:

$$
\begin{bmatrix} y_{train} \\ f_{test} \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} \mu_{train} \\ \mu_{test} \end{bmatrix}, \begin{bmatrix} K_{train, train} & K_{train, test} \\ K_{test, train} & K_{test, test} \end{bmatrix} \right)
$$

### 3.3 Conditioning
By applying the conditioning property of Gaussians, we can calculate the distribution of $f_{test}$ *given* the observed data $y_{train}$. The result is the **GP Posterior**:

$$f_{test} | X_{test}, X_{train}, y_{train} \sim \mathcal{N}(\mu_{post}, \Sigma_{post})$$

Where:
*   $\mu_{post} = K_{test, train} [K_{train, train}]^{-1} y_{train}$
*   $\Sigma_{post} = K_{test, test} - K_{test, train} [K_{train, train}]^{-1} K_{train, test}$

**Interpretation:** 
*   The mean prediction $\mu_{post}$ is a linear combination of the observed values. 
*   The uncertainty $\Sigma_{post}$ shrinks near the observed data points and expands as you move away from them.

---

## 4. Kernels (Covariance Functions)

The kernel is the heart of the GP. It defines the "similarity" between two points $x$ and $x'$. If $x$ and $x'$ are close, $k(x, x')$ is high, meaning $f(x)$ and $f(x')$ should be highly correlated.

### 4.1 Radial Basis Function (RBF) / Squared Exponential
$k(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2l^2}\right)$
*   **$\sigma^2$ (Variance/Amplitude):** How far the function wanders vertically.
*   **$l$ (Lengthscale):** How quickly the function wiggles. Large $l$ = slow, smooth changes. Small $l$ = rapid, jagged changes.
*   **Properties:** Assumes functions are infinitely differentiable (very smooth).

### 4.2 Matérn Kernel
The Matérn class is a generalization of the RBF. It is governed by a parameter $\nu$ that controls smoothness.
*   $\nu \to \infty$: Equivalent to RBF (infinitely smooth).
*   $\nu = 5/2$: Function is twice differentiable.
*   **Why use Matérn?** Real-world physical systems (like sports dynamics, financial markets, or terrain) are rarely perfectly smooth. They have sudden shifts or "roughness". Matérn is more realistic for these scenarios.

### 4.3 White Noise Kernel
$k(x, x') = \sigma_{noise}^2$ if $x = x'$, else $0$.
*   Used to model *aleatoric* uncertainty (noise in the measurements themselves). It allows the GP to pass *near* the training points rather than interpolating perfectly *through* them.

### 4.4 Composing Kernels
You can add or multiply kernels to create complex priors.
*   **Addition ($K_1 + K_2$):** Models data as a sum of two independent processes (e.g., a slow-moving trend + fast-moving seasonal noise).
*   **Multiplication ($K_1 \times K_2$):** Often acts as an "AND" operation (e.g., periodic AND decaying over time).

---

## 5. Handling Noise in GP Regression

In real data, observations are noisy: $y = f(x) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma_{noise}^2)$.
To handle this, we simply add the noise variance to the diagonal of the training covariance matrix:
$K_{train, train} \to K_{train, train} + \sigma_{noise}^2 I$

If the noise varies per observation (**heteroscedasticity**), we add a diagonal matrix with varying variances (e.g., passing `alpha = se_sq` in `scikit-learn` where `se_sq` is the standard error squared for each bin).

---

## 6. Pros and Cons of Gaussian Processes

### Advantages
1.  **Principled Uncertainty:** GP doesn't just give a point estimate; it gives a full predictive distribution (credible intervals).
2.  **Flexibility:** It can model highly non-linear functions without needing to specify a polynomial degree or architecture manually.
3.  **Hyperparameter Tuning:** Hyperparameters (like lengthscale and noise) can be optimized by maximizing the Marginal Likelihood, which inherently penalizes complex models and prevents overfitting.

### Disadvantages
1.  **Computational Complexity:** Inverting the $N \times N$ covariance matrix takes $\mathcal{O}(N^3)$ time and $\mathcal{O}(N^2)$ memory. GPs struggle when $N > 10,000$ points without using sparse approximations.
2.  **Kernel Selection:** Choosing the right kernel requires domain knowledge.
3.  **Interpretability:** Unlike linear regression, you don't get simple $\beta$ coefficients to interpret feature importance easily.

---

## 7. Applied Example: HockeyBayes

In the context of the **HockeyBayes** project:
*   **Why GP?** We don't know the mathematical equation for how a hockey team's expected goals (xG) change over time within a period. A GP lets the data dictate the trajectory shape while providing rigorous confidence bounds (credible intervals).
*   **Kernel Choice:** $Constant \times \text{Matérn}(5/2) + \text{WhiteNoise}$. We use Matérn instead of RBF because hockey play is subjected to discrete breaks (whistles, penalties, line changes) making the underlying momentum process slightly rough.
*   **Heteroscedastic Noise:** We pass the standard error squared (`se_sq`) of each time bin into the `alpha` parameter. This forces the GP to trust bins with many shots (low variance) more than bins with very few shots (high variance), correctly propagating uncertainty into the final plot.
