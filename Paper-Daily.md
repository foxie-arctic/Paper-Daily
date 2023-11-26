# Paper Daily: 3D Representation

## DMTeT

1. Deformable Tetrahedral Grid(DTG)
2. Volume Subdivision(VS)
3. Marching Tetrahedra(MT)
4. Neural Network(NN) Structure

![](./assets/DMTeT/pipeline.png)

1. DTG

* Deformable Tetrahedral Grid: 

$$ (V_T,T). $$

* Tetrahedra:

$$ T_k = \{v_{ak}, v_{bk}, v_{ck}, v_{dk}\}, T_k \in T, k\in\{1,2,...,K\},v_{ik}\in V_T.$$ 

* SDF value stored on vertices:

$$ s(v_i), v_i \in V_T. $$

* SDF value else where: $s(v)$ follows a barycentric interpolation.

2. VS

* Determine the surface tetrahedra $T_{surf}$: checking whether there are vertices in different SDF signs.

* Subdivide $T_{surf}$ for higher resolution:

$$v_{ac} = \frac{1}{2}(v_a+v_c),$$
$$s(v_{ac)} = \frac{1}{2}(s(v_a)+s(v_c)).$$

![](./assets/DMTeT/VS.png)

3. MT

* Define typology inside each grid depending on the signs of SDF values on its vertices.

![](./assets/DMTeT/MT.png)

* Determine locations of vertices: $s(v) = 0$

$$ v_{ab} = \frac{v_a \cdot s(v_b) - v_b \cdot s(v_a)}{s(v_b) - s(v_a)}.$$


4. NN Structure

* $v_i, \alpha_i$ are all learnable parameters

![](./assets/DMTeT/NN.png)

## 3D Gaussian Splatting


# Paper Daily: Score-based Generative Models

## SMLD

1. Score Matching for score estimation
2. Sampling with Langevin dynamics

Score matching with Langevin dynamics.

Suppose our datset consists of i.i.d. samples $\{x_i \in \mathbb{R}^D\}_{i=1}^N$ from an unknown data distribution $p_{data}(x)$. 

1. Score Matching for score estimation

* Score of a probability density $p(x)$ :

$$\nabla_x \log p(x).$$

* Score network to approximate the score of $p_{data}(x)$:

$$s_\theta: \mathbb{R}^D \to \mathbb{R}^D.$$

* Minimize objective:

$$\frac{1}{2}\mathbb{E}_{p_{data}(x)}\left[\|s_\theta(x)-\nabla_x\log p_{data}(x)\|_2^2\right]$$

$$\Leftrightarrow$$

$$\mathbb{E}_{p_{data}(x)}\left[tr(\nabla_xs_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|_2^2\right]+const,$$

under $s_\theta{x}$ is differentiable, $p_{data}(x)$ is differentiable, $\mathbb{E}\left[\|s_\theta(x)\|^2 \right]$ and $\mathbb{E}\left[\|\nabla_x\log p_{data}(x)\|^2 \right]$ are finite for any $\theta$, $p_{data}(x)s_\theta(x)$ goes to 0 for any $\theta$ when $\|x\| \to \infty$,where $\nabla_xs_\theta(x)$ denotes the Jacobian of $s_\theta(x)$.

Proof:

First, rewrite the original objective as:

$$J(\theta) = \int p_{data}(x) \left[\frac{1}{2}\|\nabla_x\log p_{data}(x)\|^2 + \frac{1}{2}\|s_\theta(x)\|^2 - \nabla_x\log p_{data}(x)^T s_\theta(x)\right]dx,$$

where we can ignore the $\frac{1}{2}\|\nabla_x\log p_{data}(x)\|^2$ part since it is unrelated with $\theta$ and it will be a constant. The integral of $\frac{1}{2}\|s_\theta(x)\|^2$ is exactly equal to $\mathbb{E}_{p_{data}(x)}\left[\frac{1}{2}\|s_\theta(x)\|^2\right]$. So all we need to prove is:

$$\int p_{data}(x) \left[- \nabla_x\log p_{data}(x)^T s_\theta(x)\right]dx = \int p_{data}(x) \left[tr(\nabla_xs_\theta(x))\right]dx,$$

i.e,

$$\int p_{data}(x) \left[- \nabla_x\log p_{data}(x)^T s_\theta(x)\right]dx = \int p_{data}(x) \left[\Sigma_{i=1}^N \frac{\partial s_\theta(x)_i}{\partial x_i}\right]dx.$$

To further prove this part, we first examine:

$$\int p_{data}(x) \left[- \nabla_x\log p_{data}(x)^T s_\theta(x)\right]dx = -\Sigma_{i=1}^N \int p_{data}(x) \nabla_{x_i}\log p_{data}(x) s_\theta(x)_i dx,$$

where we consider each $i$ seperately applying chain rule:

$$-\int p_{data}(x) \frac{\partial\log p_{data}(x)}{\partial x_i} s_\theta(x)_i dx = -\int p_{data}(x)\frac{1}{p_{data}(x)}\frac{\partial p_{data}(x)}{\partial x_i} s_\theta(x)_i dx, $$

$$= -\int \frac{\partial p_{data}(x)}{\partial x_i} s_\theta(x)_i dx.$$

To further prove the equation, we would like to introduce the lemma below:

$$\lim_{a \to \infty, b \to -\infty} f(a,x_2,\dots,x_n)g(a,x_2,\dots,x_n) - f(b,x_2,\dots,x_n)g(b,x_2,\dots,x_n),$$

$$=\int_{-\infty}^{\infty} f(x)\frac{\partial g(x)}{\partial x_1}dx_1 + \int_{-\infty}^{\infty} g(x)\frac{\partial f(x)}{\partial x_1}dx_1,$$

which is obvious when you regard $x_1$ as the only variable of the functions and other variables remain fixed. The only thing you need to do is to integrate over $x_1 \in \mathbb{R}$:

$$\frac{\partial f(x)g(x)}{\partial x_1} = f(x)\frac{\partial g(x)}{\partial x_1} + g(x)\frac{\partial f(x)}{\partial x_1}.$$

With this lemma, we can rewrite $-\int \frac{\partial p_{data}(x)}{\partial x_1} s_\theta(x)_1 dx$ as:

$$-\int \frac{\partial p_{data}(x)}{\partial x_1} s_\theta(x)_1 dx = -\int \left[ \int \frac{\partial p_{data}(x)}{\partial x_1} s_\theta(x)_1 dx_1 \right] d(x_2,\dots,x_n),$$

$$= -\int \left[\lim_{a \to \infty, b \to -\infty}\left[ p_{data}(a,x_2,\dots,x_n)s_\theta(a,x_2,\dots,x_n)_1 - p_{data}(b,x_2,\dots,x_n)s_\theta(b,x_2,\dots,x_n)_1\right]\right]$$

$$d(x_2,\dots,x_n)$$

$$+\int\int \frac{\partial s_\theta(x)_1}{\partial x_1} p_{data}(x) dx_1 d(x_2,\dots,x_n).$$

Since we assume that $p_{data}(x)s_\theta(x)$ goes to 0 for any $\theta$ when $\|x\| \to \infty$, then:

$$-\int \frac{\partial p_{data}(x)}{\partial x_1} s_\theta(x)_1 dx = \int \frac{\partial s_\theta(x)_1}{\partial x_1} p_{data}(x) dx,$$

and we finished the proof.

However, score matching is expensive since we need to compute and backpropagate $tr(\nabla_xs_\theta(x))$, where $x$ is a high-dimensional data.

* Denoising score matching:

Denoising score matching is a variant of score matching that
completely circumvents $tr(\nabla_xs_\theta(x))$.

Recall the minimize objective:

$$\frac{1}{2}\mathbb{E}_{p_{data}(x)}\left[\|s_\theta(x)-\nabla_x\log p_{data}(x)\|_2^2\right].$$

The problem is, we can not acquire an explicit representation of $p_{data}(x)$. To fix this, we may use a Parzen windows density estimator $q_{\sigma}(\widetilde{x})$.

The Parzen window density estimator is a non-parametric method used for estimating the probability density function (PDF) of a random variable. It is particularly useful when the underlying distribution is not known or when the data is limited. The Parzen window approach is based on the idea of placing a window (or kernel) around each data point and using these windows to estimate the overall density.

Here's a brief overview of how the Parzen window density estimator works(in 1D case):

First it needs to choose a window/kernel function. A kernel function should be symmetric and integrate to one. A general example is a Gaussian kernel:

$$K(u) = \frac{1}{\sqrt{2\pi}}\exp\{-\frac{1}{2}u^2\}.$$

For each data point $x_i$​, a window is centered at that point. The choice of the window width, often denoted as $h$ (bandwidth), determines the smoothness of the estimated density. A larger bandwidth results in a smoother estimate, but it might oversmooth the density and lose fine details. Conversely, a smaller bandwidth captures more details but may be sensitive to noise.

The density estimate $q_{\sigma}(\widetilde{x})$ at a point $\widetilde{x}$ is obtained by summing up the contributions of all data points, each scaled by the kernel function and the window width $\sigma$:

$$q_{\sigma}(\widetilde{x})=\frac{1}{n\sigma}\Sigma_{i=1}^n  K(\frac{\widetilde{x}-x_i}{\sigma}).$$

The Parzen window estimator is sensitive to the choice of the kernel function and bandwidth. Cross-validation or other methods can be used to select an optimal bandwidth for a given dataset.

While the Parzen window estimator is simple and intuitive, it may not perform well in high-dimensional spaces due to the "curse of dimensionality." Other techniques like kernel density estimation with different bandwidth selection methods or more advanced density estimation methods (e.g., mixture models) might be considered for such cases.

So we replace $p_{data}(x)$ with $q_{\sigma}(\widetilde{x})$ in the objective above.

In this case, we get a new objective along with its equivalent form:

$$\frac{1}{2}\mathbb{E}_{q_{\sigma}(\widetilde{x})}\left[\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2\right]$$

$$\Leftrightarrow$$

$$\frac{1}{2}\mathbb{E}_{q_{\sigma}(\widetilde{x}|x)p_{data}(x)}\left[\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x)\|_2^2\right],$$

where the perturbed data distribution $q_{\sigma}(\widetilde{x})=\int q_{\sigma}(\widetilde{x}|x) p_{data}(x)dx$.

Proof:

First we rewrite the objective as:

$$\frac{1}{2}\mathbb{E}_{q_{\sigma}(\widetilde{x})}\left[\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2\right],$$

$$=\frac{1}{2}\int(\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2)q_{\sigma}(\widetilde{x})d\widetilde{x},$$

$$=\frac{1}{2}\int(\|s_\theta(\widetilde{x})\|_2^2 + \|\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2)q_{\sigma}(\widetilde{x})d\widetilde{x} -\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\rangle q_{\sigma}(\widetilde{x})d\widetilde{x}.$$

Take a closer examine to the last term:

$$\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\rangle q_{\sigma}(\widetilde{x})d\widetilde{x},$$

$$=\int\langle s_\theta(\widetilde{x}), \frac{1}{q_{\sigma}(\widetilde{x})}\nabla_\widetilde{x}q_{\sigma}(\widetilde{x})\rangle q_{\sigma}(\widetilde{x})d\widetilde{x},$$

$$=\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}q_{\sigma}(\widetilde{x})\rangle d\widetilde{x},$$

$$=\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\int q_{\sigma}(\widetilde{x}|x) p_{data}(x)dx\rangle d\widetilde{x},$$

$$=\int\langle s_\theta(\widetilde{x}), \int \nabla_\widetilde{x}q_{\sigma}(\widetilde{x}|x) p_{data}(x)dx\rangle d\widetilde{x},$$

$$=\int\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}q_{\sigma}(\widetilde{x}|x) p_{data}(x)\rangle dxd\widetilde{x},$$

$$=\int\int\langle s_\theta(\widetilde{x}), q_{\sigma}(\widetilde{x}|x)\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x) p_{data}(x)\rangle dxd\widetilde{x},$$

$$=\int\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x) \rangle q_{\sigma}(\widetilde{x}|x)p_{data}(x)dxd\widetilde{x}.$$

Then, we put this term back to the objective:

$$=\frac{1}{2}\int(\|s_\theta(\widetilde{x})\|_2^2 + \|\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2)q_{\sigma}(\widetilde{x})d\widetilde{x} -\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\rangle q_{\sigma}(\widetilde{x})d\widetilde{x},$$

$$=\frac{1}{2}\int\int(\|s_\theta(\widetilde{x})\|_2^2 + \|\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})\|_2^2)q_{\sigma}(\widetilde{x}|x)p_{data}(x)dxd\widetilde{x}$$

$$-\int\int\langle s_\theta(\widetilde{x}), \nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x) \rangle q_{\sigma}(\widetilde{x}|x)p_{data}(x)dxd\widetilde{x},$$

$$=\frac{1}{2}\mathbb{E}_{q_{\sigma}(\widetilde{x}|x)p_{data}(x)}\left[\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x)\|_2^2\right].$$

And we finished the prove.

However, although $s_{\theta^*}(\widetilde{x})=\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x})$ (where ${\theta^*}=argmin_{\theta}(\frac{1}{2}\mathbb{E}_{q_{\sigma}(\widetilde{x}|x)p_{data}(x)}\left[\|s_\theta(\widetilde{x})-\nabla_\widetilde{x}\log q_{\sigma}(\widetilde{x}|x)\|_2^2\right])$) almost  surely, $\sigma$ has to be small enough to make the approximation $p_{data}(x)\approx q_{\sigma}(\widetilde{x})$ come true. Thus making it impossible to directly sample from a simple distribution like standard gaussian(if $\sigma$ can be large enough, $q_{\sigma}(\widetilde{x})$ will approximate to some simple distribution like gaussian).

* Sliced score matching:

Sliced score matching uses random projections to approximate
$tr(\nabla_{\theta}s_{\theta}(x))$ in score matching. 

:hammer: :wrench:



















## DDPM

# Paper Daily: 2D-lifted-3D Guidances with 3D Priors

## Zero-123

## MVDream

## SweetDreamer


# Paper Daily: 2D-lifted-3D Applications with 3D Priors 

## DreamCraft3D

1. Geometry Alignment with generated/provided image
2. Score Distillation Sampling(SDS) from a combination of 3D prior model and T2I model
3. Bootstrapped Score Distillation(BSD) to specifically boost the texture.

![](./assets/DreamCraft3D/pipeline.png) 

1. Geometry Alignment

* RGB Loss:

$$\mathcal{L}_{RGB} = \|\hat{m} \odot (\hat{x}-g(\theta;\hat{c}))\|_2, $$

where $\hat{m}$ is the mask, $\hat{x}$ is the reference image, $\hat{c}$ is the corresponding camera pose.

* Mask Loss:

$$\mathcal{L}_{mask} = \|\hat{m} - g_m(\theta;\hat{c})\|_2,$$

where $g_m$ renders the silhouette.

* Depth Loss:

$$\mathcal{L}_{depth} = -\frac{conv(d,\hat{d})}{\sigma(d)\sigma(\hat{d})},$$

where $\hat{d}$ is the depth prediction from a off-the-shelf single-view estimator.

* Normal Loss:

$$\mathcal{L}_{normal} = -\frac{n \cdot \hat{n}}{\|n\|_2 \cdot \|\hat{n}\|_2},$$

where $\hat{n}$ is the normal prediction from a off-the-shelf single-view estimator.

2. SDS

* SDS on T2I model:

$$ \nabla_{\theta}\mathcal{L}_{SDS}(\phi,g(\theta)) = \mathbb{E}_{t,\epsilon}\left[ \omega(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\frac{\partial x}{\partial \theta}\right], $$ 

where $\epsilon_\phi$ comes from T2I model: DeepFloyd IF based model, which operates on 64*64 pixel space.

* SDS on 3D prior model:

$$ \nabla_{\theta}\mathcal{L}_{3D-SDS}(\phi,g(\theta)) = \mathbb{E}_{t,\epsilon}\left[ \omega(t)(\epsilon_\phi(x_t;\hat{x},c,y,t)-\epsilon)\frac{\partial x}{\partial \theta}\right], $$ 

where $\epsilon_\phi$ comes from 3D prior model: Zero123.

* Hybrid SDS Loss:

$$\nabla_{\theta}\mathcal{L}_{hybrid}(\phi,g(\theta))=\nabla_{\theta}\mathcal{L}_{SDS}(\phi,g(\theta)) + \mu\nabla_{\theta}\mathcal{L}_{3D-SDS}(\phi,g(\theta)),$$

where $\mu = 2$. 

* Progressive view training: progressively enlarge the training views, gradually propagating the well-established geometry to $360\degree$ results.

* Diffusion timestep annealing: sampling larger diffusion timestep t from the range [0.7, 0.85] when computing $\nabla_{\theta}\mathcal{L}_{hybrid}(\phi,g(\theta))$ to provide the global structure, then linearly annealing
the t sampling range to [0.2, 0.5] over hundreds of iterations to refine
the structural details.

* 3D representation: NeuS in coarse stage, DMTeT in fine stage.

3. BSD

* Variational Score Distillation(VSD):

$$\mathcal{L}_{VSD} = D_{KL}(q^\mu(x_0|y)||p(x_0|y)),$$

$$\nabla_{\theta}\mathcal{L}_{VSD}(\phi,g(\theta)) = \mathbb{E}_{t,\epsilon}\left[ \omega(t)(\epsilon_\phi(x_t;y,t)-\epsilon_{lora}(x_t;y,t,c))\frac{\partial x}{\partial \theta}\right],$$

where $\epsilon_{lora}$ estimates the score of the rendered images using a LoRA (Low-rank adaptation).

* DreamBooth for fine-tuning using the multi-view rendered images:

$$ x_r = r_{t'}(x), $$

where $x_r$ stands for an augmented image renderings.

$$ x_{t'} = \alpha_{t'}x_0 + \sigma_{t'}\epsilon,$$

By choosing a large $t'$ , these augmented images reveal high-frequency details at the cost of the fidelity to the original renderings.

During finetuning, the camera parameter of each view is introduced as an additional condition. 

Initially, the 3D mesh yields blurry multi-view renderings. We adopt a large diffusion $t'$ to augment their texture quality while introducing some 3D inconsistency. The DreamBooth model trained on these augmented renderings obtains a unified 3D concept of the scene to guide texture refinement. As the 3D mesh reveals finer textures, we reduce the diffusion noises introduced to the image renderings, so the DreamBooth model learns from more consistent renderings and better captures the image distribution faithful to evolving views. In this cyclic process, the 3D mesh and diffusion prior mutually improve in a bootstrapped manner. 

* BSD Loss:

$$\nabla_{\theta}\mathcal{L}_{BSD}(\phi,g(\theta)) = \mathbb{E}_{t,\epsilon, c}\left[ \omega(t)(\epsilon_{DreamBooth}(x_t;y,t,r_{t'}(x),c)-\epsilon_{lora}(x_t;y,t,c))\frac{\partial x}{\partial \theta}\right],$$

![](./assets/DreamCraft3D/BSD.png) 

## Wonder3D

## SyncDreamer

## One-2-3-45

## Magic123

## Consistent123