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