# Power-SMC formulation and implementation notes

This note translates the formulation in `arxiv.org/abs/2602.10273` into the implementation in this repository. The goal is to make the code easy to audit against the paper rather than to restate the paper at a high level.

## 1. Target distribution

For a prompt $x$, the paper defines the sequence-level power target

$$
\begin{aligned}
\pi_\alpha(y \mid x) &\propto p_\theta(y \mid x)^\alpha, \\
\alpha &\ge 1.
\end{aligned}
$$

Using the autoregressive factorization,

$$
p_\theta(y \mid x) = \prod_{t=1}^{|y|} p_\theta(y_t \mid x, y_{<t}).
$$

The key point is that this is a sequence-level transformation. It is not the same as applying temperature token by token and then multiplying the resulting token probabilities. The implementation therefore keeps a proposal distribution and a separate importance-weight correction.

## 2. Prefix flow used by Power-SMC

The paper introduces the prefix targets

$$
\gamma_t(y_{1:t} \mid x) = p_\theta(y_{1:t} \mid x)^\alpha.
$$

This produces a standard Sequential Monte Carlo flow over prefixes. In code, the state of one particle at time $t$ is represented by:

- the generated token prefix,
- the cumulative base-model log probability $\log p_\theta(y_{1:t} \mid x)$,
- the cumulative log importance weight,
- a termination flag indicating whether EOS has already been emitted.

These quantities are maintained inside `samplers/power_smc.py`.

## 3. Proposal and exact incremental correction

For a generic prefix-only proposal $q_t(\cdot \mid x, y_{<t})$, the paper derives the exact incremental importance weight

$$
\omega_t
= \frac{p_\theta(y_t \mid x, y_{<t})^\alpha}{q_t(y_t \mid x, y_{<t})}.
$$

The implementation uses log-space updates for stability:

$$
\log \omega_t
= \alpha \, \log p_\theta(y_t \mid x, y_{<t})
   - \log q_t(y_t \mid x, y_{<t}).
$$

Concretely, in `PowerSMCSampler.generate`:

- `base_log_probs = log_softmax(logits)` computes $\log p_\theta(\cdot \mid x, y_{<t})$.
- `proposal_log_probs = log_softmax(alpha * logits)` computes the locally optimal proposal $q_t$.
- the sampled token contributes `alpha * chosen_base_log_prob - chosen_proposal_log_prob` to the particle log weight.

This exactly matches Equation (9) in the paper when written in log space.

## 4. Why the proposal uses temperature $\tau = 1 / \alpha$

The theory section proves that among all proposals that depend only on the current prefix, the unique proposal minimizing conditional incremental-weight variance is

$$
q_t^{\star}(v \mid x, y_{<t})
\propto p_\theta(v \mid x, y_{<t})^\alpha.
$$

Because
$p_\theta(\cdot \mid x, y_{<t}) = \operatorname{softmax}(\texttt{logits})$,
this is equivalent to sampling from
$\operatorname{softmax}(\alpha \cdot \texttt{logits})$,
i.e. temperature $\tau = \alpha^{-1}$.

The implementation follows this result literally. It does not use the base model distribution as the Power-SMC proposal. Instead, it samples each active particle from

$$
\texttt{proposal\_log\_probs} = \log \operatorname{softmax}(\alpha_{\text{step}} \cdot \texttt{logits}).
$$

This is the locally optimal proposal from the paper, while the importance weights recover the exact sequence-level target.

## 5. ESS-triggered resampling

The paper uses standard SIR-style resampling when weights collapse. The implementation does the same:

1. normalize particle log weights with a log-softmax-style shift,
2. compute $\operatorname{ESS}(W) = \left(\sum_i W_i^2\right)^{-1}$,
3. trigger resampling when $\operatorname{ESS}(W) < \kappa N$,
4. use systematic resampling to produce ancestor indices,
5. copy particle ancestry and reset log weights.

This is implemented in:

- `systematic_resample` for ancestor selection,
- `PowerSMCSampler._normalized_weights` for weight normalization,
- `PowerSMCSampler.generate` for the ESS check and resampling loop.

The ancestor copy updates:

- generated token histories,
- done flags,
- cumulative base-model log probabilities,
- the current decode token,
- the KV cache.

## 6. Cache reordering and alignment with the paper

The paper’s systems appendix describes a three-tier cache strategy:

1. model-provided cache reordering hooks,
2. runtime cache-object reordering methods,
3. recursive tensor reindexing.

The helper `reorder_cache` mirrors that design:

- if the model exposes `_reorder_cache`, it is used first,
- otherwise a cache object method such as `reorder_cache` or `batch_select_indices` is used,
- otherwise the code recursively index-selects tensors inside tuples, lists, mappings, or object attributes.

This keeps the sampler compatible with ordinary Hugging Face generation caches, including the Qwen 3.5 cache used in the evaluation.

## 7. EOS handling

The paper treats EOS as an absorbing state. The implementation keeps a boolean `done` mask. Once a particle emits EOS:

- it is marked done,
- future sampled tokens are forced to EOS,
- its log prefix probability and log importance weight stop changing.

This preserves the absorbing-state semantics at the sequence level, while still allowing all particles to stay in one dense batch for simple batched decoding.

The final text is decoded only from tokens before the first EOS, via `BaseSampler.trim_after_eos`.

## 8. Exact exponent bridging ($\alpha$-ramping)

The paper introduces intermediate exponents

$$
1 = \alpha^{(0)} < \alpha^{(1)} < \cdots < \alpha^{(L)} = \alpha.
$$

to improve particle stability without changing the final target. The implementation exposes this through `ramp_steps`.

The schedule built by `PowerSMCSampler.build_alpha_schedule` is:

- constant $\alpha$ when `ramp_steps = 0`,
- linear from $1$ to $\alpha$ over the first `ramp_steps` decode steps otherwise.

The implementation uses the following exact accounting:

- the proposal at decode step $t$ uses the previous stage exponent,
- after sampling the token, a bridge update

$$
\Delta \log W_t
= \left(\alpha_t - \alpha_{t-1}\right)
   \log p_\theta(y_{1:t} \mid x).
$$

is added.

This matches the appendix derivation and ensures that the final particle weights still target the same $\pi_\alpha$.

## 9. Relationship to the baseline samplers

The repository also includes two baselines built on the same prompt handling and cached decode utilities:

- `GreedySampler`: always chooses $\operatorname*{argmax}_v \, p_\theta(v \mid x, y_{<t})$.
- `StochasticSampler`: samples directly from the base-model token distribution, optionally with temperature.

These are useful controls because they separate the effect of Power-SMC’s sequence-level correction from ordinary token-level decoding decisions.

## 10. Practical scope of this implementation

This repository implementation intentionally focuses on the core text-generation algorithm and a reproducible evaluation harness.

Included:

- exact Power-SMC importance updates,
- systematic resampling,
- ESS diagnostics,
- optional $\alpha$-ramping,
- cache-safe ancestry reordering for Hugging Face models,
- baseline greedy and stochastic samplers using the same model wrapper.

Not included:

- benchmark-specific task logic from the paper,
- specialized inference-engine optimizations,
- multimodal prompt packing beyond ordinary text prompts.

That means the implementation is faithful to the statistical formulation in the paper, but intentionally conservative on systems complexity so it remains easy to read and adapt.
