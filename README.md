# Personality as a Policy: RL Agent-Based Economic Simulation

An agent-based simulation of economic exchange in which four Big Five personality traits map onto an agent's behavioural parameters. Personality is **not** an input. It is the **action** the agent chooses, learned with a contextual bandit.

Two experiments:

- **Experiment A (randomised):** personality is drawn at random before any outcome exists, which gives an unconfounded estimate of each trait's effect on wealth gain.
- **Experiment B (learning):** agents learn a personality profile from experience. The question is whether they find a favourable one on their own.

Bachelor thesis, Faculty of Technical Sciences, University of Novi Sad (2026). Supervisor: Prof. Milan Rapaić.
Thesis title: *Simulating the Effect of Personality Traits on Economic Outcomes Using Reinforcement Learning*.

---

## Key results

| Result | Value |
|---|---|
| Sampled personalities (Exp. A) | 12,000 |
| Agreeableness vs. wealth gain, Pearson r | **−0.755** |
| Learning agents (Exp. B) | 1,000 independent agents × 15,000 episodes |
| Mean wealth gain, learned profiles | **+12.0** |
| Mean wealth gain, random profiles | **−27.9** |
| Initial wealth | lognormal, Gini ≈ 0.47 |

Learned agents drive agreeableness to about 0. Conscientiousness, extraversion and neuroticism stay spread over most of their range.

**Extraversion sign flip.** In Exp. A, extraversion correlates negatively with wealth gain. After learning (Exp. B) it correlates positively (about +0.68). This follows from the trade-margin formula, not from tuning (see [The model](#the-model)).

### Comparison with the literature (sign agreement only)

| Literature | Result |
|---|---|
| Negotiation outcomes (Barry & Friedman) | 3/3 signs match |
| Wealth (meta-analysis) | 2 matches, 0 misses |
| Earnings (meta-analysis) | 1 match, 1 miss |

Only signs are compared. Correlation magnitudes are not comparable with the literature (see [Limitations](#limitations)).

---

## The model

**Traits → behaviour.** Conscientiousness, agreeableness, extraversion and neuroticism each set the agent's movement speed, food consumption, trade radius, trade cooldown, trade rate and loss on bad encounters. Everything is defined in one function.

```python
def compute_effects_fast(personality):
    consc, agree, extra, neuro = personality
    eff[EFF_SPEED]        = 1.0 + 3.0 * consc
    eff[EFF_FOOD_DECAY]   = AGENT_FOOD_DECAY * (1.0 + 0.25 * consc)
    eff[EFF_NPC_FORCE]    = 0.5 * neuro
    eff[EFF_TRADE_RADIUS] = 1.0 + 2.0 * extra
    eff[EFF_COOLDOWN]     = 30.0 * (1.0 - 0.5 * extra)
    eff[EFF_AGREE_RATE]   = 1.0 - 0.4 * agree
    eff[EFF_AGREE_LOSS]   = 1.0 + agree
    return eff
```

Openness is not modelled, because no meaningful mapping to behaviour was found in this environment.

**Economy.** The agent buys food from suppliers at a wholesale price and sells it to traders at a retail price. A third NPC type (bully) takes money and food. Rates are scaled by `a = 1 − 0.4·agreeableness`. The margin of a full buy-and-sell cycle is:

```
margin(a) = 0.89·a − 0.78      → positive only for a > 0.876, i.e. agreeableness < 0.31
```

**Reward.** Per step, based on net worth (`wealth = money + 0.39·food`):

- +0.01 for survival
- ×1.00 on gains
- ×2.25 on losses (loss aversion, from Tversky & Kahneman)

The episode ends with a penalty if the agent runs out of food.

**Learning.** One decision per episode, so the problem is a contextual bandit and not an MDP.

- State: initial (money, food) on a 3×3 grid, giving 9 states.
- Action: a trait level from 11 values (0.0, 0.1, …, 1.0), chosen **independently per trait** (four value tables).
- Update, once per episode and per trait, with no bootstrap and no discounting:

```python
Q[p, s0, a_p] += lr * (G - Q[p, s0, a_p])
```

- Exploration: ε-greedy with decaying ε.

The code is Python. The main loop is compiled with Numba as a pure speed optimisation and does not change model behaviour.

---

## Limitations

- **Distributive exchange only.** One issue per trade at a fixed price, so agreeableness can only cost the agent. The results therefore match the *distributive* negotiation literature and do not describe earnings.
- **Agreeableness cost is built in.** The negative sign was present before any run. The extraversion sign flip and the neutral conscientiousness effect were not designed.
- **Literature was matched after the fact.** The negotiation literature was found after the model and results existed. The comparison is justified by the model's mechanics, not by a pre-registered criterion. The thesis says so explicitly.
- **Magnitudes are inflated.** The model has about five sources of variance. Real outcomes have hundreds. Traits are also sampled uniformly (SD ≈ 0.29), while real questionnaire scores are roughly normal with about half that spread.
- **Four traits, not five.** Openness is excluded.
- **Four independent value tables share one reward**, so each table treats the others as part of the environment. This is non-stationary, and convergence guarantees do not apply.
- **No game theory.** NPCs have no goals or memory. The agent can exploit the same trader repeatedly with no consequence.
- **Fixed mechanics.** The agent does not choose whether to trade and cannot see NPC positions or types.


## References

Barry & Friedman, personality and distributive negotiation. Tversky & Kahneman, loss aversion. Meta-analyses of Big Five traits and earnings and wealth: see the thesis bibliography for full citations.


