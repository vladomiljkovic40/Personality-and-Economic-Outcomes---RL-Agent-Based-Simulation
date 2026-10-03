"""
Simulacija uticaja crta licnosti na ekonomske ishode primenom ucenja potkrepljivanjem.

Agent se krece kroz dvodimenzionalni svet sa trgovcima, dobavljacima i nasilnicima.
Cetiri crte licnosti iz opsega [0,1] odredjuju njegovo ponasanje. Licnost se bira
jednom na pocetku epizode i ostaje nepromenjena do njenog kraja, pa je problem
formulisan kao kontekstualni bandit, a ne kao Q-ucenje.

Eksperiment A: 12.000 agenata sa nasumicno dodeljenom licnoscu, bez ucenja.
Eksperiment B: 1.000 nezavisnih agenata koji licnost uce kroz 15.000 epizoda.

Oznake konstanti:
  [ANKER]     izvedeno iz citiranog izvora, ne podesavati
  [SLOBODNO]  projektni izbor, u radu se navodi kao slobodan parametar
  [IZVEDENO]  izracunato iz konstanti iznad, ne postavljati rucno

Opis modela, kalibracije, eksperimenata i rezultata nalazi se u radu.
"""

import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from numba import njit

WIDTH, HEIGHT = 1200, 700
RADIUS = 14

# Ekonomija
AGENT_START_MONEY = 197.0    # [ANKER] dve nedelje medijane zarade (51.370 USD godisnje)
AGENT_START_FOOD  = 14.0     # [ANKER] nedelja dana hrane pri 2.000 kcal dnevno
AGENT_FOOD_DECAY  = 0.0175   # [ANKER] 2 jedinice dnevno kroz 14 dana i 1.600 koraka
FARMER_PRICE      = 0.39     # [ANKER] veleprodajna cena; maloprodajna * (1 - 0,22)
MERCHANT_PRICE    = 0.50     # [ANKER] maloprodajna cena; 5 USD po 1.000 kcal
TRADE_SIZE_MEAN   = 9.14     # [ANKER] prosecna transakcija u prodavnici, 45,70 USD
TRADE_SIZE_SPREAD = 0.6      # [SLOBODNO] sirina raspodele velicine jedne transakcije

# Nagrada: cilj je neto imovina = novac + FOOD_VALUE * hrana
FOOD_VALUE      = 0.39       # [ANKER] zalihe se vrednuju po nabavnoj, ne po prodajnoj ceni
REWARD_GAIN     = 1.0        # [ANKER] tezina dobitka
REWARD_LOSS     = 2.25       # [ANKER] averzija prema gubitku, Tverski i Kaneman (1992)
REWARD_SURVIVAL = 0.01       # [SLOBODNO] nagrada po koraku za prezivljavanje

START_WEALTH = AGENT_START_MONEY + FOOD_VALUE * AGENT_START_FOOD   # [IZVEDENO]
DEATH_PENALTY_FRAC = 1.00    # [SLOBODNO] udeo pocetne imovine koji se gubi pri gladovanju
DEATH_PENALTY = START_WEALTH * DEATH_PENALTY_FRAC * REWARD_LOSS    # [IZVEDENO]

# Svet
START_SPREAD      = 0.75     # [SLOBODNO] raspon pocetnih uslova tokom treninga
BULLY_MONEY_LOSS  = 3.0      # [SLOBODNO] gubitak novca po susretu sa nasilnikom
BULLY_FOOD_LOSS   = 1.0      # [SLOBODNO] gubitak hrane po susretu sa nasilnikom
MAX_NUM_MERCHANTS = 33       # [SLOBODNO]
MAX_NUM_FARMERS   = 33       # [SLOBODNO]
TOTAL_NUM_NPC     = MAX_NUM_MERCHANTS + MAX_NUM_FARMERS

# Ucenje
NUM_EPISODES      = 15000    # [SLOBODNO] broj epizoda po agentu
STEPS_PER_EPISODE = 1600     # [SLOBODNO] broj koraka po epizodi
RUNS_PER_GOAL     = 1000     # [SLOBODNO] broj nezavisnih agenata u eksperimentu B
N_EVAL_EPISODES   = 10       # [SLOBODNO] broj epizoda preko kojih se usrednjava ishod
LEARNING_RATE_START = 0.10   # [SLOBODNO]
LEARNING_RATE_END   = 0.005  # [SLOBODNO]
LEARNING_RATE_DECAY = (LEARNING_RATE_END / LEARNING_RATE_START) ** (1 / NUM_EPISODES)
EXPLORATION_START   = 1.0    # [SLOBODNO]
EXPLORATION_END     = 0.01   # [SLOBODNO]
EXPLORATION_DECAY   = (EXPLORATION_END / EXPLORATION_START) ** (1 / NUM_EPISODES)
CONVERGENCE_CHECK_EVERY = 500
CONVERGENCE_THRESHOLD   = 0.005
CONVERGENCE_PATIENCE    = 3
MIN_EPISODES            = 3500
FREEZE_PERSONALITY = True
RANDOMIZE_START    = True

# Diskretizacija stanja: 15.000 / (9 * 11) = 151,5 uzoraka po celiji
STATE_MONEY_BINS = 3         # [SLOBODNO]
STATE_FOOD_BINS  = 3         # [SLOBODNO]

START_GINI    = 0.47         # [ANKER] Dzinijev koeficijent nejednakosti
START_SIGMA   = 0.888        # [IZVEDENO] sqrt(2) * Phi^-1((START_GINI + 1) / 2)
START_CLIP_LO = 0.15         # [SLOBODNO] odsecanje donjeg repa, u odnosu na medijanu
START_CLIP_HI = 6.00         # [SLOBODNO] odsecanje gornjeg repa, u odnosu na medijanu
START_MONEY_MIN = AGENT_START_MONEY * START_CLIP_LO
START_MONEY_MAX = AGENT_START_MONEY * START_CLIP_HI
START_FOOD_MIN  = AGENT_START_FOOD  * (1.0 - START_SPREAD)
START_FOOD_MAX  = AGENT_START_FOOD  * (1.0 + START_SPREAD)
STATE_MONEY_MAX = START_MONEY_MAX * 1.2
STATE_FOOD_MAX  = START_FOOD_MAX  * 1.2
NUM_STATES = STATE_MONEY_BINS * STATE_FOOD_BINS


def _bin_of(v, vmax, nbins):
    x = v / vmax
    if x < 0.0: x = 0.0
    elif x > 0.9999: x = 0.9999
    return int(x * nbins)


assert MERCHANT_PRICE > FARMER_PRICE, "razmena mora biti profitabilna"
_eval_food_bin  = _bin_of(AGENT_START_FOOD,  STATE_FOOD_MAX,  STATE_FOOD_BINS)
_eval_money_bin = _bin_of(AGENT_START_MONEY, STATE_MONEY_MAX, STATE_MONEY_BINS)
_train_food_bins = {_bin_of(v, STATE_FOOD_MAX, STATE_FOOD_BINS)
                    for v in np.linspace(START_FOOD_MIN, START_FOOD_MAX, 4000)[:-1]}
_train_money_bins = {_bin_of(v, STATE_MONEY_MAX, STATE_MONEY_BINS)
                     for v in np.linspace(START_MONEY_MIN, START_MONEY_MAX, 4000)[:-1]}
assert _eval_food_bin in _train_food_bins, "celija hrane se ne posecuje tokom treninga"
assert _eval_money_bin in _train_money_bins, "celija novca se ne posecuje tokom treninga"

NUM_LEVELS = 11
NUM_PARAMS = 4
MAX_NPCS = MAX_NUM_MERCHANTS + MAX_NUM_FARMERS + int(TOTAL_NUM_NPC * 5 * 0.02) + 1
PERSONALITY_NAMES = ["Conscientiousness", "Agreeableness", "Extraversion", "Neuroticism"]
SHORT_NAMES = ["Consc", "Agree", "Extra", "Neuro"]

# Indeksi niza efekata
EFF_SPEED         = 0
EFF_NPC_FORCE     = 1
EFF_TRADE_RADIUS  = 2
EFF_COOLDOWN      = 3
EFF_AGREE_RATE    = 4
EFF_AGREE_LOSS    = 5
EFF_FOOD_DECAY    = 6
NUM_EFFECTS       = 7

@njit(fastmath=True)
# Jedino mesto na kome se crte licnosti pretvaraju u parametre ponasanja.
def compute_effects_fast(personality):
    eff = np.zeros(NUM_EFFECTS)
    consc = personality[0]
    agree = personality[1]
    extra = personality[2]
    neuro = personality[3]

    eff[EFF_SPEED]        = 1.0 + 3.0 * consc
    eff[EFF_FOOD_DECAY]   = AGENT_FOOD_DECAY * (1.0 + 0.25 * consc)

    eff[EFF_NPC_FORCE]    = 0.5 * neuro

    eff[EFF_TRADE_RADIUS] = 1.0 + 2.0 * extra
    eff[EFF_COOLDOWN]     = 30.0 * (1.0 - 0.5 * extra)

    eff[EFF_AGREE_RATE]   = 1.0 - 0.4 * agree
    eff[EFF_AGREE_LOSS]   = 1.0 + agree

    return eff


EPSILON_SCHEDULE = np.empty(NUM_EPISODES, dtype=np.float64)
LR_SCHEDULE = np.empty(NUM_EPISODES, dtype=np.float64)
eps_val = EXPLORATION_START
lr_val = LEARNING_RATE_START
for i in range(NUM_EPISODES):
    EPSILON_SCHEDULE[i] = eps_val
    LR_SCHEDULE[i] = lr_val
    eps_val = max(EXPLORATION_END, eps_val * EXPLORATION_DECAY)
    lr_val = max(LEARNING_RATE_END, lr_val * LEARNING_RATE_DECAY)

@njit(fastmath=True)
def distance_fast(x1, y1, x2, y2):
    dx = x2 - x1; dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)

@njit(fastmath=True)
def update_position_fast(x, y, vx, vy, radius):
    x += vx; y += vy
    if x - radius < 0.0: vx = abs(vx); x = float(radius)
    elif x + radius > WIDTH: vx = -abs(vx); x = float(WIDTH - radius)
    if y - radius < 0.0: vy = abs(vy); y = float(radius)
    elif y + radius > HEIGHT: vy = -abs(vy); y = float(HEIGHT - radius)
    return x, y, vx, vy

@njit(fastmath=True)
def draw_start_money():
    """Izvlaci pocetni novac iz lognormalne raspodele."""
    m = AGENT_START_MONEY * math.exp(START_SIGMA * np.random.normal())
    if m < START_MONEY_MIN: m = START_MONEY_MIN
    elif m > START_MONEY_MAX: m = START_MONEY_MAX
    return m


@njit(fastmath=True)
def get_state_fast(money, food):
    """Diskretizuje par (novac, hrana) u mrezu stanja."""
    m = money / STATE_MONEY_MAX
    if m < 0.0: m = 0.0
    elif m > 0.9999: m = 0.9999
    f = food / STATE_FOOD_MAX
    if f < 0.0: f = 0.0
    elif f > 0.9999: f = 0.9999
    return int(m * STATE_MONEY_BINS) * STATE_FOOD_BINS + int(f * STATE_FOOD_BINS)

@njit(fastmath=True)
def choose_actions_fast(q_tables, state, epsilon, rng_val1, rng_val2, rng_val3):
    actions = np.zeros(NUM_PARAMS, dtype=np.int32)
    personality = np.zeros(NUM_PARAMS)
    for p in range(NUM_PARAMS):
        # Pri izjednacenim vrednostima bira se najnizi nivo crte.
        best_a = 0; best_v = q_tables[p, state, 0]
        for a in range(1, NUM_LEVELS):
            v = q_tables[p, state, a]
            if v > best_v: best_v = v; best_a = a
        actions[p] = best_a
    if rng_val1 < epsilon:
        idx = int(rng_val2 * NUM_PARAMS) % NUM_PARAMS
        actions[idx] = int(rng_val3 * NUM_LEVELS) % NUM_LEVELS
    for p in range(NUM_PARAMS): personality[p] = actions[p] / 10.0
    return actions, personality


@njit(fastmath=True)
def compute_reward_fast(money, food, money_before, food_before):
    """Nagrada po koraku, racunata na promenu neto imovine."""
    wealth_before = money_before + FOOD_VALUE * food_before
    wealth        = money        + FOOD_VALUE * food
    dw = wealth - wealth_before

    r = REWARD_SURVIVAL
    if dw > 0.0: r += dw * REWARD_GAIN
    else:        r += dw * REWARD_LOSS
    return r


@njit(fastmath=True)
def get_trade_offer_fast(npc_type, eff, npc_money, npc_food):
    """Promena novca i hrane iz jedne interakcije sa nepokretnim likom."""
    # npc_type: 0 = trgovac, 1 = nasilnik, 2 = dobavljac
    size = TRADE_SIZE_MEAN * (1.0 - TRADE_SIZE_SPREAD
                              + 2.0 * TRADE_SIZE_SPREAD * np.random.random())

    if npc_type == 0:
        food_spent   = size
        money_gained = MERCHANT_PRICE * size * eff[EFF_AGREE_RATE]
        if npc_money >= money_gained: return money_gained, -food_spent
        return 0.0, 0.0
    elif npc_type == 2:
        money_spent = FARMER_PRICE * size
        food_gained = size * eff[EFF_AGREE_RATE]
        if npc_food >= food_gained: return -money_spent, food_gained
        return 0.0, 0.0
    elif npc_type == 1:
        money_loss = BULLY_MONEY_LOSS * eff[EFF_AGREE_LOSS]
        food_loss  = BULLY_FOOD_LOSS  * eff[EFF_AGREE_LOSS]
        if money_loss < 2.0: money_loss = 2.0
        if food_loss  < 0.5: food_loss  = 0.5
        return -money_loss, -food_loss
    return 0.0, 0.0

@njit(fastmath=True)
def train_one_run_jit(epsilon_schedule, lr_schedule, seed):
    """Trenira jednog nezavisnog agenta i vraca naucenu licnost i ishod."""
    np.random.seed(seed)
    q_tables = np.zeros((NUM_PARAMS, NUM_STATES, NUM_LEVELS))
    npc_xs = np.zeros(MAX_NPCS); npc_ys = np.zeros(MAX_NPCS)
    npc_vxs = np.zeros(MAX_NPCS); npc_vys = np.zeros(MAX_NPCS)
    npc_types = np.zeros(MAX_NPCS, dtype=np.int32)
    npc_money = np.zeros(MAX_NPCS); npc_food = np.zeros(MAX_NPCS)
    npc_cooldowns = np.zeros(MAX_NPCS, dtype=np.int32)

    NUM_SAMPLES = 200
    sample_interval = max(1, NUM_EPISODES // NUM_SAMPLES)
    p_samples = np.zeros((NUM_PARAMS, NUM_SAMPLES))
    r_samples = np.zeros(NUM_SAMPLES)
    sample_idx = 0

    eval_state = get_state_fast(AGENT_START_MONEY, AGENT_START_FOOD)
    prev_probe = np.zeros(NUM_PARAMS)
    stable_checks = 0
    actual_episodes = NUM_EPISODES
    final_personality = np.zeros(NUM_PARAMS)
    last_ep_reward = 0.0

    # Jedna epizoda: izbor licnosti, simulacija, jedno azuriranje tabele.
    for ep in range(NUM_EPISODES):
        eps = epsilon_schedule[ep]; lr = lr_schedule[ep]

        if RANDOMIZE_START:
            start_money = draw_start_money()
            start_food  = START_FOOD_MIN  + np.random.random() * (START_FOOD_MAX  - START_FOOD_MIN)
        else:
            start_money = AGENT_START_MONEY
            start_food  = AGENT_START_FOOD

        agent_x = WIDTH / 2.0 + (np.random.random() - 0.5) * 100.0
        agent_y = HEIGHT / 2.0 + (np.random.random() - 0.5) * 100.0
        agent_vx = (np.random.random() - 0.5) * 4.0
        agent_vy = (np.random.random() - 0.5) * 4.0
        agent_money = start_money; agent_food = start_food
        agent_radius = float(RADIUS + 2); total_trades = 0

        # Nasumican broj i raspored nepokretnih likova za ovu epizodu.
        num_m = np.random.randint(1, MAX_NUM_MERCHANTS + 1)
        num_f = np.random.randint(1, MAX_NUM_FARMERS + 1)
        num_b = int(TOTAL_NUM_NPC * np.random.randint(1, 6) * 0.02)
        idx = 0
        for _ in range(num_m):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 0
            npc_money[idx] = 250.0; npc_food[idx] = 7.0; npc_cooldowns[idx] = 0
            idx += 1
        for _ in range(num_b):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 1
            npc_money[idx] = 25.0; npc_food[idx] = 14.0; npc_cooldowns[idx] = 0
            idx += 1
        for _ in range(num_f):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 2
            npc_money[idx] = 25.0; npc_food[idx] = 56.0; npc_cooldowns[idx] = 0
            idx += 1
        actual_npcs = idx

        s0 = get_state_fast(agent_money, agent_food)
        # Licnost se bira jednom, iz pocetnog stanja, i ostaje zamrznuta.
        actions, personality = choose_actions_fast(
            q_tables, s0, eps,
            np.random.random(), np.random.random(), np.random.random())

        ep_reward = 0.0
        eff = compute_effects_fast(personality)
        cooldown_val = int(eff[EFF_COOLDOWN])
        if cooldown_val < 5: cooldown_val = 5
        interaction_dist = (agent_radius + RADIUS) * eff[EFF_TRADE_RADIUS]

        for step in range(STEPS_PER_EPISODE):
            money_before = agent_money; food_before = agent_food

            # Kretanje agenta: brzina od savesnosti, polje sile od neuroticizma.
            speed_sq = agent_vx*agent_vx + agent_vy*agent_vy
            if speed_sq > 0.0001:
                speed = math.sqrt(speed_sq)
                scale = eff[EFF_SPEED] / speed
                agent_vx *= scale; agent_vy *= scale
            for i in range(actual_npcs):
                dx = npc_xs[i]-agent_x; dy = npc_ys[i]-agent_y; dist_sq = dx*dx+dy*dy
                if dist_sq > 0.0 and dist_sq < 40000.0:
                    dist = math.sqrt(dist_sq); nx = dx/dist; ny = dy/dist
                    if npc_types[i] == 1:
                        agent_vx -= nx*eff[EFF_NPC_FORCE]; agent_vy -= ny*eff[EFF_NPC_FORCE]
                    else:
                        agent_vx += nx*eff[EFF_NPC_FORCE]; agent_vy += ny*eff[EFF_NPC_FORCE]
            agent_x, agent_y, agent_vx, agent_vy = update_position_fast(
                agent_x, agent_y, agent_vx, agent_vy, agent_radius)

            for i in range(actual_npcs):
                npc_xs[i] += npc_vxs[i]; npc_ys[i] += npc_vys[i]
                if npc_xs[i]-RADIUS < 0.0: npc_vxs[i] = abs(npc_vxs[i]); npc_xs[i] = float(RADIUS)
                elif npc_xs[i]+RADIUS > WIDTH: npc_vxs[i] = -abs(npc_vxs[i]); npc_xs[i] = float(WIDTH-RADIUS)
                if npc_ys[i]-RADIUS < 0.0: npc_vys[i] = abs(npc_vys[i]); npc_ys[i] = float(RADIUS)
                elif npc_ys[i]+RADIUS > HEIGHT: npc_vys[i] = -abs(npc_vys[i]); npc_ys[i] = float(HEIGHT-RADIUS)
                if npc_types[i] == 2:
                    nf = npc_food[i]+0.1
                    if nf > 70.0: nf = 70.0
                    npc_food[i] = nf
                elif npc_types[i] == 0:
                    nm = npc_money[i]+0.1
                    if nm > 300.0: nm = 300.0
                    npc_money[i] = nm
                if npc_cooldowns[i] > 0: npc_cooldowns[i] -= 1; continue
                dx = agent_x-npc_xs[i]; dy = agent_y-npc_ys[i]; dist_sq = dx*dx+dy*dy
                if dist_sq < interaction_dist*interaction_dist:
                    mc, fc = get_trade_offer_fast(npc_types[i], eff, npc_money[i], npc_food[i])
                    if mc < 0.0 and agent_money < -mc: continue
                    if fc < 0.0 and agent_food < -fc: continue
                    agent_money += mc; agent_food += fc
                    npc_money[i] -= mc; npc_food[i] -= fc; total_trades += 1
                    if agent_money < 0.0: agent_money = 0.0
                    if agent_food < 0.0: agent_food = 0.0
                    if npc_money[i] < 0.0: npc_money[i] = 0.0
                    if npc_food[i] < 0.0: npc_food[i] = 0.0
                    npc_cooldowns[i] = cooldown_val

            # Potrosnja hrane; prazna zaliha prekida epizodu uz kaznu.
            agent_food -= eff[EFF_FOOD_DECAY]
            if agent_food < 0.0: agent_food = 0.0

            ep_reward += compute_reward_fast(agent_money, agent_food,
                                             money_before, food_before)
            if agent_food <= 0.0:
                ep_reward -= DEATH_PENALTY
                break

        for p in range(NUM_PARAMS):
        # Kontekstualni bandit: jedno azuriranje po epizodi, za svaku crtu.
            old = q_tables[p, s0, actions[p]]
            q_tables[p, s0, actions[p]] = old + lr * (ep_reward - old)

        last_ep_reward = ep_reward
        for p in range(NUM_PARAMS):
            final_personality[p] = personality[p]

        if sample_idx < NUM_SAMPLES and ep >= sample_idx * sample_interval:
            probe_a, probe_p = choose_actions_fast(q_tables, eval_state, 0.0, 1.0, 0.0, 0.0)
            for p in range(NUM_PARAMS):
                p_samples[p, sample_idx] = probe_p[p]
            r_samples[sample_idx] = ep_reward
            sample_idx += 1

        # Rano zaustavljanje kada se pohlepno izabrana licnost ustali.
        if ep > 0 and ep % CONVERGENCE_CHECK_EVERY == 0 and ep >= MIN_EPISODES:
            probe_a, probe_p = choose_actions_fast(q_tables, eval_state, 0.0, 1.0, 0.0, 0.0)
            max_delta = 0.0
            for p in range(NUM_PARAMS):
                d = abs(probe_p[p] - prev_probe[p])
                if d > max_delta: max_delta = d
            if max_delta < CONVERGENCE_THRESHOLD: stable_checks += 1
            else: stable_checks = 0
            for p in range(NUM_PARAMS): prev_probe[p] = probe_p[p]
            if stable_checks >= CONVERGENCE_PATIENCE:
                actual_episodes = ep + 1
                while sample_idx < NUM_SAMPLES:
                    for p in range(NUM_PARAMS):
                        p_samples[p, sample_idx] = probe_p[p]
                    r_samples[sample_idx] = last_ep_reward
                    sample_idx += 1
                break

    s0 = get_state_fast(AGENT_START_MONEY, AGENT_START_FOOD)
    actions, personality = choose_actions_fast(q_tables, s0, 0.0, 1.0, 0.0, 0.0)
    for p in range(NUM_PARAMS):
        final_personality[p] = personality[p]
    eff = compute_effects_fast(personality)

    run_start_money = draw_start_money() if RANDOMIZE_START else AGENT_START_MONEY
    eval_money = 0.0; eval_food = 0.0; eval_trades = 0.0
    for _ev in range(N_EVAL_EPISODES):
        agent_x = WIDTH/2.0+(np.random.random()-0.5)*100.0
        agent_y = HEIGHT/2.0+(np.random.random()-0.5)*100.0
        agent_vx = (np.random.random()-0.5)*4.0; agent_vy = (np.random.random()-0.5)*4.0
        agent_money = run_start_money; agent_food = AGENT_START_FOOD
        agent_radius = float(RADIUS+2); total_trades = 0
        num_m = np.random.randint(1, MAX_NUM_MERCHANTS+1)
        num_f = np.random.randint(1, MAX_NUM_FARMERS+1)
        num_b = int(TOTAL_NUM_NPC*np.random.randint(1, 6)*0.02)
        idx = 0
        for _ in range(num_m):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 0; npc_money[idx] = 250.0; npc_food[idx] = 7.0; npc_cooldowns[idx] = 0; idx += 1
        for _ in range(num_b):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 1; npc_money[idx] = 25.0; npc_food[idx] = 14.0; npc_cooldowns[idx] = 0; idx += 1
        for _ in range(num_f):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 2; npc_money[idx] = 25.0; npc_food[idx] = 56.0; npc_cooldowns[idx] = 0; idx += 1
        actual_npcs = idx

        cooldown_val = int(eff[EFF_COOLDOWN])
        if cooldown_val < 5: cooldown_val = 5
        interaction_dist = (agent_radius+RADIUS)*eff[EFF_TRADE_RADIUS]

        for step in range(STEPS_PER_EPISODE):
            speed_sq = agent_vx*agent_vx+agent_vy*agent_vy
            if speed_sq > 0.0001:
                speed = math.sqrt(speed_sq)
                scale = eff[EFF_SPEED] / speed
                agent_vx *= scale; agent_vy *= scale
            for i in range(actual_npcs):
                dx = npc_xs[i]-agent_x; dy = npc_ys[i]-agent_y; dist_sq = dx*dx+dy*dy
                if dist_sq > 0.0 and dist_sq < 40000.0:
                    dist = math.sqrt(dist_sq); nx = dx/dist; ny = dy/dist
                    if npc_types[i] == 1:
                        agent_vx -= nx*eff[EFF_NPC_FORCE]; agent_vy -= ny*eff[EFF_NPC_FORCE]
                    else:
                        agent_vx += nx*eff[EFF_NPC_FORCE]; agent_vy += ny*eff[EFF_NPC_FORCE]
            agent_x, agent_y, agent_vx, agent_vy = update_position_fast(
                agent_x, agent_y, agent_vx, agent_vy, agent_radius)

            for i in range(actual_npcs):
                npc_xs[i] += npc_vxs[i]; npc_ys[i] += npc_vys[i]
                if npc_xs[i]-RADIUS < 0.0: npc_vxs[i] = abs(npc_vxs[i]); npc_xs[i] = float(RADIUS)
                elif npc_xs[i]+RADIUS > WIDTH: npc_vxs[i] = -abs(npc_vxs[i]); npc_xs[i] = float(WIDTH-RADIUS)
                if npc_ys[i]-RADIUS < 0.0: npc_vys[i] = abs(npc_vys[i]); npc_ys[i] = float(RADIUS)
                elif npc_ys[i]+RADIUS > HEIGHT: npc_vys[i] = -abs(npc_vys[i]); npc_ys[i] = float(HEIGHT-RADIUS)
                if npc_types[i] == 2:
                    nf = npc_food[i]+0.1
                    if nf > 70.0: nf = 70.0
                    npc_food[i] = nf
                elif npc_types[i] == 0:
                    nm = npc_money[i]+0.1
                    if nm > 300.0: nm = 300.0
                    npc_money[i] = nm
                if npc_cooldowns[i] > 0: npc_cooldowns[i] -= 1; continue
                ddx = agent_x-npc_xs[i]; ddy = agent_y-npc_ys[i]; dist_sq = ddx*ddx+ddy*ddy
                if dist_sq < interaction_dist*interaction_dist:
                    mc, fc = get_trade_offer_fast(npc_types[i], eff, npc_money[i], npc_food[i])
                    if mc < 0.0 and agent_money < -mc: continue
                    if fc < 0.0 and agent_food < -fc: continue
                    agent_money += mc; agent_food += fc
                    npc_money[i] -= mc; npc_food[i] -= fc; total_trades += 1
                    if agent_money < 0.0: agent_money = 0.0
                    if agent_food < 0.0: agent_food = 0.0
                    if npc_money[i] < 0.0: npc_money[i] = 0.0
                    if npc_food[i] < 0.0: npc_food[i] = 0.0
                    npc_cooldowns[i] = cooldown_val

            agent_food -= eff[EFF_FOOD_DECAY]
            if agent_food < 0.0: agent_food = 0.0
            if agent_food <= 0.0: break

        eval_money += agent_money
        eval_food  += agent_food
        eval_trades += total_trades

    agent_money  = eval_money / N_EVAL_EPISODES
    agent_food   = eval_food  / N_EVAL_EPISODES
    total_trades = eval_trades / N_EVAL_EPISODES

    return (final_personality, agent_money, agent_food, total_trades,
            p_samples, r_samples, actual_episodes, run_start_money)


@njit(fastmath=True)
def _batch_kernel(eps_sched, lr_sched, seeds, out_p, out_scalars, out_psamp, out_rsamp):
    """Pokrece sve agente jednog paketa."""
    for i in range(seeds.shape[0]):
        fp, fm, ff, ft, ps, rs, ae, sm = train_one_run_jit(eps_sched, lr_sched, seeds[i])
        for p in range(NUM_PARAMS):
            out_p[i, p] = fp[p]
        out_scalars[i, 0] = fm
        out_scalars[i, 1] = ff
        out_scalars[i, 2] = ft
        out_scalars[i, 3] = ae
        out_scalars[i, 4] = sm
        for p in range(NUM_PARAMS):
            for k in range(ps.shape[1]):
                out_psamp[i, p, k] = ps[p, k]
        for k in range(rs.shape[0]):
            out_rsamp[i, k] = rs[k]


def run_batch():
    n = RUNS_PER_GOAL
    print(f"\n  {'=' * 60}")
    print(f"  EXPERIMENT B | {n} agents x {NUM_EPISODES} episodes")
    print(f"  {'=' * 60}")
    print("  Compiling JIT (first call only)...", end=" ", flush=True)
    t_c = time.time(); _ = train_one_run_jit(EPSILON_SCHEDULE, LR_SCHEDULE, 0)
    print(f"done ({time.time() - t_c:.1f}s)")

    NUM_SAMPLES = 200
    seeds = (np.arange(n, dtype=np.int64) * 12345 + 42)
    out_p = np.zeros((n, NUM_PARAMS))
    out_s = np.zeros((n, 5))
    out_ps = np.zeros((n, NUM_PARAMS, NUM_SAMPLES))
    out_rs = np.zeros((n, NUM_SAMPLES))

    chunk = max(1, min(n, 20))
    print("  Training:", flush=True)
    t0 = time.time()
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        _batch_kernel(EPSILON_SCHEDULE, LR_SCHEDULE, seeds[a:b],
                      out_p[a:b], out_s[a:b], out_ps[a:b], out_rs[a:b])
        done = b
        el = time.time() - t0
        per = el / done
        eta = per * (n - done)
        pct = done / n
        bar = '#' * int(40 * pct) + '.' * (40 - int(40 * pct))
        sys.stdout.write(f'\r  [{bar}] {done}/{n} ({pct*100:5.1f}%) '
                         f'elapsed: {el:>5.0f}s ETA: {eta:>5.0f}s ({per:.2f}s/agent)')
        sys.stdout.flush()
    dt = time.time() - t0
    sys.stdout.write('\r' + ' ' * 100 + '\r')
    print(f"  Completed in {dt:.1f}s ({dt / n:.3f}s per agent)")

    p_mean = out_ps.mean(axis=0); p_std = out_ps.std(axis=0)
    r_mean = out_rs.mean(axis=0); r_std = out_rs.std(axis=0)
    ep = out_s[:, 3]
    print(f"    Mean money: {out_s[:, 0].mean():.1f}  Mean food: {out_s[:, 1].mean():.1f}")
    print(f"    Early stopping: {int((ep < NUM_EPISODES).sum())}/{n} converged early")
    print(f"    Episodes used: avg {ep.mean():.0f} | min {ep.min():.0f} | max {ep.max():.0f}")

    results = []
    for i in range(n):
        results.append({
            "Consc": out_p[i, 0], "Agree": out_p[i, 1],
            "Extra": out_p[i, 2], "Neuro": out_p[i, 3],
            "Start_Money": out_s[i, 4],
            "Start_Wealth": out_s[i, 4] + FOOD_VALUE * AGENT_START_FOOD,
            "Final_Money": out_s[i, 0], "Final_Food": out_s[i, 1],
            "Total_Wealth": out_s[i, 0] + FOOD_VALUE * out_s[i, 1],
            "Trades": out_s[i, 2],
            "actual_episodes": int(out_s[i, 3]),
            "time": dt / n, "run_id": i + 1,
        })
    results[0]["_p_mean"] = p_mean; results[0]["_p_std"] = p_std
    results[0]["_r_mean"] = r_mean; results[0]["_r_std"] = r_std
    return results


def _get_history(results, key):
    """Vraca unapred agregirane nizove istorije treninga."""
    return results[0].get(key)


def smooth_history(mean_arr, std_arr, window):
    """Pokretni prosek nad nizovima srednje vrednosti i standardne devijacije."""
    m = np.asarray(mean_arr.astype(np.float32))
    s = np.asarray(std_arr.astype(np.float32))
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    sm = np.convolve(np.pad(m, pad, mode='edge'), kernel, mode='valid')[:len(m)]
    ss = np.convolve(np.pad(s, pad, mode='edge'), kernel, mode='valid')[:len(s)]
    return np.asarray(sm), np.asarray(ss)


def corr_matrix(df, cols):
    return df[cols].corr()


def histogram(values, bins, rmin, rmax):
    counts, _ = np.histogram(values, bins=bins, range=(rmin, rmax))
    return counts


SIGN_TARGETS = (
    ("BARGAINING (Barry & Friedman 1998)  <- MATCHES THE MECHANICS",
     {"Consc": 0, "Agree": -1, "Extra": -1, "Neuro": None}),
    ("WEALTH (Fenton-O'Creevy 2023)",
     {"Consc": +1, "Agree": -1, "Extra": -1, "Neuro": -1}),
    ("EARNINGS (Alderotti 2023)",
     {"Consc": +1, "Agree": -1, "Extra": +1, "Neuro": -1}),
)


NEGLIGIBLE_R = 0.10


def _sign_report(df, outcome, sig, use_partial=False):
    """Test predznaka u odnosu na tri literature."""
    sig = max(sig, NEGLIGIBLE_R)
    rs = {}
    for t in SHORT_NAMES:
        rs[t] = (partial_corr(df, t, outcome, "Start_Wealth") if use_partial
                 else df[t].corr(df[outcome]))
    txt = ""
    for label, exp in SIGN_TARGETS:
        hits = miss = null = 0
        lines = ""
        for t in SHORT_NAMES:
            r = rs[t]; e = exp[t]
            if e is None:            v = "no prediction"
            elif abs(r) < sig:
                v = "null"; null += 1
                if e == 0: v = "null  MATCH (predicted null)"; hits += 1; null -= 1
            elif e == 0:             v = "MISMATCH (predicted null)"; miss += 1
            elif np.sign(r) == e:    v = "match"; hits += 1
            else:                    v = "MISMATCH"; miss += 1
            sym = {1: '+', -1: '-', 0: '0', None: '?'}[e]
            lines += "   %-6s r=%+0.3f  exp %s  %s\n" % (t, r, sym, v)
        txt += label + "\n" + lines + "   %d match / %d mismatch / %d null\n\n" % (hits, miss, null)
    return txt, rs


def _corr_heatmap(ax, df, title, subtitle):
    """Korelaciona matrica: crte licnosti naspram ishoda."""
    cols = SHORT_NAMES + ["Start_Wealth", "Total_Wealth", "Gain"]
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if df[c].std() > 1e-12]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax.set_title(title + "\n" + subtitle, fontweight='bold', fontsize=11)


def plot_comprehensive_analysis(results, sweep_df=None):
    """Crta slike koje se koriste u radu."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if not k.startswith('_') and k not in ['time', 'run_id', 'actual_episodes']}
                       for r in results])
    df["Gain"] = df["Total_Wealth"] - df["Start_Wealth"]
    actual_eps = [r.get('actual_episodes', NUM_EPISODES) for r in results]
    avg_ep = np.mean(actual_eps)
    stopped_early = sum(1 for e in actual_eps if e < NUM_EPISODES)

    p_mean = _get_history(results, "_p_mean"); p_std = _get_history(results, "_p_std")
    r_mean = _get_history(results, "_r_mean"); r_std = _get_history(results, "_r_std")

    fig = plt.figure(figsize=(26, 17))
    gs = fig.add_gridspec(3, 7, hspace=0.42, wspace=0.9)
    fig.suptitle(
        "EXPERIMENT B — LEARNED PERSONALITY (contextual bandit)\n"
        f"{RUNS_PER_GOAL} agents x {NUM_EPISODES} max ep, avg converged @ {avg_ep:.0f} ep, "
        f"{stopped_early}/{RUNS_PER_GOAL} stopped early   |   bottom row: A vs B correlations",
        fontsize=15, fontweight='bold', y=0.985)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    final_values = {n: [r[n] for r in results] for n in SHORT_NAMES}
    x = np.arange(NUM_PARAMS)

    ax = fig.add_subplot(gs[0, 0:2])
    n_top = max(1, int(len(df) * 0.10))
    top = df.nlargest(n_top, 'Gain')
    tm = [top[n].mean() for n in SHORT_NAMES]; ts = [top[n].std() for n in SHORT_NAMES]
    pm = [df[n].mean() for n in SHORT_NAMES]
    ax.bar(x, tm, yerr=ts, color=colors, alpha=0.85, capsize=5, edgecolor='black',
           label='Elite (top 10% by GAIN)', zorder=3)
    ax.scatter(x, pm, marker='D', color='black', s=80, zorder=5, label='Population mean')
    ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES); ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean trait value'); ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_title('Elite vs Population\n(top 10% by wealth GAIN, not level)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

    ax = fig.add_subplot(gs[0, 2:4])
    parts = ax.violinplot([final_values[n] for n in SHORT_NAMES], positions=range(NUM_PARAMS),
                          showmeans=True, showmedians=True)
    for k, pc in enumerate(parts['bodies']): pc.set_facecolor(colors[k]); pc.set_alpha(0.7)
    ax.set_xticks(range(NUM_PARAMS)); ax.set_xticklabels(SHORT_NAMES)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Chosen trait value')
    n_flat = int((df['Agree'] <= 0.1).sum())
    ax.set_title('Chosen Trait Distribution\n'
                 f'(Agree <= 0.1 in {n_flat}/{len(df)} runs)', fontweight='bold')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5); ax.grid(axis='y', alpha=0.3)

    ax = fig.add_subplot(gs[0, 4:6])
    win = max(1, 200 // 20); xa = np.linspace(0, NUM_EPISODES, 200)
    for p, (n, c) in enumerate(zip(SHORT_NAMES, colors)):
        sm, ss = smooth_history(p_mean[p], p_std[p], win)
        ax.plot(xa, sm, label=n, color=c, lw=2)
        ax.fill_between(xa, sm - ss, sm + ss, color=c, alpha=0.15)
    ax.set_xlabel('Episode'); ax.set_ylabel('Greedy trait value'); ax.set_ylim(-0.05, 1.05)
    ax.set_title('Learning Convergence', fontweight='bold')
    ax.legend(fontsize=8); ax.axhline(0.5, color='gray', ls=':', alpha=0.5); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0:2])
    sr, srs = smooth_history(r_mean, r_std, win)
    ax.plot(xa, sr, color='#2c3e50', lw=2)
    ax.fill_between(xa, sr - srs, sr + srs, color='#2c3e50', alpha=0.2)
    z = np.polyfit(xa, sr, 1)
    ax.plot(xa, np.poly1d(z)(xa), 'r--', lw=2, label=f'trend {z[0]:.4f}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Episode reward')
    ax.set_title('Reward Over Training', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:4]); ax.axis('off')
    rows = []
    for n in SHORT_NAMES + ['Start_Wealth', 'Final_Money', 'Final_Food', 'Total_Wealth', 'Gain', 'Trades']:
        v = df[n]
        row = [n, f"{v.mean():.2f}", f"{v.std():.2f}", f"{v.min():.2f}", f"{v.max():.2f}"]
        if sweep_df is not None and n in sweep_df.columns:
            row.append(f"{sweep_df[n].mean():.2f}")
        else:
            row.append("-")
        rows.append(row)
    cols = ['Metric', 'B mean', 'B std', 'B min', 'B max', 'A mean']
    tb = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1.1, 1.35)
    for k in range(len(cols)):
        tb[(0, k)].set_facecolor('#3498db'); tb[(0, k)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Summary Statistics  (B = learner, A = sweep)', fontweight='bold')

    ax = fig.add_subplot(gs[1, 4:6])
    sc = ax.scatter(df['Start_Wealth'], df['Gain'], c=df['Extra'], cmap='viridis',
                    s=28, alpha=0.75, edgecolors='black', linewidth=0.2)
    plt.colorbar(sc, ax=ax, label='Extraversion')
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('Starting wealth'); ax.set_ylabel('Wealth gain')
    rg = df['Start_Wealth'].corr(df['Gain'])
    _verdict = ('no relationship' if abs(rg) < NEGLIGIBLE_R
                else ('poorer agents gain more' if rg < 0 else 'richer agents gain more'))
    ax.set_title(f'Gain vs Starting Wealth\n(r = {rg:+.3f} — {_verdict})', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 0:3])
    if sweep_df is not None:
        sd = sweep_df.copy(); sd["Gain"] = sd["Total_Wealth"] - sd["Start_Wealth"]
        _corr_heatmap(ax, sd, 'EXPERIMENT A — NO LEARNING',
                      f'n={len(sd)} random personalities, everyone starts at the '
                      f'median ({AGENT_START_MONEY:.0f})')
    else:
        ax.axis('off'); ax.text(0.5, 0.5, 'Experiment A not run', ha='center')

    ax = fig.add_subplot(gs[2, 4:7])
    _corr_heatmap(ax, df, 'EXPERIMENT B — AFTER LEARNING',
                  f'n={len(df)} agents that trained {NUM_EPISODES} episodes and '
                  f'CHOSE their own personality')

    plt.savefig("validity_analysis_money.png", dpi=140, bbox_inches='tight')
    try:
        from IPython.display import display; display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)
    print("  Saved: validity_analysis_money.png")


@njit(fastmath=True)
def eval_personality_jit(personality, seed, fixed_start):
    """Ocenjuje fiksnu licnost kroz N_EVAL_EPISODES epizoda."""
    np.random.seed(seed)
    npc_xs = np.zeros(MAX_NPCS); npc_ys = np.zeros(MAX_NPCS)
    npc_vxs = np.zeros(MAX_NPCS); npc_vys = np.zeros(MAX_NPCS)
    npc_types = np.zeros(MAX_NPCS, dtype=np.int32)
    npc_money = np.zeros(MAX_NPCS); npc_food = np.zeros(MAX_NPCS)
    npc_cooldowns = np.zeros(MAX_NPCS, dtype=np.int32)
    eff = compute_effects_fast(personality)
    run_start_money = AGENT_START_MONEY if fixed_start else draw_start_money()
    eval_money = 0.0; eval_food = 0.0; eval_trades = 0.0
    for _ev in range(N_EVAL_EPISODES):
        agent_x = WIDTH/2.0+(np.random.random()-0.5)*100.0
        agent_y = HEIGHT/2.0+(np.random.random()-0.5)*100.0
        agent_vx = (np.random.random()-0.5)*4.0; agent_vy = (np.random.random()-0.5)*4.0
        agent_money = run_start_money; agent_food = AGENT_START_FOOD
        agent_radius = float(RADIUS+2); total_trades = 0
        num_m = np.random.randint(1, MAX_NUM_MERCHANTS+1)
        num_f = np.random.randint(1, MAX_NUM_FARMERS+1)
        num_b = int(TOTAL_NUM_NPC*np.random.randint(1, 6)*0.02)
        idx = 0
        for _ in range(num_m):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 0; npc_money[idx] = 250.0; npc_food[idx] = 7.0; npc_cooldowns[idx] = 0; idx += 1
        for _ in range(num_b):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 1; npc_money[idx] = 25.0; npc_food[idx] = 14.0; npc_cooldowns[idx] = 0; idx += 1
        for _ in range(num_f):
            if idx >= MAX_NPCS: break
            npc_xs[idx] = np.random.randint(80, WIDTH-80); npc_ys[idx] = np.random.randint(80, HEIGHT-80)
            npc_vxs[idx] = (np.random.random()-0.5)*3.0; npc_vys[idx] = (np.random.random()-0.5)*3.0
            npc_types[idx] = 2; npc_money[idx] = 25.0; npc_food[idx] = 56.0; npc_cooldowns[idx] = 0; idx += 1
        actual_npcs = idx

        cooldown_val = int(eff[EFF_COOLDOWN])
        if cooldown_val < 5: cooldown_val = 5
        interaction_dist = (agent_radius+RADIUS)*eff[EFF_TRADE_RADIUS]

        for step in range(STEPS_PER_EPISODE):
            speed_sq = agent_vx*agent_vx+agent_vy*agent_vy
            if speed_sq > 0.0001:
                speed = math.sqrt(speed_sq)
                scale = eff[EFF_SPEED] / speed
                agent_vx *= scale; agent_vy *= scale
            for i in range(actual_npcs):
                dx = npc_xs[i]-agent_x; dy = npc_ys[i]-agent_y; dist_sq = dx*dx+dy*dy
                if dist_sq > 0.0 and dist_sq < 40000.0:
                    dist = math.sqrt(dist_sq); nx = dx/dist; ny = dy/dist
                    if npc_types[i] == 1:
                        agent_vx -= nx*eff[EFF_NPC_FORCE]; agent_vy -= ny*eff[EFF_NPC_FORCE]
                    else:
                        agent_vx += nx*eff[EFF_NPC_FORCE]; agent_vy += ny*eff[EFF_NPC_FORCE]
            agent_x, agent_y, agent_vx, agent_vy = update_position_fast(
                agent_x, agent_y, agent_vx, agent_vy, agent_radius)

            for i in range(actual_npcs):
                npc_xs[i] += npc_vxs[i]; npc_ys[i] += npc_vys[i]
                if npc_xs[i]-RADIUS < 0.0: npc_vxs[i] = abs(npc_vxs[i]); npc_xs[i] = float(RADIUS)
                elif npc_xs[i]+RADIUS > WIDTH: npc_vxs[i] = -abs(npc_vxs[i]); npc_xs[i] = float(WIDTH-RADIUS)
                if npc_ys[i]-RADIUS < 0.0: npc_vys[i] = abs(npc_vys[i]); npc_ys[i] = float(RADIUS)
                elif npc_ys[i]+RADIUS > HEIGHT: npc_vys[i] = -abs(npc_vys[i]); npc_ys[i] = float(HEIGHT-RADIUS)
                if npc_types[i] == 2:
                    nf = npc_food[i]+0.1
                    if nf > 70.0: nf = 70.0
                    npc_food[i] = nf
                elif npc_types[i] == 0:
                    nm = npc_money[i]+0.1
                    if nm > 300.0: nm = 300.0
                    npc_money[i] = nm
                if npc_cooldowns[i] > 0: npc_cooldowns[i] -= 1; continue
                ddx = agent_x-npc_xs[i]; ddy = agent_y-npc_ys[i]; dist_sq = ddx*ddx+ddy*ddy
                if dist_sq < interaction_dist*interaction_dist:
                    mc, fc = get_trade_offer_fast(npc_types[i], eff, npc_money[i], npc_food[i])
                    if mc < 0.0 and agent_money < -mc: continue
                    if fc < 0.0 and agent_food < -fc: continue
                    agent_money += mc; agent_food += fc
                    npc_money[i] -= mc; npc_food[i] -= fc; total_trades += 1
                    if agent_money < 0.0: agent_money = 0.0
                    if agent_food < 0.0: agent_food = 0.0
                    if npc_money[i] < 0.0: npc_money[i] = 0.0
                    if npc_food[i] < 0.0: npc_food[i] = 0.0
                    npc_cooldowns[i] = cooldown_val

            agent_food -= eff[EFF_FOOD_DECAY]
            if agent_food < 0.0: agent_food = 0.0
            if agent_food <= 0.0: break

        eval_money += agent_money
        eval_food  += agent_food
        eval_trades += total_trades

    return (eval_money / N_EVAL_EPISODES,
            eval_food / N_EVAL_EPISODES,
            eval_trades / N_EVAL_EPISODES,
            run_start_money)


@njit(fastmath=True)
def _sweep_kernel(traits, out, fixed_start):
    for i in range(traits.shape[0]):
        m, f, t, sm = eval_personality_jit(traits[i], i * 7919 + 13, fixed_start)
        out[i, 0] = m; out[i, 1] = f; out[i, 2] = t; out[i, 3] = sm


def compare_start_regimes(df_fixed, df_gini):
    """Uporedjuje rezim fiksnog pocetka i rezim raspodele pocetnog bogatstva."""
    for d in (df_fixed, df_gini):
        d["Gain"] = d["Total_Wealth"] - d["Start_Wealth"]
    print("\n  START-REGIME COMPARISON  (Experiment A, uniform traits)")
    print("  " + "=" * 72)
    print("  %-6s | %-21s | %-21s" % ("trait", "FIXED start (median)", "GINI start (lognormal)"))
    print("  %-6s | %10s %10s | %10s %10s" % ("", "r(Wealth)", "r(Gain)", "r(Wealth)", "r(Gain)"))
    for t in SHORT_NAMES:
        print("  %-6s | %+10.3f %+10.3f | %+10.3f %+10.3f" % (
            t, df_fixed[t].corr(df_fixed["Total_Wealth"]), df_fixed[t].corr(df_fixed["Gain"]),
            df_gini[t].corr(df_gini["Total_Wealth"]), df_gini[t].corr(df_gini["Gain"])))
    print("  r(Start,Wealth): fixed %.3f | gini %.3f" % (
        df_fixed["Start_Wealth"].corr(df_fixed["Total_Wealth"]),
        df_gini["Start_Wealth"].corr(df_gini["Total_Wealth"])))
    print("  Under a FIXED start there is no start variance, so the LEVEL and the")
    print("  GAIN carry the same information. Under the GINI start the level is")
    print("  swamped by the initial endowment and only the GAIN is informative.")


def partial_corr(df, x, y, z):
    """Parcijalna korelacija r(x, y | z)."""
    rxy = df[x].corr(df[y]); rxz = df[x].corr(df[z]); rzy = df[z].corr(df[y])
    d = math.sqrt(max(1e-12, (1 - rxz**2) * (1 - rzy**2)))
    return (rxy - rxz * rzy) / d


def run_trait_sweep(n_samples=12000, seed=0, fixed_start=False):
    """Eksperiment A: nasumicne crte licnosti, bez ucenja."""
    rng = np.random.default_rng(seed)
    traits = rng.random((n_samples, NUM_PARAMS))
    out = np.zeros((n_samples, 4))
    t0 = time.time(); _sweep_kernel(traits, out, fixed_start); dt = time.time() - t0

    df = pd.DataFrame(traits, columns=SHORT_NAMES)
    df["Final_Money"] = out[:, 0]; df["Final_Food"] = out[:, 1]
    df["Trades"] = out[:, 2]
    df["Start_Money"] = out[:, 3]
    df["Start_Wealth"] = df["Start_Money"] + FOOD_VALUE * AGENT_START_FOOD
    df["Total_Wealth"] = df["Final_Money"] + FOOD_VALUE * df["Final_Food"]

    sig = 1.96 / math.sqrt(n_samples)
    mode = "FIXED start (median only)" if fixed_start else "GINI start (lognormal)"
    print(f"\n  EXPERIMENT A — EXOGENOUS SWEEP [{mode}]: {n_samples} draws in {dt:.1f}s")
    print(f"  {'=' * 66}")
    print(f"  start wealth {START_WEALTH:.1f} | mean wealth {df['Total_Wealth'].mean():.1f} "
          f"| trades {df['Trades'].mean():.0f} | |r| > {sig:.3f} to count")
    for label, outcome, exp in (
            ("EARNINGS (Alderotti 2023)", "Final_Money",
             {"Consc": +1, "Agree": -1, "Extra": +1, "Neuro": -1}),
            ("WEALTH (Fenton-O'Creevy 2023)", "Total_Wealth",
             {"Consc": +1, "Agree": -1, "Extra": -1, "Neuro": -1})):
        hits = miss = null = 0
        print(f"  --- vs {label}, outcome {outcome} ---")
        for name in SHORT_NAMES:
            r  = df[name].corr(df[outcome])
            rp = partial_corr(df, name, outcome, "Start_Wealth")
            e = '+' if exp[name] > 0 else '-'
            if abs(r) < sig: v = "null (n.s.)"; null += 1
            elif np.sign(r) == exp[name]: v = "match"; hits += 1
            else: v = "MISMATCH"; miss += 1
            print("    %-6s raw r=%+0.3f  partial r=%+0.3f  exp %s  %s" % (name, r, rp, e, v))
        print(f"    {hits} match / {miss} mismatch / {null} null")
    fn = "sweep_results_fixed.csv" if fixed_start else "sweep_results_gini.csv"
    df.to_csv(fn, index=False)
    print("  Saved: " + fn)
    return df


def main():
    print(f"\n  {'=' * 60}")
    print(f"  RL PERSONALITY TRAINING — frozen-trait contextual bandit")
    print(f"  States: {NUM_STATES} | Freeze personality: {FREEZE_PERSONALITY} | Eval eps: {N_EVAL_EPISODES}")
    print(f"  Objective: NET WORTH = money + {FOOD_VALUE} * food  (start {START_WEALTH:.2f})")
    print(f"  Episode: 14 days = {STEPS_PER_EPISODE} steps, food decay {AGENT_FOOD_DECAY}/step")
    print(f"  Start (eval): money {AGENT_START_MONEY:.1f}, food {AGENT_START_FOOD:.1f}")
    print(f"  Start range:  money {START_MONEY_MIN:.1f}-{START_MONEY_MAX:.1f}, "
          f"food {START_FOOD_MIN:.1f}-{START_FOOD_MAX:.1f} (+/-{START_SPREAD:.0%})")
    print(f"  Prices: wholesale {FARMER_PRICE} -> retail {MERCHANT_PRICE} (both ANCHORED)")
    print(f"  States: {NUM_STATES} | eval bins money {_eval_money_bin} food {_eval_food_bin}")
    print(f"  {'=' * 60}")
    t0 = time.time()

    sweep_fixed = run_trait_sweep(12000, fixed_start=True)
    sweep_gini  = run_trait_sweep(12000, fixed_start=False)
    compare_start_regimes(sweep_fixed, sweep_gini)
    sweep_df = sweep_fixed

    results = run_batch()
    plot_comprehensive_analysis(results, sweep_df)

    total_time = time.time() - t0
    print(f"\n  {'=' * 60}")
    print(f"  TOTAL TIME: {total_time/60:.1f} min")
    print(f"  {'=' * 60}")
    print(f"\n  Generated files:")
    print(f"    - validity_analysis_money.png")
    print(f"\n  Done.\n")

if __name__ == "__main__":
    main()
