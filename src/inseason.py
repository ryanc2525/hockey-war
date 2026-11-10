"""
In-season learning, level 1 (notebook 28): update each player's projection with this season's real games.

The idea (Ryan's), in plain language:
  - Before the season, the simulator's random draw z for a player is a guess at how his season turns out:
    WAR/60 = projection + z x (his draw size), and his scoring rates x exp(k x z). Draw 4,000 guesses from his
    notebook 17 pool: that's what we believed before any games (the prior).
  - Real games are evidence. For each guess, ask "how likely are his real goals, assists (and WAR) so far if this
    guess were true?" That number is the guess's weight (the likelihood). Guesses that fit get more weight.
  - The weighted guesses are the updated belief (the posterior). The rest of the season is simulated from them.
  This is Bayesian updating by importance weighting (a particle filter), one player at a time: each player's draw
  is independent, so there's no need for one simulated league that matches every real player at once.

Two versions of a guess (notebook 28 backtests both and adopts the better one):
  - "whole": z is a whole-season outcome (real talent change + that season's luck), and the rest of the season
    uses the same z. WAR evidence noise: b x (1/minutes so far - 1/full-season minutes), since the guess already
    contains a full season's luck.
  - "split": z is split with notebook 17's luck fit (variance of a miss = a + b / minutes). The talent part is
    t = z x sqrt(a) (the share of the draw that isn't luck, with the pool's shape). Evidence updates t only, and
    the rest of the season gets fresh luck, a new draw with variance b / remaining minutes. WAR evidence noise:
    b / minutes so far. The result goes back to the simulator as one number, z_rest = (t + fresh luck) / draw size.

    import inseason as IS
    P = IS.prior(pool_ids, pools, n=4000, seed=1)                        # guesses, players x n
    w = IS.weights(IS.log_lik_counts(...) + IS.log_lik_war(...))        # each guess's weight
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln

VERSIONS = ["whole", "split"]
N_PARTICLES = 4000


# ---------------------------------------------------------------------------
# 1. The prior: guesses from each player's notebook 17 pool
# ---------------------------------------------------------------------------
def prior(pool_ids, pools, n=N_PARTICLES, seed=1):
    """n guesses z for every player (players x n), each from his own draw pool (pools: list of (z, chances))."""
    rng = np.random.default_rng(seed)
    pool_ids = np.asarray(pool_ids)
    Z = np.zeros((len(pool_ids), n))
    for k in sorted(set(pool_ids)):
        rows = np.flatnonzero(pool_ids == k)
        z, p = pools[k]
        Z[rows] = rng.choice(z, size=(len(rows), n), p=p)
    return Z


def draw_size(fit, group, minutes):
    """Notebook 17: the size (SD) of a player's WAR/60 miss over `minutes`, sqrt(a + b / minutes)."""
    g = pd.Series(group)
    a, b = g.map(fit["a"]).to_numpy(), g.map(fit["b"]).to_numpy()
    return np.sqrt(a + b / np.maximum(np.asarray(minutes, float), 1.0))


def evidence_z(Z, version, fit, group, season_minutes):
    """The part of each guess the evidence speaks to, in z units (players x n): the whole z, or the talent part
    t / draw size = z x sqrt(a) / sqrt(a + b / M) (M = full-season minutes)."""
    if version == "whole":
        return Z
    g = pd.Series(group)
    a = g.map(fit["a"]).to_numpy()
    return Z * (np.sqrt(a) / draw_size(fit, group, season_minutes))[:, None]


# ---------------------------------------------------------------------------
# 2. The evidence: how likely is what really happened, guess by guess
# ---------------------------------------------------------------------------
def scoring_factor(Zs, k):
    """exp(k x z) divided by its average over the player's prior guesses, so before any evidence the average
    factor is 1 and his projected rate stays the center."""
    e = np.exp(k * Zs)
    return e / e.mean(axis=1, keepdims=True)


def log_lik_counts(count, rate60, minutes, factor):
    """
    Poisson log-likelihood of a real count (goals or assists so far) for every guess: given a guess, his count
    should be a Poisson number with mean rate/60 x factor x minutes / 60. A guess that makes the real count very
    unlikely gets a very low value. count, rate60, minutes: one per player; factor: players x n.
    """
    mu = np.maximum(np.asarray(rate60, float)[:, None] * factor * np.asarray(minutes, float)[:, None] / 60, 1e-9)
    c = np.asarray(count, float)[:, None]
    return c * np.log(mu) - mu - gammaln(c + 1)


def war_noise_var(version, fit, group, minutes_so_far, season_minutes):
    """How noisy a season-to-date WAR/60 is (variance, per player). whole: b x (1/m - 1/M); split: b / m."""
    b = pd.Series(group).map(fit["b"]).to_numpy()
    m = np.maximum(np.asarray(minutes_so_far, float), 1.0)
    if version == "whole":
        return b * np.maximum(1 / m - 1 / np.maximum(np.asarray(season_minutes, float), m), 1e-12)
    return b / m


def log_lik_war(obs_war60, proj_war60, level_z, size, noise_var):
    """
    Normal log-likelihood of a real season-to-date WAR/60: real = the guessed level + luck noise.
    level_z: the guess in z units (players x n); level = projection + level_z x size. Players without an
    observation (NaN) get 0, i.e. no information.
    """
    obs = np.asarray(obs_war60, float)[:, None]
    level = np.asarray(proj_war60, float)[:, None] + level_z * np.asarray(size, float)[:, None]
    v = np.asarray(noise_var, float)[:, None]
    ll = -0.5 * (obs - level) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    return np.where(np.isnan(obs), 0.0, ll)


def weights(log_lik):
    """Normalized weights from summed log-likelihoods (players x n), and each player's effective sample size:
    1 / sum(w^2), from n (no information) down to 1 (one guess carries everything)."""
    ll = log_lik - log_lik.max(axis=1, keepdims=True)
    w = np.exp(ll)
    w /= w.sum(axis=1, keepdims=True)
    return w, 1 / (w ** 2).sum(axis=1)


# ---------------------------------------------------------------------------
# 3. The rest of the season, from the weighted guesses
# ---------------------------------------------------------------------------
def rest_z(Z, w, version, fit, group, season_minutes, rest_minutes, n, seed=2):
    """
    n draws of each player's z for the rest of the season (players x n): resample guesses by weight; with
    "split", the talent part plus fresh luck for the remaining minutes (variance b / rest minutes), turned back
    into z units with his full-season draw size, so the simulator can use it exactly like a pool draw.
    """
    rng = np.random.default_rng(seed)
    P, N = Z.shape
    picked = np.vstack([Z[i, rng.choice(N, size=n, p=w[i])] for i in range(P)])     # resample by weight
    if version == "whole":
        return picked
    g = pd.Series(group)
    a, b = g.map(fit["a"]).to_numpy(), g.map(fit["b"]).to_numpy()
    size = draw_size(fit, group, season_minutes)
    talent = picked * np.sqrt(a)[:, None]
    luck = rng.normal(size=(P, n)) * np.sqrt(b / np.maximum(np.asarray(rest_minutes, float), 1.0))[:, None]
    return (talent + luck) / size[:, None]


def posterior_mean(values, w):
    """The weighted average of anything per guess (players x n)."""
    return (values * w).sum(axis=1)


# ---------------------------------------------------------------------------
# 4. Teams: what's left of real results after the player update
# ---------------------------------------------------------------------------
def team_weight(games, g0):
    """How much a team's leftover results so far count: w = g / (g + g0). g0 = (luck variance of a team's points
    per game) / (variance of real team effects): the number of games at which evidence and prior count equally."""
    g = np.asarray(games, float)
    return g / (g + g0)


def updated_bonus(bonus0, gap_ppg, games, g0, beta):
    """New team bonus (normal goals per game) = preseason bonus + w x (points-per-game gap so far) / beta."""
    return np.asarray(bonus0, float) + team_weight(games, g0) * np.asarray(gap_ppg, float) / beta
