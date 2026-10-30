"""Per-PSF fits started from the CNN, with a convergence stop and slot refill.

A fixed number of slots (one per PSF on the GPU) runs in parallel; a PSF stops when its parameters move
less than a tolerance for `patience` accepted steps in a row (or after `max_rejects` rejected steps in a row, or
`cap` steps), its slot then takes the next PSF of the pool.
Two optimizers fitting position, photons, orientation and the background of the 6 channels (3 planes x 2
polarisations) together: Adam (one step count per slot) or Levenberg-Marquardt.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _bcast(mask, a):
    return mask.reshape((-1,) + (1,) * (a.ndim - 1))


def make_refill(init):
    """Jitted refill: slots where mask is True restart from init(ids, pool)."""
    return jax.jit(lambda s, i, m, p: jax.tree_util.tree_map(lambda a, b: jnp.where(_bcast(m, a), b, a), s, init(i, p)))


def stream(n, slots, init, refill, step, watch, move, extract, pool, cap, patience, accepted=None, max_rejects=None, hold=0):
    """Runs `step` on `slots` PSF at a time until each of the n PSF of the pool has converged.

    init(ids, pool) -> state, step(state, ids, pool) -> state: pytrees with a leading slot axis, refill = make_refill(init).
    watch(state) -> (slots, k) numpy, move(prev, cur) -> (slots,) movement / tolerance.
    extract(state) -> dict of (slots, ...) numpy. Returns dict of (n, ...) arrays and the steps of each PSF.
    accepted(state) -> (slots,) numpy bool, was the last step accepted (None: always). A rejected step does not move
    the parameters: it neither counts as calm nor resets the calm count; max_rejects rejected steps in a row stop the
    PSF (converged: no smaller step lowers the loss). No stop during the first `hold` steps.
    """
    ids = np.minimum(np.arange(slots), n - 1)
    alive = np.arange(slots) < n
    nxt = slots
    state = init(jnp.asarray(ids), pool)
    calm, age, rej = np.zeros(slots, int), np.zeros(slots, int), np.zeros(slots, int)
    steps, out = np.zeros(n, int), None
    prev = watch(state)
    while alive.any():
        state = step(state, jnp.asarray(ids), pool)
        age += 1
        cur = watch(state)
        acc = np.ones(slots, bool) if accepted is None else accepted(state)
        calm = np.where(age <= hold, 0, np.where(acc, np.where(move(prev, cur) < 1, calm + 1, 0), calm))
        rej = np.where(acc, 0, rej + 1)
        stop = (calm >= patience) if max_rejects is None else (calm >= patience) | (rej >= max_rejects)
        done = alive & ((stop & (age > hold)) | (age >= cap))  # calm counted from the end of the hold
        if done.any():
            vals = extract(state)
            if out is None:
                out = {k: np.zeros((n,) + v.shape[1:], v.dtype) for k, v in vals.items()}
            for k, v in vals.items():
                out[k][ids[done]] = v[done]
            steps[ids[done]] = age[done]
            new = np.zeros(slots, bool)
            for s in np.where(done)[0]:
                if nxt < n:
                    ids[s], nxt, new[s] = nxt, nxt + 1, True
                else:
                    alive[s] = False
            if new.any():
                state = refill(state, jnp.asarray(ids), jnp.asarray(new), pool)
                calm[new], age[new], rej[new] = 0, 0, 0
                cur = watch(state)
        prev = cur
    return out, steps


def adam_slots(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam with one step count per slot (t, (slots,)), same update as optax.adam for each slot."""
    t1 = t + 1

    def upd(p, g, m_, v_):
        m_ = b1 * m_ + (1 - b1) * g
        v_ = b2 * v_ + (1 - b2) * g ** 2
        c1 = _bcast(t1, p).astype(p.dtype)
        mh, vh = m_ / (1 - b1 ** c1), v_ / (1 - b2 ** c1)
        return p - lr * mh / (jnp.sqrt(vh) + eps), m_, v_

    res = {k: upd(params[k], grads[k], m[k], v[k]) for k in params}
    return {k: r[0] for k, r in res.items()}, {k: r[1] for k, r in res.items()}, {k: r[2] for k, r in res.items()}, t1


def _axis(rho, eta):
    r, e = np.radians(rho), np.radians(eta)
    return np.stack([np.sin(e) * np.cos(r), np.sin(e) * np.sin(r), np.cos(e)], -1)


def move_orient(prev, cur, tol_deg=0.2, tol_nm=2.):
    """prev, cur: (slots, 6) x, y, z (um), rho, eta, delta (deg). Largest of axis angle, delta, position steps."""
    a = np.degrees(np.arccos(np.clip(np.abs(np.sum(_axis(cur[:, 3], cur[:, 4]) * _axis(prev[:, 3], prev[:, 4]), -1)), 0, 1)))
    return np.maximum.reduce([a / tol_deg, np.abs(cur[:, 5] - prev[:, 5]) / tol_deg,
                              np.abs(cur[:, :3] - prev[:, :3]).max(1) * 1e3 / tol_nm])


def move_orient_bg(prev, cur, tol_bg=0.01):
    """As move_orient, with more columns (background offsets, photons/pixel) and their tolerance."""
    return np.maximum(move_orient(prev[:, :6], cur[:, :6]), np.abs(cur[:, 6:] - prev[:, 6:]).max(1) / tol_bg)


def make_fit_adam(loss2, sp, lr, bg_speed, slots, cap=400, patience=10):
    """Builds fit(pool), compiled once: one Adam fit of orientation, N, x, y, z and the background of the 6 channels
    (offsets added to the starting background of each channel), with the loss of SGD2.

    pool: dict of device arrays: data, noise, background (n, 6) start, x0, y0, z0, N0, rho0, eta0, delta0.
    loss2(params, data, background_flat, noise): summed loss of SGD2. sp: speeds of SGD2 (delta_speed,
    nphotons_speed2, xy_speed2, z_speed2). bg_speed: speed of the background offsets (photons/pixel).
    fit(pool) returns dict of (n,) arrays (x, y, z, N, rho, eta, delta, background (n, 6), steps).
    """
    zeros = lambda p: jax.tree_util.tree_map(jnp.zeros_like, p)
    bg_of = lambda prm, bg0: (bg0 + prm['bgo'] * bg_speed).reshape(-1)

    def init(ids, p):
        prm = {'rho': p['rho0'][ids], 'eta': p['eta0'][ids], 'delta': p['delta0'][ids] / sp['delta_speed'],
               'N_photons': p['N0'][ids] / sp['nphotons_speed2'], 'x': p['x0'][ids] / sp['xy_speed2'],
               'y': p['y0'][ids] / sp['xy_speed2'], 'z': p['z0'][ids] / sp['z_speed2'], 'bgo': jnp.zeros((ids.shape[0], 6))}
        return prm, zeros(prm), zeros(prm), jnp.zeros(ids.shape[0], jnp.int32)

    @jax.jit
    def step(state, ids, p):
        prm, m, v, t = state
        bg0 = p['background'][ids]
        lf = lambda q: loss2(dict({k: q[k] for k in q if k != 'bgo'}, zern_x=jnp.zeros(45), zern_y=jnp.zeros(45)),
                             p['data'][ids], bg_of(q, bg0), p['noise'][ids])
        prm, m, v, t = adam_slots(prm, jax.grad(lf)(prm), m, v, t, lr)
        prm['delta'] = jnp.maximum(prm['delta'], 2. / sp['delta_speed'])  # wobble >= 2 deg (0/0 in PSF_jax near 0)
        return prm, m, v, t

    wj = jax.jit(lambda s: jnp.concatenate([jnp.stack([s[0]['x'] * sp['xy_speed2'], s[0]['y'] * sp['xy_speed2'], s[0]['z'] * sp['z_speed2'],
                                                       s[0]['rho'], s[0]['eta'], s[0]['delta'] * sp['delta_speed']], 1),
                                            s[0]['bgo'] * bg_speed], 1))
    w = lambda s: np.asarray(wj(s))  # one transfer per step

    def ex(s):
        q = s[0]
        return {'x': np.array(q['x']) * float(sp['xy_speed2']), 'y': np.array(q['y']) * float(sp['xy_speed2']),
                'z': np.array(q['z']) * float(sp['z_speed2']), 'rho': np.array(q['rho']), 'eta': np.array(q['eta']),
                'delta': np.array(q['delta']) * float(sp['delta_speed']), 'N': np.array(q['N_photons']) * float(sp['nphotons_speed2']),
                'bgo': np.array(q['bgo']) * bg_speed}
    refill = make_refill(init)

    def fit(pool):
        n = pool['data'].shape[0]
        r, steps = stream(n, slots, init, refill, step, w, move_orient_bg, ex, pool, cap, patience)
        bg = np.array(pool['background']) + r.pop('bgo')
        return dict(r, background=bg, steps=steps)
    return fit


def make_fit_lm(model, pixel_loss, pixel_weight, slots, cap=200, patience=2, max_rejects=8, lam0=1., delta_step=20., delta_hold=5,
                lam_down=3., tol_deg=1., tol_nm=10.):
    """Builds fit(pool), compiled once: Levenberg-Marquardt fit of x, y, z, N, rho, eta, delta and the background of
    the 6 channels (offsets added to the starting background of each channel), with the loss of the pixels given.
    lam0: starting damping (Marquardt, relative to the diagonal of J^T W J; / lam_down after an accepted step, x 4 after a
    rejected one). delta_step: largest change of delta in one step (deg), the whole step is shortened to it.
    delta_hold: delta fixed at its start for the first steps (the axis turns first; a free delta from a start far
    from the axis ran to the 179 deg clip, almost isotropic, where rho and eta have no derivative left).
    Stop: patience accepted steps in a row moving less than the tolerance (tol_deg on rho, eta, delta, tol_nm on x, y, z),
    or max_rejects rejected steps in a row.

    pool: dict of device arrays: data, noise, background (n, 6) start, x0, y0, z0, N0, rho0, eta0, delta0, and the
    parameters shared by the pool that model reads (e.g. the aberrations: they can change between two calls of fit
    without a new compilation).
    model(th7, background_flat, pool) -> expected image (slots, 3, 2, 13, 13), th7 = x, y, z (um), N, rho, eta, delta (deg).
    pixel_loss(mu, data, noise): loss of each pixel. pixel_weight(mu, data, noise): its curvature vs mu (> 0), the
    weight of each pixel in the Gauss-Newton matrix J^T W J.
    fit(pool) returns dict of (n,) arrays (x, y, z, N, rho, eta, delta, background (n, 6), steps = iterations).
    """

    def full(th, bg0):
        return th[:, :7], (bg0 + th[:, 7:]).reshape(-1)

    def nll(th, bg0, data, noise, p):
        t7, bg = full(th, bg0)
        return jnp.sum(pixel_loss(model(t7, bg, p), data, noise), axis=(1, 2, 3, 4))

    def init(ids, p):
        th = jnp.stack([p[k][ids] for k in ('x0', 'y0', 'z0', 'N0', 'rho0', 'eta0', 'delta0')], 1)
        n = ids.shape[0]
        return jnp.concatenate([th, jnp.zeros((n, 6))], 1), jnp.full(n, lam0), jnp.zeros(n, jnp.int32), jnp.ones(n, bool)

    @jax.jit
    def step(state, ids, p):
        th, lam, k, _ = state
        data, noise, bg0 = p['data'][ids], p['noise'][ids], p['background'][ids]
        f = lambda t7: model(t7, full(th, bg0)[1], p)
        mu = f(th[:, :7])
        jp = jnp.stack([jax.jvp(f, (th[:, :7],), (jnp.zeros_like(th[:, :7]).at[:, k].set(1.),))[1] for k in range(7)], 1)
        jb = jnp.broadcast_to(jnp.eye(6).reshape(1, 6, 3, 2, 1, 1), (th.shape[0], 6, 3, 2) + mu.shape[-2:])  # background of each channel
        free = (k >= delta_hold).astype(mu.dtype)  # 0: delta held
        J = jnp.concatenate([jp, jb], 1).at[:, 6].multiply(free[:, None, None, None, None])
        dl = jax.grad(lambda m: jnp.sum(pixel_loss(m, data, noise)))(mu)  # derivative of the loss of each pixel vs mu
        grad = jnp.sum(J * dl[:, None], axis=(2, 3, 4, 5))
        F = jnp.einsum('nkabcd,nlabcd->nkl', J * pixel_weight(mu, data, noise)[:, None], J)  # Gauss-Newton matrix of each PSF
        A = F + lam[:, None, None] * jnp.eye(13)[None] * jnp.diagonal(F, axis1=1, axis2=2)[:, :, None]
        A = A.at[:, 6, 6].add(1. - free)  # held delta: row and column 0, 1 on the diagonal -> no step
        dth = jnp.linalg.solve(A, -grad[..., None])[..., 0]
        dth = dth * jnp.minimum(1., delta_step / jnp.maximum(jnp.abs(dth[:, 6]), 1e-12))[:, None]
        cand = th + dth
        cand = cand.at[:, 6].set(jnp.clip(cand[:, 6], 2., 179.)).at[:, 3].set(jnp.maximum(cand[:, 3], 1.)).at[:, 2].set(jnp.maximum(cand[:, 2], 0.))
        ok = nll(cand, bg0, data, noise, p) < nll(th, bg0, data, noise, p)  # False for NaN
        return jnp.where(ok[:, None], cand, th), jnp.where(ok, lam / lam_down, lam * 4.), k + 1, ok

    w = lambda s: np.asarray(s[0])[:, [0, 1, 2, 4, 5, 6]]
    ex = lambda s: {'th': np.array(s[0])}
    refill = make_refill(init)

    def fit(pool):
        n = pool['data'].shape[0]
        r, steps = stream(n, slots, init, refill, step, w, lambda a, b: move_orient(a, b, tol_deg, tol_nm), ex, pool, cap, patience,
                          accepted=lambda s: np.asarray(s[3]), max_rejects=max_rejects, hold=delta_hold)
        th = r['th']
        bg = np.array(pool['background']) + th[:, 7:]
        out = {k: th[:, j] for j, k in enumerate(('x', 'y', 'z', 'N', 'rho', 'eta', 'delta'))}
        return dict(out, background=bg, steps=steps)
    return fit
