"""
watch_anm.py
LIVE visual view of a gym-anm episode - the gym-anm counterpart of watch_grid.py.

gym-anm renders into a WEB BROWSER: calling render() starts a small local server
and prints a URL. Open that URL and the 6-bus network animates as the episode
runs, showing device powers, branch loadings and bus voltages.

    python watch_anm.py                         # baseline, browser view
    python watch_anm.py --policy random --skip 10
    python watch_anm.py --no-render             # console only

Tips
  - render(skip_frames=N) only redraws every N+1 steps; the library itself
    recommends N>0 because real-time updates are too fast to follow.
  - the console trace prints the SAFETY state each redraw so the log lines and
    the browser picture stay in step.
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def safety(sim):
    v = sim.state["bus_v_magn"]["pu"]
    i = sim.state["branch_i_magn"]["pu"]
    volts = np.array([float(np.real(x)) for x in v.values()])
    load = []
    for br, cur in i.items():
        rate = getattr(sim.branches[br], "rate", None)
        if rate:
            load.append(abs(float(np.real(cur))) / rate)
    return volts, (max(load) if load else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ANM6Easy-v0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--skip", type=int, default=5,
                    help="redraw every skip+1 steps (0 = every step, too fast)")
    ap.add_argument("--policy", default="nocurt",
                    choices=["nocurt", "zero", "random"])
    ap.add_argument("--pause", type=float, default=0.15,
                    help="seconds between steps, so the browser can keep up")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    import gymnasium, gym_anm
    env = gymnasium.make(args.env)
    obs, info = env.reset()
    u = env.unwrapped
    sim = u.simulator

    if not args.no_render:
        print("starting the browser renderer...")
        u.render(mode="human", skip_frames=args.skip)
        print(">>> OPEN THE URL ABOVE IN YOUR BROWSER, then watch it animate <<<\n")
        time.sleep(3)   # give yourself a moment to open it

    hi = env.action_space.high.copy()

    def act():
        if args.policy == "nocurt":
            a = hi.copy(); a[2:] = 0.0
            return a
        if args.policy == "zero":
            return np.zeros_like(hi)
        return env.action_space.sample()

    print(f"running '{args.policy}' for {args.steps} steps "
          f"(redraw every {args.skip+1})")
    print(f"{'step':>5} {'Vmin':>7} {'Vmax':>7} {'maxload':>8}  state")
    term = trunc = False
    s = 0
    try:
        while not (term or trunc) and s < args.steps:
            obs, r, term, trunc, info = env.step(act())
            s += 1
            if not args.no_render:
                u.render(mode="human", skip_frames=args.skip)
            volts, ml = safety(sim)
            vn = volts[1:]           # exclude slack bus
            if ml > 1.0:
                state = "OVERLOAD"
            elif vn.min() < 0.95 or vn.max() > 1.05:
                state = "VOLTAGE VIOLATION"
            elif ml > 0.9:
                state = "near limit"
            else:
                state = "ok"
            if s % (args.skip + 1) == 0:
                print(f"{s:>5} {vn.min():>7.4f} {vn.max():>7.4f} "
                      f"{ml:>8.3f}  {state}")
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted")

    print(f"\nended after {s} steps (terminated={term}, truncated={trunc})")
    print("Safety properties used here: voltage band 0.95-1.05 pu (EN 50160 style)")
    print("and branch current vs its thermal rate (>1.0 = overload).")


if __name__ == "__main__":
    main()
