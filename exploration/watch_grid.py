"""
watch_grid.py
LIVE visual view of a Grid2Op episode - the analog of the Donkey camera window.

Opens a matplotlib window and redraws the network each step while a do-nothing
agent runs, so you can watch line loadings climb and the grid collapse. Line
labels show loading %; the title shows step, max rho and the safety state
(OK / NEAR LIMIT / OVERLOAD), coloured green/orange/red like the guard HUD.

    python watch_grid.py                       # live window, scenario 0
    python watch_grid.py --scenario 2 --pause 0.05
    python watch_grid.py --save frames         # also write PNGs (for a GIF/video)
    python watch_grid.py --headless --save frames   # no window, frames only

If no window appears, your matplotlib backend is non-interactive: run
    python -c "import matplotlib; print(matplotlib.get_backend())"
and if it prints 'Agg', install a GUI backend (pip install pyqt5) or use --save.
"""
import argparse, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="rte_case5_example")
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--pause", type=float, default=0.08, help="seconds per frame")
    ap.add_argument("--every", type=int, default=1, help="draw every Nth step")
    ap.add_argument("--save", default=None, help="folder to write PNG frames into")
    ap.add_argument("--headless", action="store_true", help="no window (Agg)")
    args = ap.parse_args()

    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("matplotlib backend:", matplotlib.get_backend())

    import grid2op
    from grid2op.Agent import DoNothingAgent
    from grid2op.PlotGrid import PlotMatplot

    env = grid2op.make(args.env, test=True)
    agent = DoNothingAgent(env.action_space)
    plotter = PlotMatplot(env.observation_space)

    # advance to the requested scenario
    obs = env.reset()
    for _ in range(args.scenario):
        obs = env.reset()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
    if not args.headless:
        plt.ion()

    fig = plotter.plot_obs(obs)
    ax = fig.axes[0]
    if not args.headless:
        plt.show(block=False)

    done, step = False, 0
    print(f"watching '{args.env}' scenario {args.scenario} "
          f"(do-nothing agent, max {args.max_steps} steps)")
    try:
        while not done and step < args.max_steps:
            obs, reward, done, info = env.step(agent.act(obs, 0.0, done))
            step += 1
            if step % args.every:
                continue

            mx = float(obs.rho.max())
            if mx > 1.0:
                state, colour = "OVERLOAD", "red"
            elif mx > 0.9:
                state, colour = "NEAR LIMIT", "darkorange"
            else:
                state, colour = "OK", "green"

            ax.clear()
            plotter.plot_obs(obs, figure=fig)
            fig.axes[0].set_title(
                f"step {step}   max rho {mx:.3f}   {state}",
                color=colour, fontsize=13, fontweight="bold")
            if args.save:
                fig.savefig(os.path.join(args.save, f"f{step:05d}.png"), dpi=90)
            if not args.headless:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(args.pause)
            if step % 20 == 0:
                print(f"  step {step:4d}  max rho {mx:.3f}  {state}", end="\r")
    except KeyboardInterrupt:
        print("\ninterrupted")

    print(f"\nepisode ended after {step} steps. game_over={done}")
    if done and info.get("exception"):
        print("cause:", str(info["exception"])[:140])
    if args.save:
        print(f"frames -> {args.save}/  (make a gif: "
              f"ffmpeg -framerate 10 -i {args.save}/f%05d.png grid.gif)")
    if not args.headless:
        plt.ioff()
        print("close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
