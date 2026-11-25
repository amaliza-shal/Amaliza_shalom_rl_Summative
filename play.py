"""Play script: run a trained School Director agent with the GUI and terminal output.

Same as main.py (kept so either name works):

  uv run play.py                        # best agent (PPO)
  uv run play.py --seed 4 --delay 6     # the demo episode used in the report and video
  uv run play.py --algo dqn --view3d    # any agent, plus the three.js 3D replay
"""

from main import main

if __name__ == "__main__":
    main()
