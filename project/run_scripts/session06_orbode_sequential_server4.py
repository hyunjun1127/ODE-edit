#!/usr/bin/env python3
"""Server4 entrypoint for ORBODE B100x10 sequential execution."""

from project.run_scripts.ordered_response_barrier_ode.sequential_launch import main


if __name__ == "__main__":
    raise SystemExit(main())

