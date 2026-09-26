"""Entry point for ``python -m hand_tracker``; delegates to :func:`hand_tracker.app.main`."""

from hand_tracker.app import main

raise SystemExit(main())
