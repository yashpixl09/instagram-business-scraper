"""The HTTP surface.

    schemas.py   the wire contract: request parsing, response models, the error envelope
    deps.py      the composition root and the service layer both surfaces call
    routes.py    paths and status codes, three lines apiece
    app.py       the application, the lifespan, and the four exception handlers

The API is the complete surface. The CLI in `lead_engine/cli.py` is a client of `deps.Engine`
and holds no behaviour of its own, so a web frontend can never need a capability that only a
terminal has.

Nothing is re-exported here. `create_app` lives in `app`, and importing it at package level
would run FastAPI's import cost for anything that only wanted `schemas`.
"""
