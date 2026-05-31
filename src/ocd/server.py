"""Local web server — the single, multi-user UI for the whole pipeline.

All real work goes through :mod:`ocd.service` (the shared layer the CLI uses too), so this
module is just HTTP + auth glue. Each request operates on the logged-in user's workspace
(``data/users/<user>/``).

Flow mirrors the pipeline: set **categories** → upload **statements** → **analyze**
(extract + categorize, streamed) → **review & correct** flagged rows → **finalize**, which
builds the report. Auth is lightweight (see :mod:`ocd.auth`): fine for local/LAN, not a
hardened public service. Launch with ``ocd serve``.
"""
import json
import logging
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

from . import auth, paths, service

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).with_name("web")
COOKIE = "ocd_session"
_REPORTS: dict[str, str] = {}  # job_id -> report HTML

# Cross-origin demo mode: when the frontend is served from a different origin
# (e.g. GitHub Pages) than this backend (e.g. an ngrok tunnel), set
# ``OCD_CORS_ORIGINS`` to a comma-separated allow-list of page origins. That
# enables credentialed CORS and a ``SameSite=None; Secure`` session cookie so
# the auth cookie survives cross-site requests. Unset → local same-origin
# behaviour (open CORS, ``SameSite=Lax``), unchanged.
PUBLIC_ORIGINS = [o.strip() for o in os.environ.get("OCD_CORS_ORIGINS", "").split(",") if o.strip()]
CROSS_SITE = bool(PUBLIC_ORIGINS)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def build_app():
    from fastapi import Cookie, Depends, FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                   StreamingResponse)
    from pydantic import BaseModel

    app = FastAPI(title="OCD — local server")
    if CROSS_SITE:
        # Credentialed CORS cannot use "*"; echo the configured page origins.
        app.add_middleware(CORSMiddleware, allow_origins=PUBLIC_ORIGINS,
                           allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    else:
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

    class Credentials(BaseModel):
        username: str
        password: str

    class CategoriesIn(BaseModel):
        categories: list[dict]

    class CorrectionsIn(BaseModel):
        corrections: dict[str, str]

    def current_user(ocd_session: Optional[str] = Cookie(default=None)) -> str:
        user = auth.session_user(ocd_session)
        if not user:
            raise HTTPException(status_code=401, detail="Please log in.")
        return user

    def home_of(user: str) -> Path:
        return paths.user_home(user)

    def _auth_response(user: str):
        token = auth.new_session(user)
        resp = JSONResponse({"user": user})
        if CROSS_SITE:
            # SameSite=None requires Secure; the tunnel terminates HTTPS so this holds.
            resp.set_cookie(COOKIE, token, httponly=True, samesite="none", secure=True,
                            max_age=7 * 24 * 3600)
        else:
            resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
        return resp

    # ---- static + health ----
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    def health():
        return service.model_health()

    # ---- auth ----
    @app.post("/api/signup")
    def signup(creds: Credentials):
        try:
            user = auth.create_user(creds.username, creds.password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        service.ensure_workspace(home_of(user))
        return _auth_response(user)

    @app.post("/api/login")
    def login(creds: Credentials):
        username = creds.username.strip().lower()
        if not auth.verify_password(username, creds.password):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        return _auth_response(username)

    @app.post("/api/logout")
    def logout(ocd_session: Optional[str] = Cookie(default=None)):
        auth.end_session(ocd_session)
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/api/me")
    def me(user: str = Depends(current_user)):
        return {"user": user, "statements": service.list_statements(home_of(user))}

    # ---- categories (preferences) ----
    @app.get("/api/categories")
    def categories(user: str = Depends(current_user)):
        return {"categories": service.get_categories(home_of(user))}

    @app.put("/api/categories")
    def set_categories(body: CategoriesIn, user: str = Depends(current_user)):
        try:
            return {"categories": service.save_categories(home_of(user), body.categories)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---- statements ----
    @app.get("/api/statements")
    def statements(user: str = Depends(current_user)):
        return {"statements": service.list_statements(home_of(user))}

    @app.post("/api/upload")
    async def upload(user: str = Depends(current_user), files: list[UploadFile] = File(...)):
        payload = {f.filename: await f.read() for f in files if f.filename}
        saved = service.save_statements(home_of(user), payload)
        if saved == 0:
            raise HTTPException(status_code=400, detail="No PDF files in the upload.")
        return {"added": saved, "statements": service.list_statements(home_of(user))}

    @app.delete("/api/statements/{name}")
    def delete_statement(name: str, user: str = Depends(current_user)):
        if not service.delete_statement(home_of(user), name):
            raise HTTPException(status_code=404, detail="No such statement.")
        return {"statements": service.list_statements(home_of(user))}

    # ---- analyze (extract + categorize), streamed ----
    @app.get("/api/analyze/stream")
    def analyze_stream(user: str = Depends(current_user)):
        home = home_of(user)

        def gen():
            q: queue.Queue = queue.Queue()

            def worker():
                try:
                    service.analyze(home, on_event=q.put)
                except Exception as e:  # noqa: BLE001
                    q.put({"stage": "error", "detail": str(e)})
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield _sse(ev)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- review + corrections ----
    @app.get("/api/review")
    def review(user: str = Depends(current_user)):
        return service.get_review(home_of(user))

    @app.post("/api/corrections")
    def corrections(body: CorrectionsIn, user: str = Depends(current_user)):
        try:
            return service.apply_corrections(home_of(user), body.corrections)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Run Analyze first.")

    @app.post("/api/recategorize")
    def recategorize(body: CorrectionsIn, user: str = Depends(current_user)):
        """Apply the user's edits, surface any contradictions, then re-run categorization
        (the model re-decides everything not user-confirmed). Streamed as SSE."""
        home = home_of(user)
        corr = body.corrections

        def gen():
            q: queue.Queue = queue.Queue()

            def worker():
                try:
                    conflicts = service.apply_and_check(home, corr)
                    if conflicts:
                        q.put({"stage": "conflict", "conflicts": conflicts})
                    else:
                        q.put({"stage": "done", "review": service.recategorize(home, on_event=q.put)})
                except FileNotFoundError:
                    q.put({"stage": "error", "detail": "Run Analyze first."})
                except Exception as e:  # noqa: BLE001
                    q.put({"stage": "error", "detail": str(e)})
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield _sse(ev)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- finalize + report ----
    @app.post("/api/finalize")
    def finalize(user: str = Depends(current_user)):
        try:
            html = service.finalize_and_report(home_of(user))
        except Exception as e:  # noqa: BLE001
            logger.warning("Finalize failed for %s: %s", user, e)
            raise HTTPException(status_code=500, detail=str(e))
        job_id = uuid.uuid4().hex[:12]
        _REPORTS[job_id] = html
        return {"report_url": f"/report/{job_id}"}

    @app.get("/report/{job_id}")
    def report(job_id: str):
        html = _REPORTS.get(job_id)
        if html is None:
            raise HTTPException(status_code=404, detail="Unknown or expired report id.")
        return HTMLResponse(html)

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    logger.info("Starting OCD server on http://%s:%d", host, port)
    uvicorn.run(build_app(), host=host, port=port)
