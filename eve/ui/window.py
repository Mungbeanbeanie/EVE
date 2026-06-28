"""Native desktop window for EVE.

The visualizer is a web page (see :mod:`eve.ui.server`). This module wraps that
page in a *real* macOS window so EVE feels like an app instead of a browser tab —
and, crucially, lets it live in the **menu bar with no Dock icon** so it can run
always-on in the background without cluttering the app switcher.

Three backends, tried in order; the first that imports wins:

1. **rumps + WKWebView** (preferred) — a menu-bar (status-bar) app hosting the web
   UI in a native ``WKWebView``. ``LSUIElement`` activation policy ⇒ no Dock icon,
   not in Cmd-Tab. Show/Hide/Quit live in the menu-bar dropdown. Delivers the full
   "always-on background assistant" behavior in one process.
2. **pywebview** — a plain native window (no menu-bar item). Used when rumps/pyobjc
   isn't installed.
3. **browser tab** — final fallback: opens the orb in the default browser. Returns
   immediately (non-blocking) so the caller keeps the agent on the main thread.

The native backends run the macOS event loop and therefore **block the calling
thread**, which must be the process's main thread (Cocoa's hard requirement). The
agent loop runs on a worker thread in that case — see ``main.py``.
"""

from __future__ import annotations

import logging
import sys
import webbrowser
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Window geometry — a tall, narrow glass panel that suits the orb.
_WIDTH = 420
_HEIGHT = 620


def launch_window(
    url: str,
    *,
    title: str = "EVE",
    hide_dock: bool = True,
    menu_bar: bool = True,
    on_quit: Optional[Callable[[], None]] = None,
) -> bool:
    """Open EVE's window, blocking on the current (main) thread until it closes.

    Returns ``True`` if a native backend ran the UI loop (the call blocked until
    quit), or ``False`` if no native backend was available — in which case the orb
    was opened in a browser tab and the caller is responsible for staying alive.

    Args:
        url: The local visualizer URL (e.g. ``http://127.0.0.1:8765/``).
        title: Window / menu-bar title.
        hide_dock: Hide the Dock icon (menu-bar-only). macOS native backends only.
        menu_bar: Prefer the rumps menu-bar backend with Show/Hide/Quit controls.
        on_quit: Called once when the user quits the window, before the loop ends.
    """
    if sys.platform == "darwin" and menu_bar and _run_menu_bar_app(
        url, title=title, hide_dock=hide_dock, on_quit=on_quit
    ):
        return True

    if _run_pywebview(url, title=title, hide_dock=hide_dock, on_quit=on_quit):
        return True

    log.warning(
        "No native window backend available — opening the orb in your browser. "
        "Install the desktop deps for a real window: `make install-window`."
    )
    webbrowser.open(url)
    return False


# ── Backend 1: rumps menu-bar app hosting a WKWebView ────────────────────────
def _run_menu_bar_app(
    url: str,
    *,
    title: str,
    hide_dock: bool,
    on_quit: Optional[Callable[[], None]],
) -> bool:
    """Run a status-bar app with a native WebKit window. Blocks until quit."""
    try:
        import rumps
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSApplicationActivationPolicyRegular,
            NSBackingStoreBuffered,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskResizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSMakeRect, NSURL, NSURLRequest
        from WebKit import WKWebView, WKWebViewConfiguration
    except Exception as exc:  # pragma: no cover - import availability is runtime
        log.debug("menu-bar backend unavailable: %s", exc)
        return False

    _titled_style = (
        NSWindowStyleMaskTitled
        | NSWindowStyleMaskClosable
        | NSWindowStyleMaskMiniaturizable
        | NSWindowStyleMaskResizable
    )

    class _EveApp(rumps.App):
        """Menu-bar host. Builds the WebKit window lazily on first show."""

        def __init__(self) -> None:
            super().__init__(title, title=title, quit_button=None)
            self._window = None  # NSWindow, created on first _show
            self.menu = [
                rumps.MenuItem("Show EVE", callback=self._show),
                rumps.MenuItem("Hide EVE", callback=self._hide),
                None,
                rumps.MenuItem("Quit EVE", callback=self._quit),
            ]

        def _ensure_window(self) -> None:
            if self._window is not None:
                return
            rect = NSMakeRect(0, 0, _WIDTH, _HEIGHT)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, _titled_style, NSBackingStoreBuffered, False
            )
            win.setTitle_(title)
            # Keep the window object alive across close so Show can reopen it; a
            # closed window otherwise gets released and the next Show would crash.
            win.setReleasedWhenClosed_(False)

            webview = WKWebView.alloc().initWithFrame_configuration_(
                rect, WKWebViewConfiguration.alloc().init()
            )
            request = NSURLRequest.requestWithURL_(NSURL.URLWithString_(url))
            webview.loadRequest_(request)
            win.setContentView_(webview)
            win.center()
            self._window = win

        def _show(self, _sender) -> None:
            self._ensure_window()
            self._window.makeKeyAndOrderFront_(None)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        def _hide(self, _sender) -> None:
            if self._window is not None:
                self._window.orderOut_(None)  # hide without destroying

        def _quit(self, _sender) -> None:
            if on_quit is not None:
                on_quit()
            rumps.quit_application()

    app = _EveApp()
    # No Dock icon / not in Cmd-Tab when hidden; a normal app otherwise.
    NSApplication.sharedApplication().setActivationPolicy_(
        NSApplicationActivationPolicyAccessory
        if hide_dock
        else NSApplicationActivationPolicyRegular
    )
    log.info("EVE menu-bar app running (Dock icon %s).", "hidden" if hide_dock else "shown")
    # Open the window immediately so EVE is visible on launch; users can hide it.
    app._show(None)
    app.run()  # blocks on the main thread until Quit
    return True


# ── Backend 2: pywebview plain window ────────────────────────────────────────
def _run_pywebview(
    url: str,
    *,
    title: str,
    hide_dock: bool,
    on_quit: Optional[Callable[[], None]],
) -> bool:
    """Run a plain native window via pywebview. Blocks until the window closes."""
    try:
        import webview
    except Exception as exc:  # pragma: no cover - import availability is runtime
        log.debug("pywebview backend unavailable: %s", exc)
        return False

    window = webview.create_window(title, url, width=_WIDTH, height=_HEIGHT)
    if on_quit is not None:
        window.events.closed += on_quit
    if hide_dock and sys.platform == "darwin":
        log.info(
            "pywebview window has a Dock icon; install the menu-bar deps "
            "(`make install-window`) for a Dock-less menu-bar app."
        )
    log.info("EVE pywebview window running.")
    webview.start()  # blocks on the main thread until the window closes
    return True
