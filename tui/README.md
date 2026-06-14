# TUI

The terminal dashboard for Negev, built with [Textual](https://textual.textualize.io/).

## A note on what's hand-written here

This project is something I'm building to learn ML and security tooling
in depth, so the `core/` engine and the `api/` layer are written entirely
by hand. The UI is the deliberate exception.

Frontend and visual layout aren't where my learning focus is for this
project, and hand-crafting widget trees and styling would eat time I'd
rather spend on the parts I actually want to understand. So the visual
layer of this TUI, the widget composition, the layout, the `.tcss`
styling, is AI-assisted.

What I do write by hand in this folder is everything that isn't
purely visual:

- the HTTP logic that talks to the API (building requests, sending them,
  handling responses)
- how the TUI consumes and reacts to API data
- the wiring between user actions and backend calls
- some documentation AI writes for its visual functions that I believe could be better written are rewritten by me

In other words: the UI is a thin, replaceable shell, and it's treated as
one. The interface could be swapped for a web