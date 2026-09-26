# Franka Stack integration branch

This fork carries the small set of OpenPI changes required by the
[Franka Stack](https://github.com/Loule0-0/franka-stack) reference policy
backend. The `franka-stack` branch is consumed as a pinned Git submodule; the
standalone tutorial, robot adapters, safety contracts, and data tools remain in
the parent repository.

The integration adds:

- the `pi05_franka_jointpos` training and serving configuration;
- the 8D Franka observation/action transforms;
- dataset and checkpoint provenance hooks;
- strict policy metadata exchange in the WebSocket client and server.

The Franka-specific training path imports the companion `franka-runtime`
package from the parent checkout. The parent bootstrap script installs that
package into this fork's locked environment after `uv sync --frozen`.

Upstream OpenPI authorship, history, and licenses are intentionally preserved
in this repository. General OpenPI changes should continue to target
[Physical Intelligence/OpenPI](https://github.com/Physical-Intelligence/openpi).
