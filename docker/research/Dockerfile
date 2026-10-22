FROM python:3.11-slim-bookworm
# scripts/build_research_image.py supplies the version-checked project CLI and
# bundled rg, plus imageio-ffmpeg's standalone binary. No credentials or repo code.
COPY codex codex-code-mode-host rg ffmpeg /usr/local/bin/
RUN test "$(codex --version)" = "codex-cli 0.162.1"
COPY bridge.py /opt/simex/bridge.py
COPY research_rpc.py /usr/local/bin/simex-research
RUN chmod 755 /usr/local/bin/simex-research
ENV PATH=/usr/local/bin:/usr/bin:/bin CODEX_HOME=/session LANG=C.UTF-8
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/simex/bridge.py"]
