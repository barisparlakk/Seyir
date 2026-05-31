#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_carla_server.sh
#
# Run this on a fresh Ubuntu 22.04 server (GPU or CPU) to install CARLA 0.9.15
# and everything it needs.
#
# Usage:
#   chmod +x setup/install_carla_server.sh
#   ./setup/install_carla_server.sh          # GPU server
#   ./setup/install_carla_server.sh --no-gpu # CPU-only server
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU=true
if [[ "${1:-}" == "--no-gpu" ]]; then
  GPU=false
fi

CARLA_VERSION="0.9.15"
CARLA_DIR="$HOME/CARLA_${CARLA_VERSION}"
CARLA_URL="https://github.com/carla-simulator/carla/releases/download/${CARLA_VERSION}/CARLA_${CARLA_VERSION}.tar.gz"

echo "========================================================"
echo " Seyir — CARLA ${CARLA_VERSION} server setup"
echo " GPU mode: ${GPU}"
echo " Target  : ${CARLA_DIR}"
echo "========================================================"

# ── 1. System packages ────────────────────────────────────────────────────── #
echo
echo "[1/6] Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  wget curl tar python3 python3-pip python3-venv \
  libomp5 libsdl2-2.0-0 libx11-6 libxrandr2 \
  libvulkan1 libgl1 libglu1-mesa

# ── 2. NVIDIA driver + CUDA (GPU only) ───────────────────────────────────── #
if [[ "$GPU" == "true" ]]; then
  echo
  echo "[2/6] Installing NVIDIA driver and CUDA toolkit…"
  sudo apt-get install -y --no-install-recommends \
    nvidia-driver-535 nvidia-cuda-toolkit
  echo "      Verifying GPU…"
  nvidia-smi || {
    echo "WARNING: nvidia-smi failed. Reboot may be required."
    echo "         Run this script again after reboot, or use --no-gpu."
  }
else
  echo "[2/6] Skipping GPU drivers (--no-gpu)"
fi

# ── 3. Download CARLA ─────────────────────────────────────────────────────── #
echo
echo "[3/6] Downloading CARLA ${CARLA_VERSION}…"
if [[ ! -f "/tmp/CARLA_${CARLA_VERSION}.tar.gz" ]]; then
  wget -q --show-progress -O "/tmp/CARLA_${CARLA_VERSION}.tar.gz" "${CARLA_URL}"
else
  echo "      Archive already downloaded, skipping."
fi

echo "      Extracting…"
mkdir -p "${CARLA_DIR}"
tar -xf "/tmp/CARLA_${CARLA_VERSION}.tar.gz" -C "${CARLA_DIR}" --strip-components=0
echo "      Extracted to ${CARLA_DIR}"

# ── 4. Install CARLA Python egg ───────────────────────────────────────────── #
echo
echo "[4/6] Installing CARLA Python API…"
CARLA_EGG=$(find "${CARLA_DIR}" -name "carla-*.egg" | head -1)
if [[ -n "$CARLA_EGG" ]]; then
  pip3 install --quiet "${CARLA_EGG}"
  echo "      Installed from egg: ${CARLA_EGG}"
else
  pip3 install --quiet carla=="${CARLA_VERSION}" && echo "      Installed via pip"
fi

# ── 5. Create start/stop scripts ─────────────────────────────────────────── #
echo
echo "[5/6] Writing start/stop helper scripts…"

cat > "$HOME/start_carla.sh" <<EOF
#!/usr/bin/env bash
# Start CARLA server in background
CARLA_DIR="${CARLA_DIR}"
EOF

if [[ "$GPU" == "true" ]]; then
  cat >> "$HOME/start_carla.sh" <<'SCRIPT'
"\${CARLA_DIR}/CarlaUE4.sh" \
  -RenderOffScreen \
  -nosound \
  -carla-rpc-port=2000 \
  -fps=20 \
  -quality-level=Low \
  > "$HOME/carla.log" 2>&1 &
SCRIPT
else
  cat >> "$HOME/start_carla.sh" <<'SCRIPT'
SDL_VIDEODRIVER=offscreen "\${CARLA_DIR}/CarlaUE4.sh" \
  -RenderOffScreen \
  -nosound \
  -opengl \
  -carla-rpc-port=2000 \
  -fps=10 \
  -quality-level=Low \
  > "$HOME/carla.log" 2>&1 &
SCRIPT
fi

cat >> "$HOME/start_carla.sh" <<'EOF'
echo "CARLA starting (PID $!)… tail -f ~/carla.log to monitor"
echo $! > "$HOME/carla.pid"
EOF

cat > "$HOME/stop_carla.sh" <<'EOF'
#!/usr/bin/env bash
if [[ -f "$HOME/carla.pid" ]]; then
  kill "$(cat "$HOME/carla.pid")" 2>/dev/null && echo "CARLA stopped"
  rm "$HOME/carla.pid"
else
  pkill -f CarlaUE4 && echo "CARLA stopped" || echo "CARLA not running"
fi
EOF

chmod +x "$HOME/start_carla.sh" "$HOME/stop_carla.sh"

# ── 6. Create systemd service (auto-start on boot) ───────────────────────── #
echo
echo "[6/6] Creating systemd service (carla.service)…"

EXEC_CMD="${CARLA_DIR}/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -fps=20 -quality-level=Low"
if [[ "$GPU" == "false" ]]; then
  EXEC_CMD="SDL_VIDEODRIVER=offscreen ${EXEC_CMD} -opengl"
fi

sudo tee /etc/systemd/system/carla.service > /dev/null <<EOF
[Unit]
Description=CARLA Simulator Server ${CARLA_VERSION}
After=network.target

[Service]
Type=simple
User=${USER}
ExecStart=${EXEC_CMD}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable carla.service
echo "      systemd service created (but NOT started yet)"

# ── Done ─────────────────────────────────────────────────────────────────── #
EXTERNAL_IP=$(curl -s --max-time 3 https://ipinfo.io/ip 2>/dev/null || echo "unknown")

echo
echo "========================================================"
echo " Installation complete!"
echo
echo " Start CARLA:          ~/start_carla.sh"
echo " Stop  CARLA:          ~/stop_carla.sh"
echo " Logs:                 tail -f ~/carla.log"
echo
echo " Auto-start on boot:   sudo systemctl start carla"
echo
echo " Your server IP:       ${EXTERNAL_IP}"
echo
echo " From your Mac, verify the connection:"
echo "   python scripts/check_connection.py --host ${EXTERNAL_IP}"
echo
echo " Then run a scenario:"
echo "   python scripts/run_simulation.py --host ${EXTERNAL_IP} --scenario narrow_street"
echo "========================================================"
