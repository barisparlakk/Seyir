# GCP Quick-Start — Free CARLA Server for Seyir (CPU-only, $0)

> **Note:** GCP blocks GPU quota on free trial accounts. This guide uses a
> CPU-only VM instead. CARLA runs in software rendering at ~5-8 fps — slow
> but fully functional for headless data collection, training, and evaluation.
> Your $300 free credits cover ~1000 hours on this instance.

---

## Step 1 — Create your Google Cloud account

1. Go to https://cloud.google.com → click **Start free**
2. Sign in with your Google account
3. Enter a credit card (**you will NOT be charged during the trial**)
4. You get **$300 free credits** valid for 90 days

---

## Step 2 — Create the VM

### Via the web console

1. Open: https://console.cloud.google.com/compute/instances
2. Click **Create Instance**
3. Fill in:

   | Field | Value |
   |---|---|
   | Name | `seyir-carla` |
   | Region | `us-central1` (Iowa) — cheapest |
   | Zone | `us-central1-a` |
   | Machine family | **General purpose** |
   | Series | **E2** |
   | Machine type | `e2-standard-8` (8 vCPU, 32 GB RAM) |
   | Boot disk OS | **Ubuntu 22.04 LTS** |
   | Boot disk size | **60 GB** SSD |

4. Click **Create** — VM is ready in ~1 minute.

**Cost:** `e2-standard-8` costs ~$0.27/hr, covered by free credits.
At this rate your $300 covers ~1100 hours — more than enough.

### Via `gcloud` CLI

```bash
gcloud compute instances create seyir-carla \
  --zone=us-central1-a \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB \
  --boot-disk-type=pd-ssd
```

---

## Step 3 — Open firewall ports for CARLA

### Web console

1. **VPC Network → Firewall → Create Firewall Rule**

   | Field | Value |
   |---|---|
   | Name | `carla-ports` |
   | Direction | Ingress |
   | Targets | All instances in network |
   | Source IPv4 | `0.0.0.0/0` |
   | Protocols/ports | `tcp:2000-2002` |

2. Click **Create**

### Via `gcloud` CLI

```bash
gcloud compute firewall-rules create carla-ports \
  --allow tcp:2000-2002 \
  --source-ranges 0.0.0.0/0 \
  --description "CARLA RPC and streaming ports"
```

---

## Step 4 — SSH into the VM

Click the **SSH** button in the console next to your instance, or:

```bash
gcloud compute ssh seyir-carla --zone us-central1-a
```

---

## Step 5 — Install CARLA (one command)

Inside the VM, run:

```bash
wget -O install.sh https://raw.githubusercontent.com/YOUR_REPO/main/setup/install_carla_server.sh
chmod +x install.sh && ./install.sh --no-gpu
```

Or if you have the repo cloned on the VM:

```bash
bash setup/install_carla_server.sh --no-gpu
```

This takes ~3-5 minutes. It will:
- Install required system packages
- Download CARLA 0.9.15 (~10 GB)
- Create `~/start_carla.sh` and `~/stop_carla.sh`
- Set up a systemd service

---

## Step 6 — Start CARLA

```bash
~/start_carla.sh

# Watch the startup log (takes ~30-60 seconds):
tail -f ~/carla.log

# You should eventually see something like:
# LogCarla: Initialized
```

---

## Step 7 — Note your external IP

In the GCP console → Compute Engine → VM Instances, copy the **External IP**.

Or from inside the VM:
```bash
curl -s https://ipinfo.io/ip
```

---

## Step 8 — Connect from your Mac

```bash
# On your Mac, in the Seyir project:
python scripts/check_connection.py --host <external-ip>
```

Expected output:
```
  [1/3] TCP port 2000 … OK
  [2/3] CARLA Python package … OK
  [3/3] CARLA handshake … OK

  ✓ Connected successfully
  Server version  : 0.9.15
  Current map     : /Game/Carla/Maps/Town03
```

---

## Step 9 — Run Seyir

All scripts accept `--host`:

```bash
python scripts/collect_data.py   --host <external-ip> --frames 5000
python scripts/train_detector.py
python scripts/train_predictor.py
python scripts/run_simulation.py --host <external-ip> --scenario narrow_street --record
python scripts/evaluate.py       --host <external-ip> --runs 3
```

Training scripts (`train_detector.py`, `train_predictor.py`) run entirely
on your Mac using collected data — no server needed for those.

---

## Cost management — stop the VM when not in use

The VM costs ~$0.27/hr **only while running**. Stop it between sessions:

```bash
# Stop (keeps disk, ~$0.01/hr for storage only):
gcloud compute instances stop seyir-carla --zone us-central1-a

# Start again later:
gcloud compute instances start seyir-carla --zone us-central1-a

# Delete permanently when done:
gcloud compute instances delete seyir-carla --zone us-central1-a
```

Or click the **Stop** / **Start** buttons in the GCP console.

---

## Troubleshooting

**CARLA starts but immediately crashes:**
```bash
# Check available memory:
free -h
# e2-standard-8 has 32 GB — should be fine.
# If using a smaller instance, CARLA needs at least 8 GB.
```

**`check_connection.py` fails on TCP step:**
- Confirm the firewall rule was created (Step 3)
- Confirm CARLA is running: `cat ~/carla.pid && ps aux | grep Carla`
- Try from the VM itself: `python3 -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"`

**Very slow (< 3 fps):**
This is normal for CPU-only without Vulkan. Add to `start_carla.sh`:
```
-quality-level=Low -benchmark -fps=5
```
At 5 fps, 5000 training frames takes ~17 minutes — acceptable.

**External IP changed after restart:**
GCP external IPs are ephemeral by default. Either:
- Reserve a static IP (GCP → VPC Network → IP addresses → Reserve)
- Or just re-run `check_connection.py` with the new IP each time
