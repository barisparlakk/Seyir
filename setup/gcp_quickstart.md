# GCP Quick-Start — CARLA Server for Seyir

## Step 1 — Create your Google Cloud account

1. Go to https://cloud.google.com
2. Click **Start free** → sign in with your Google account
3. Enter a credit card (required for verification — **you will NOT be charged during the trial**)
4. You receive **$300 free credits** valid for 90 days

---

## Step 2 — Create the VM

### Via the web console (easiest)

1. Open https://console.cloud.google.com/compute/instances
2. Click **Create Instance**
3. Fill in:

   | Field | Value |
   |---|---|
   | Name | `seyir-carla` |
   | Region | `us-central1` (Iowa) — cheapest |
   | Zone | `us-central1-a` |
   | Machine family | **GPU** tab |
   | GPU type | **NVIDIA T4** |
   | Number of GPUs | 1 |
   | Machine type | `n1-standard-4` (4 vCPU, 15 GB RAM) |
   | Boot disk OS | **Ubuntu 22.04 LTS** |
   | Boot disk size | **100 GB** SSD |
   | Firewall | ✅ Allow HTTP traffic, ✅ Allow HTTPS traffic |

4. Click **Create** — the VM starts in ~2 minutes.

### Via `gcloud` CLI (if you have it installed)

```bash
gcloud compute instances create seyir-carla \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --tags=carla-server
```

---

## Step 3 — Open firewall ports for CARLA

CARLA needs TCP ports 2000–2002.

### Web console

1. **VPC Network → Firewall → Create Firewall Rule**
2. Fill in:

   | Field | Value |
   |---|---|
   | Name | `carla-ports` |
   | Direction | Ingress |
   | Targets | All instances in network |
   | Source IPv4 | `0.0.0.0/0` |
   | Protocols/ports | `tcp:2000-2002` |

3. Click **Create**

### Via `gcloud` CLI

```bash
gcloud compute firewall-rules create carla-ports \
  --allow tcp:2000-2002 \
  --source-ranges 0.0.0.0/0 \
  --description "CARLA simulator RPC + streaming ports"
```

---

## Step 4 — SSH into the VM

Click the **SSH** button next to your instance in the console, or:

```bash
gcloud compute ssh seyir-carla --zone us-central1-a
```

---

## Step 5 — Install CARLA

Once inside the VM:

```bash
# Upload the install script (from your Mac):
gcloud compute scp setup/install_carla_server.sh seyir-carla:~ --zone us-central1-a

# SSH in and run it:
gcloud compute ssh seyir-carla --zone us-central1-a
chmod +x ~/install_carla_server.sh
~/install_carla_server.sh
```

This script:
- Installs NVIDIA drivers + CUDA
- Downloads and extracts CARLA 0.9.15
- Creates `~/start_carla.sh` and `~/stop_carla.sh`
- Sets up a systemd service for auto-start on reboot

---

## Step 6 — Start CARLA

```bash
~/start_carla.sh
# Monitor startup (takes ~30 s):
tail -f ~/carla.log
# You should see: "4.27.2-0+++UE4+Release-4.27 522 0"
```

---

## Step 7 — Connect from your Mac

Find your VM's **External IP** in the GCP console (Compute Engine → VM Instances).

```bash
# On your Mac:
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

## Step 8 — Run Seyir

```bash
# All scripts accept --host:
python scripts/collect_data.py  --host <external-ip> --frames 5000
python scripts/run_simulation.py --host <external-ip> --scenario narrow_street --record
python scripts/evaluate.py       --host <external-ip> --runs 3
```

---

## Cost management

| Action | Command |
|---|---|
| **Stop the VM** (saves ~90% cost) | `gcloud compute instances stop seyir-carla --zone us-central1-a` |
| **Start it again** | `gcloud compute instances start seyir-carla --zone us-central1-a` |
| **Delete permanently** | `gcloud compute instances delete seyir-carla --zone us-central1-a` |

> Stop the VM whenever you're not using it. A T4 instance costs ~$0.35/hr running
> but only ~$0.01/hr stopped (just the disk). Over a 90-day trial you have $300 —
> that's ~857 hours of GPU time, more than enough for the full Seyir pipeline.

---

## No-GPU alternative (Oracle Free Tier, $0 forever)

1. Sign up at https://cloud.oracle.com/free
2. Create **VM.Standard.E2.4** (4 OCPU, 24 GB RAM, Ubuntu 22.04) — free forever
3. Run the install script with `--no-gpu`:
   ```bash
   ~/install_carla_server.sh --no-gpu
   ```
4. CARLA runs at ~5-8 fps in software rendering mode — sufficient for headless
   data collection and scenario evaluation, not great for real-time visualisation.
