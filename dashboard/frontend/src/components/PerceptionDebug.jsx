import React, { useEffect, useRef, useState } from "react";

export default function PerceptionDebug({ live }) {
  const canvasRef = useRef(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLanes, setShowLanes] = useState(true);
  const [showDepth, setShowDepth] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, W, H);

    if (!live) {
      ctx.fillStyle = "#2d3748";
      ctx.font = "13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Camera feed not available", W / 2, H / 2);
      return;
    }

    const detections = live.detections ?? [];

    // Depth overlay (faint gradient background when enabled)
    if (showDepth) {
      const grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "#1e3a5f44");
      grad.addColorStop(1, "#0f111700");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }

    // Detection bounding boxes
    if (showBoxes) {
      detections.forEach((det) => {
        const { bbox, class: cls, confidence, distance } = det;
        if (!bbox) return;
        const [x1, y1, x2, y2] = bbox;

        const sx = (x1 / 1280) * W;
        const sy = (y1 / 720) * H;
        const sw = ((x2 - x1) / 1280) * W;
        const sh = ((y2 - y1) / 720) * H;

        const color = CLASS_COLORS[cls] ?? "#94a3b8";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx, sy, sw, sh);

        // Label background
        const label = `${cls ?? "?"} ${confidence ? (confidence * 100).toFixed(0) + "%" : ""}`;
        ctx.font = "bold 11px monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color + "cc";
        ctx.fillRect(sx, sy - 18, tw + 8, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, sx + 4, sy - 4);

        if (distance != null) {
          ctx.fillStyle = color;
          ctx.font = "10px monospace";
          ctx.fillText(`${distance.toFixed(1)}m`, sx + 4, sy + sh - 4);
        }
      });
    }

    // Lane lines overlay
    if (showLanes && live.lanes) {
      live.lanes.forEach((lane) => {
        if (!lane.points || lane.points.length < 2) return;
        const color = lane.lane_type === "implicit" ? "#f59e0b" : "#22c55e";
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash(lane.lane_type === "dashed" ? [10, 8] : []);
        ctx.beginPath();
        lane.points.forEach(([px, py], i) => {
          const sx = (px / 1280) * W;
          const sy = (py / 720) * H;
          i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // No-signal indicator
    if (detections.length === 0 && (!live.lanes || live.lanes.length === 0)) {
      ctx.fillStyle = "#2d374888";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No detections", W / 2, H - 14);
    }
  }, [live, showBoxes, showLanes, showDepth]);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Perception Debug</h3>
        <div style={styles.toggles}>
          <Toggle label="Boxes" active={showBoxes} onToggle={() => setShowBoxes((p) => !p)} color="#3b82f6" />
          <Toggle label="Lanes" active={showLanes} onToggle={() => setShowLanes((p) => !p)} color="#22c55e" />
          <Toggle label="Depth" active={showDepth} onToggle={() => setShowDepth((p) => !p)} color="#06b6d4" />
        </div>
      </div>
      <canvas ref={canvasRef} width={640} height={360} style={styles.canvas} />
    </div>
  );
}

function Toggle({ label, active, onToggle, color }) {
  return (
    <button
      onClick={onToggle}
      style={{
        padding: "4px 10px",
        borderRadius: 5,
        border: `1.5px solid ${active ? color : "#2d3748"}`,
        background: active ? color + "22" : "transparent",
        color: active ? color : "#475569",
        fontSize: 11,
        fontWeight: 600,
        cursor: "pointer",
        transition: "all 0.15s",
      }}
    >
      {label}
    </button>
  );
}

const CLASS_COLORS = {
  car: "#3b82f6", truck: "#f59e0b", bus: "#8b5cf6",
  motorcycle: "#06b6d4", person: "#ec4899",
  turkish_stop_sign: "#ef4444", turkish_speed_limit: "#f97316",
  horse_cart: "#84cc16", tractor: "#a78bfa",
};

const styles = {
  container: {
    background: "#161b27",
    border: "1px solid #2d3748",
    borderRadius: 10,
    padding: "16px 20px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  toggles: { display: "flex", gap: 6 },
  canvas: {
    borderRadius: 6,
    display: "block",
    width: "100%",
    maxWidth: 640,
  },
};
