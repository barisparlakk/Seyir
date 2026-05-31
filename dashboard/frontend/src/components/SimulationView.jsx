import React, { useEffect, useRef } from "react";

const CANVAS_SIZE = 400;
const SCALE = 4;         // pixels per metre
const EGO_HALF = CANVAS_SIZE / 2;

const CLASS_COLORS = {
  car:        "#3b82f6",
  truck:      "#f59e0b",
  bus:        "#8b5cf6",
  motorcycle: "#06b6d4",
  person:     "#ec4899",
  default:    "#94a3b8",
};

export default function SimulationView({ live }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Background
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid
    ctx.strokeStyle = "#1e2433";
    ctx.lineWidth = 1;
    for (let i = 0; i <= CANVAS_SIZE; i += SCALE * 5) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, CANVAS_SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(CANVAS_SIZE, i); ctx.stroke();
    }

    if (!live) {
      ctx.fillStyle = "#2d3748";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for simulation…", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      return;
    }

    const ego = live.ego ?? {};
    const detections = live.detections ?? [];

    // Draw detected agents relative to ego
    detections.forEach((det) => {
      const cx = EGO_HALF + (det.y ?? 0) * SCALE;  // lateral
      const cy = EGO_HALF - (det.x ?? 0) * SCALE;  // forward

      const color = CLASS_COLORS[det.class] ?? CLASS_COLORS.default;
      ctx.fillStyle = color + "99";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(cx - 8, cy - 12, 16, 24, 3);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(det.class ?? "?", cx, cy - 15);
      ctx.fillText(`${(det.distance ?? 0).toFixed(1)}m`, cx, cy + 30);

      // Predicted trajectory line (if provided)
      if (det.predicted_path && det.predicted_path.length > 1) {
        ctx.strokeStyle = color + "66";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        det.predicted_path.forEach(([px, py], i) => {
          const sx = EGO_HALF + py * SCALE;
          const sy = EGO_HALF - px * SCALE;
          i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Ego vehicle (arrow)
    ctx.save();
    ctx.translate(EGO_HALF, EGO_HALF);
    ctx.rotate(ego.heading ? -(ego.heading * Math.PI) / 180 : 0);
    ctx.fillStyle = "#22c55e";
    ctx.strokeStyle = "#16a34a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -18);
    ctx.lineTo(-10, 12);
    ctx.lineTo(0, 6);
    ctx.lineTo(10, 12);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Compass
    ctx.fillStyle = "#475569";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("N", CANVAS_SIZE / 2, 14);

    // Speed overlay
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "bold 13px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${(ego.speed_kmh ?? 0).toFixed(1)} km/h`, 8, CANVAS_SIZE - 8);
  }, [live]);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Simulation View</h3>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        style={styles.canvas}
      />
    </div>
  );
}

const styles = {
  container: {
    background: "#161b27",
    border: "1px solid #2d3748",
    borderRadius: 10,
    padding: "16px 20px",
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    marginBottom: 12,
  },
  canvas: {
    borderRadius: 6,
    display: "block",
    width: "100%",
    maxWidth: CANVAS_SIZE,
  },
};
