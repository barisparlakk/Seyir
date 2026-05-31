import React, { useEffect, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";

const MAX_HISTORY = 120; // ~12 s at 10 Hz

function Gauge({ label, value, unit, min = 0, max = 100, color = "#3b82f6" }) {
  const pct = Math.min(1, Math.max(0, (value - min) / (max - min)));
  const deg = -135 + pct * 270;

  return (
    <div style={styles.gauge}>
      <svg viewBox="0 0 100 70" style={{ width: "100%", height: 70 }}>
        {/* Track arc */}
        <path d="M 10 65 A 45 45 0 1 1 90 65" fill="none" stroke="#2d3748" strokeWidth={8} strokeLinecap="round" />
        {/* Value arc */}
        <path
          d="M 10 65 A 45 45 0 1 1 90 65"
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={`${pct * 141.4} 141.4`}
        />
        {/* Needle */}
        <g transform={`rotate(${deg}, 50, 65)`}>
          <line x1="50" y1="65" x2="50" y2="25" stroke={color} strokeWidth={2} strokeLinecap="round" />
          <circle cx="50" cy="65" r="4" fill={color} />
        </g>
        <text x="50" y="58" textAnchor="middle" fill="#e2e8f0" fontSize={14} fontWeight={700}>
          {typeof value === "number" ? value.toFixed(1) : "--"}
        </text>
        <text x="50" y="68" textAnchor="middle" fill="#64748b" fontSize={7}>
          {unit}
        </text>
      </svg>
      <div style={styles.gaugeLabel}>{label}</div>
    </div>
  );
}

export default function MetricsPanel({ live, metrics }) {
  const [history, setHistory] = useState([]);
  const tickRef = useRef(0);

  useEffect(() => {
    if (!live) return;
    tickRef.current += 1;
    setHistory((prev) => [
      ...prev.slice(-MAX_HISTORY + 1),
      { t: tickRef.current, speed: live.ego?.speed_kmh ?? 0 },
    ]);
  }, [live]);

  const speed   = live?.ego?.speed_kmh ?? 0;
  const ttc     = metrics?.min_ttc ?? 99;
  const jerkLon = metrics?.avg_longitudinal_jerk ?? 0;
  const collisions = metrics?.collision_count ?? 0;

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Live Metrics</h3>

      <div style={styles.gaugeRow}>
        <Gauge label="Speed" value={speed}   unit="km/h" min={0} max={120} color="#3b82f6" />
        <Gauge label="TTC"   value={Math.min(ttc, 10)} unit="s" min={0} max={10}
               color={ttc < 2 ? "#ef4444" : ttc < 4 ? "#f59e0b" : "#22c55e"} />
        <Gauge label="Jerk"  value={jerkLon} unit="m/s³" min={0} max={5}
               color={jerkLon > 3 ? "#ef4444" : "#a855f7"} />
      </div>

      <div style={styles.collisionBadge}>
        <span style={{ color: collisions > 0 ? "#ef4444" : "#22c55e", fontWeight: 700 }}>
          {collisions}
        </span>
        <span style={{ color: "#64748b", fontSize: 12 }}> collisions</span>
      </div>

      <div style={styles.chartWrap}>
        <div style={styles.chartLabel}>Speed history (km/h)</div>
        <ResponsiveContainer width="100%" height={90}>
          <LineChart data={history}>
            <XAxis dataKey="t" hide />
            <YAxis domain={[0, 120]} hide />
            <Tooltip
              contentStyle={{ background: "#161b27", border: "1px solid #2d3748", borderRadius: 6 }}
              labelStyle={{ display: "none" }}
              formatter={(v) => [`${v.toFixed(1)} km/h`, "Speed"]}
            />
            <Line type="monotone" dataKey="speed" stroke="#3b82f6" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
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
    marginBottom: 14,
  },
  gaugeRow: {
    display: "flex",
    gap: 8,
    justifyContent: "space-between",
  },
  gauge: {
    flex: 1,
    textAlign: "center",
  },
  gaugeLabel: {
    fontSize: 11,
    color: "#64748b",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginTop: -4,
  },
  collisionBadge: {
    textAlign: "center",
    margin: "8px 0",
    fontSize: 14,
  },
  chartWrap: {
    marginTop: 12,
  },
  chartLabel: {
    fontSize: 11,
    color: "#64748b",
    marginBottom: 4,
  },
};
