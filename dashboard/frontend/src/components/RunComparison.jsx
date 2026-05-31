import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell,
} from "recharts";

const METRICS_TO_COMPARE = [
  { key: "avg_speed_kmh",           label: "Avg Speed (km/h)",     good: "high" },
  { key: "collision_count",         label: "Collisions",           good: "low"  },
  { key: "traffic_violation_count", label: "Traffic Violations",   good: "low"  },
  { key: "route_completion_pct",    label: "Route Completion (%)", good: "high" },
  { key: "min_ttc",                 label: "Min TTC (s)",          good: "high" },
  { key: "avg_longitudinal_jerk",   label: "Long. Jerk (m/s³)",   good: "low"  },
  { key: "emergency_brake_count",   label: "Emergency Brakes",     good: "low"  },
];

const COLORS_A = "#3b82f6";
const COLORS_B = "#a855f7";

export default function RunComparison() {
  const [runs, setRuns] = useState([]);
  const [runA, setRunA] = useState("");
  const [runB, setRunB] = useState("");
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/api/runs")
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => setRuns([]));
  }, []);

  const handleCompare = async () => {
    if (!runA || !runB) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/runs/compare?run_a=${runA}&run_b=${runB}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setComparison(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = comparison
    ? METRICS_TO_COMPARE.map(({ key, label }) => ({
        metric: label,
        [runA]: comparison[runA]?.[key] ?? 0,
        [runB]: comparison[runB]?.[key] ?? 0,
      }))
    : [];

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Run Comparison</h3>

      <div style={styles.selectors}>
        <select value={runA} onChange={(e) => setRunA(e.target.value)} style={styles.select}>
          <option value="">Run A</option>
          {runs.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
        </select>
        <select value={runB} onChange={(e) => setRunB(e.target.value)} style={styles.select}>
          <option value="">Run B</option>
          {runs.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
        </select>
        <button onClick={handleCompare} disabled={!runA || !runB || loading} style={styles.btn}>
          {loading ? "Loading…" : "Compare"}
        </button>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {comparison && (
        <>
          {/* Summary table */}
          <div style={{ overflowX: "auto", marginBottom: 20 }}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Metric</th>
                  <th style={{ ...styles.th, color: COLORS_A }}>{runA}</th>
                  <th style={{ ...styles.th, color: COLORS_B }}>{runB}</th>
                  <th style={styles.th}>Winner</th>
                </tr>
              </thead>
              <tbody>
                {METRICS_TO_COMPARE.map(({ key, label, good }) => {
                  const a = comparison[runA]?.[key] ?? 0;
                  const b = comparison[runB]?.[key] ?? 0;
                  const winner =
                    a === b ? "tie"
                    : (good === "high" ? (a > b ? runA : runB)
                                       : (a < b ? runA : runB));
                  return (
                    <tr key={key} style={styles.tr}>
                      <td style={styles.td}>{label}</td>
                      <td style={{ ...styles.td, color: COLORS_A, textAlign: "right" }}>
                        {typeof a === "number" ? a.toFixed(2) : a}
                      </td>
                      <td style={{ ...styles.td, color: COLORS_B, textAlign: "right" }}>
                        {typeof b === "number" ? b.toFixed(2) : b}
                      </td>
                      <td style={{ ...styles.td, textAlign: "center",
                        color: winner === "tie" ? "#94a3b8" : winner === runA ? COLORS_A : COLORS_B }}>
                        {winner === "tie" ? "—" : winner === runA ? "A ✓" : "B ✓"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Bar chart */}
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 0, right: 10, bottom: 40, left: 10 }}>
              <XAxis dataKey="metric" tick={{ fill: "#64748b", fontSize: 10 }}
                     angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#161b27", border: "1px solid #2d3748", borderRadius: 6 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Bar dataKey={runA} fill={COLORS_A} radius={[3, 3, 0, 0]} />
              <Bar dataKey={runB} fill={COLORS_B} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}

      {!comparison && !loading && runs.length === 0 && (
        <div style={styles.empty}>No saved runs found. Run a simulation with --record first.</div>
      )}
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
  selectors: { display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" },
  select: {
    background: "#0f1117",
    border: "1px solid #2d3748",
    color: "#e2e8f0",
    padding: "6px 10px",
    borderRadius: 6,
    fontSize: 13,
    flex: 1,
    minWidth: 120,
  },
  btn: {
    background: "#3b82f6",
    border: "none",
    color: "#fff",
    padding: "6px 18px",
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  table: { width: "100%", borderCollapse: "collapse", marginBottom: 4 },
  th: {
    borderBottom: "1px solid #2d3748",
    padding: "6px 10px",
    fontSize: 11,
    fontWeight: 600,
    color: "#94a3b8",
    textAlign: "left",
    textTransform: "uppercase",
  },
  tr: { borderBottom: "1px solid #1e2433" },
  td: { padding: "5px 10px", fontSize: 12, color: "#e2e8f0" },
  error: { color: "#ef4444", fontSize: 12, marginBottom: 10 },
  empty: { color: "#475569", fontSize: 13, textAlign: "center", padding: "20px 0" },
};
