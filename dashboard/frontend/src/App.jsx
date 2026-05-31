import React, { useState } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import SimulationView from "./components/SimulationView";
import MetricsPanel from "./components/MetricsPanel";
import BehaviorStateIndicator from "./components/BehaviorStateIndicator";
import RunComparison from "./components/RunComparison";
import PerceptionDebug from "./components/PerceptionDebug";

const TABS = ["Live", "Comparison"];

export default function App() {
  const { state: live, connected } = useWebSocket();
  const [tab, setTab] = useState("Live");

  const behaviorState = live?.behavior_state ?? null;
  const metrics       = live?.metrics ?? null;

  return (
    <div style={styles.root}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoAccent}>Seyir</span>
          <span style={styles.logoSub}> Autonomous Driving System</span>
        </div>
        <div style={styles.headerRight}>
          <div style={{ ...styles.dot, background: connected ? "#22c55e" : "#ef4444" }} />
          <span style={{ ...styles.connStatus, color: connected ? "#22c55e" : "#ef4444" }}>
            {connected ? "LIVE" : "DISCONNECTED"}
          </span>
          <nav style={styles.nav}>
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{ ...styles.navBtn, ...(tab === t ? styles.navBtnActive : {}) }}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main style={styles.main}>
        {tab === "Live" && (
          <div style={styles.liveLayout}>
            {/* Left column */}
            <div style={styles.leftCol}>
              <SimulationView live={live} />
              <div style={{ marginTop: 16 }}>
                <PerceptionDebug live={live} />
              </div>
            </div>
            {/* Right column */}
            <div style={styles.rightCol}>
              <MetricsPanel live={live} metrics={metrics} />
              <div style={{ marginTop: 16 }}>
                <BehaviorStateIndicator state={behaviorState} />
              </div>
              {live && (
                <div style={{ marginTop: 16 }}>
                  <RawStatePanel live={live} />
                </div>
              )}
            </div>
          </div>
        )}
        {tab === "Comparison" && (
          <div style={{ maxWidth: 900, margin: "0 auto" }}>
            <RunComparison />
          </div>
        )}
      </main>
    </div>
  );
}

function RawStatePanel({ live }) {
  const ego = live?.ego ?? {};
  return (
    <div style={rawStyles.container}>
      <h3 style={rawStyles.title}>Ego Telemetry</h3>
      <div style={rawStyles.grid}>
        <Stat label="X" value={ego.x?.toFixed(1) ?? "--"} unit="m" />
        <Stat label="Y" value={ego.y?.toFixed(1) ?? "--"} unit="m" />
        <Stat label="Speed" value={ego.speed_kmh?.toFixed(1) ?? "--"} unit="km/h" />
        <Stat label="Heading" value={ego.heading?.toFixed(1) ?? "--"} unit="°" />
        <Stat label="Detections" value={live?.detections?.length ?? 0} unit="" />
        <Stat label="Timestamp" value={(live?.timestamp ?? 0).toFixed(2)} unit="s" />
      </div>
    </div>
  );
}

function Stat({ label, value, unit }) {
  return (
    <div style={rawStyles.stat}>
      <div style={rawStyles.statLabel}>{label}</div>
      <div style={rawStyles.statValue}>{value}<span style={rawStyles.unit}> {unit}</span></div>
    </div>
  );
}

const styles = {
  root: { minHeight: "100vh", background: "#0f1117", color: "#e2e8f0" },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 24px",
    borderBottom: "1px solid #1e2433",
    background: "#0a0d14",
    position: "sticky",
    top: 0,
    zIndex: 100,
  },
  logo: { fontSize: 18, fontWeight: 700, letterSpacing: "-0.01em" },
  logoAccent: { color: "#3b82f6" },
  logoSub: { color: "#475569", fontSize: 14, fontWeight: 400 },
  headerRight: { display: "flex", alignItems: "center", gap: 12 },
  dot: { width: 8, height: 8, borderRadius: "50%" },
  connStatus: { fontSize: 11, fontWeight: 700, letterSpacing: "0.08em" },
  nav: { display: "flex", gap: 4, marginLeft: 16 },
  navBtn: {
    background: "transparent",
    border: "1px solid #2d3748",
    color: "#64748b",
    padding: "5px 14px",
    borderRadius: 6,
    fontSize: 13,
    cursor: "pointer",
    fontWeight: 500,
    transition: "all 0.15s",
  },
  navBtnActive: {
    background: "#3b82f622",
    border: "1px solid #3b82f6",
    color: "#93c5fd",
  },
  main: { padding: "20px 24px" },
  liveLayout: { display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, alignItems: "start" },
  leftCol: {},
  rightCol: {},
};

const rawStyles = {
  container: {
    background: "#161b27",
    border: "1px solid #2d3748",
    borderRadius: 10,
    padding: "14px 18px",
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    marginBottom: 12,
  },
  grid: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 },
  stat: {},
  statLabel: { fontSize: 10, color: "#475569", fontWeight: 600, textTransform: "uppercase" },
  statValue: { fontSize: 15, fontWeight: 700, color: "#e2e8f0", fontVariantNumeric: "tabular-nums" },
  unit: { fontSize: 11, color: "#64748b", fontWeight: 400 },
};
