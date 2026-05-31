import React from "react";

const STATE_COLORS = {
  cruise:       { bg: "#22c55e", label: "CRUISE" },
  follow:       { bg: "#3b82f6", label: "FOLLOW" },
  overtake:     { bg: "#a855f7", label: "OVERTAKE" },
  yield:        { bg: "#f59e0b", label: "YIELD" },
  emergency:    { bg: "#ef4444", label: "EMERGENCY" },
  intersection: { bg: "#06b6d4", label: "INTERSECTION" },
};

const ALL_STATES = Object.keys(STATE_COLORS);

export default function BehaviorStateIndicator({ state }) {
  const active = (state ?? "cruise").toLowerCase();

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Behavior State</h3>
      <div style={styles.grid}>
        {ALL_STATES.map((s) => {
          const { bg, label } = STATE_COLORS[s];
          const isActive = s === active;
          return (
            <div
              key={s}
              style={{
                ...styles.chip,
                background: isActive ? bg : "#1e2433",
                border: `2px solid ${isActive ? bg : "#2d3748"}`,
                color: isActive ? "#fff" : "#718096",
                transform: isActive ? "scale(1.05)" : "scale(1)",
                boxShadow: isActive ? `0 0 12px ${bg}88` : "none",
              }}
            >
              {label}
            </div>
          );
        })}
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
    marginBottom: 12,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 8,
  },
  chip: {
    padding: "8px 6px",
    borderRadius: 6,
    textAlign: "center",
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.06em",
    transition: "all 0.2s ease",
    cursor: "default",
  },
};
