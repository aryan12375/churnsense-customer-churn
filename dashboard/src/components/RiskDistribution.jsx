/**
 * RiskDistribution.jsx
 * Donut chart showing distribution of customers across risk tiers.
 * Uses Recharts PieChart with custom label + animated center stat.
 */

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

const TIER_CONFIG = {
  critical: { color: "#FF6B6B", label: "Critical",  sublabel: "≥ 80%"  },
  high:     { color: "#FF9F43", label: "High",      sublabel: "65–79%" },
  medium:   { color: "#FAD02C", label: "Medium",    sublabel: "50–64%" },
};

export default function RiskDistribution({ customers }) {
  const counts = { critical: 0, high: 0, medium: 0 };
  customers.forEach(c => { if (counts[c.risk_tier] !== undefined) counts[c.risk_tier]++; });

  const chartData = Object.entries(TIER_CONFIG).map(([key, cfg]) => ({
    name:  cfg.label,
    value: counts[key],
    color: cfg.color,
    sub:   cfg.sublabel,
  })).filter(d => d.value > 0);

  const total = customers.length;

  return (
    <div style={{
      background: "#111827",
      border: "1px solid #1E2535",
      borderRadius: 12,
      padding: "24px",
    }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#F1F5F9" }}>
          Risk Tier Distribution
        </div>
        <div style={{ fontSize: 12, color: "#64748B", marginTop: 3 }}>
          {total} customers in current report
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
        {/* Donut chart */}
        <div style={{ position: "relative", width: 160, height: 160, flexShrink: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={52}
                outerRadius={72}
                paddingAngle={3}
                dataKey="value"
                strokeWidth={0}
              >
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#1E2535",
                  border: "1px solid #2A3045",
                  borderRadius: 8,
                  fontSize: 13,
                  color: "#E2E8F0",
                }}
                formatter={(val, name) => [`${val} customers`, name]}
              />
            </PieChart>
          </ResponsiveContainer>

          {/* Center stat */}
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
          }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#F1F5F9", lineHeight: 1 }}>
              {total}
            </div>
            <div style={{ fontSize: 10, color: "#64748B", marginTop: 3 }}>
              at-risk
            </div>
          </div>
        </div>

        {/* Legend with horizontal bars */}
        <div style={{ flex: 1 }}>
          {Object.entries(TIER_CONFIG).map(([key, cfg]) => {
            const count = counts[key];
            const pct   = total > 0 ? (count / total * 100) : 0;

            return (
              <div key={key} style={{ marginBottom: 14 }}>
                <div style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 5,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                      width: 8, height: 8,
                      borderRadius: "50%",
                      background: cfg.color,
                    }} />
                    <span style={{ fontSize: 13, color: "#CBD5E1", fontWeight: 500 }}>
                      {cfg.label}
                    </span>
                    <span style={{ fontSize: 11, color: "#475569" }}>
                      {cfg.sublabel}
                    </span>
                  </div>
                  <span style={{ fontSize: 13, fontWeight: 600, color: cfg.color }}>
                    {count}
                  </span>
                </div>

                {/* Progress bar */}
                <div style={{
                  height: 4,
                  background: "#1E2535",
                  borderRadius: 2,
                  overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%",
                    width: `${pct}%`,
                    background: cfg.color,
                    borderRadius: 2,
                    transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
