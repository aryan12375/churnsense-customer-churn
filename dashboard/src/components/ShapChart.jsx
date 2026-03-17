/**
 * ShapChart.jsx
 * Aggregated SHAP feature importance bar chart.
 * Shows the top-10 drivers most frequently appearing across all at-risk customers.
 */

import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

export default function ShapChart({ customers }) {
  const [view, setView] = useState("frequency"); // "frequency" | "avg_shap"

  // Aggregate drivers across all customers
  const featureStats = {};

  customers.forEach(c => {
    (c.top_drivers ?? []).forEach(d => {
      if (!featureStats[d.feature]) {
        featureStats[d.feature] = { count: 0, totalShap: 0, positive: 0 };
      }
      featureStats[d.feature].count++;
      featureStats[d.feature].totalShap += Math.abs(d.shap_value);
      if (d.direction === "increases_churn") featureStats[d.feature].positive++;
    });
  });

  const chartData = Object.entries(featureStats)
    .map(([name, stats]) => ({
      name: cleanName(name),
      frequency: stats.count,
      avg_shap:  parseFloat((stats.totalShap / stats.count).toFixed(4)),
      pct_positive: stats.count > 0 ? stats.positive / stats.count : 0,
    }))
    .sort((a, b) => b[view] - a[view])
    .slice(0, 10)
    .reverse(); // recharts horizontal bars render bottom-to-top

  const maxVal = chartData.length > 0 ? Math.max(...chartData.map(d => d[view])) : 1;

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div style={{
        background: "#1E2535",
        border: "1px solid #2A3045",
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 13,
        color: "#E2E8F0",
        minWidth: 200,
      }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>{label}</div>
        <div style={{ color: "#94A3B8" }}>
          Appears in: <span style={{ color: "#6C63FF", fontWeight: 600 }}>{d.frequency} customers</span>
        </div>
        <div style={{ color: "#94A3B8" }}>
          Avg |SHAP|: <span style={{ color: "#00E5A0", fontWeight: 600 }}>{d.avg_shap.toFixed(4)}</span>
        </div>
        <div style={{ color: "#94A3B8" }}>
          Churn driver: <span style={{ color: "#FF9F43", fontWeight: 600 }}>
            {(d.pct_positive * 100).toFixed(0)}% of cases
          </span>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      background: "#111827",
      border: "1px solid #1E2535",
      borderRadius: 12,
      padding: "24px",
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        marginBottom: 20,
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#F1F5F9" }}>
            Top Churn Drivers
          </div>
          <div style={{ fontSize: 12, color: "#64748B", marginTop: 3 }}>
            Aggregated SHAP analysis across all at-risk customers
          </div>
        </div>

        {/* Toggle */}
        <div style={{
          display: "flex",
          background: "#1E2535",
          borderRadius: 8,
          padding: 3,
          gap: 2,
        }}>
          {[
            { key: "frequency", label: "Frequency" },
            { key: "avg_shap",  label: "SHAP Impact" },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setView(tab.key)}
              style={{
                padding: "5px 12px",
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
                fontSize: 12,
                fontWeight: 500,
                transition: "all 0.15s",
                background: view === tab.key ? "#6C63FF" : "transparent",
                color:      view === tab.key ? "#fff"    : "#64748B",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ left: 0, right: 40, top: 0, bottom: 0 }}
        >
          <CartesianGrid
            horizontal={false}
            stroke="#1E2535"
            strokeDasharray="3 3"
          />
          <XAxis
            type="number"
            tick={{ fill: "#475569", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            domain={[0, maxVal * 1.15]}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={170}
            tick={{ fill: "#94A3B8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#1E2535" }} />
          <Bar dataKey={view} radius={[0, 4, 4, 0]} maxBarSize={18}>
            {chartData.map((entry, idx) => (
              <Cell
                key={idx}
                fill={
                  idx === chartData.length - 1  // top feature = champion color
                    ? "#00E5A0"
                    : idx >= chartData.length - 4
                    ? "#6C63FF"
                    : "#3D3A7A"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function cleanName(name) {
  return name
    .replace(/^cat_|^num_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
    .slice(0, 32);
}
