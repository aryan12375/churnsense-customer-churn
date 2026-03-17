/**
 * RiskTable.jsx
 * Main customer risk table with filtering, sorting, search, and row click.
 */

const RISK_CONFIG = {
  critical: { color: "#FF6B6B", bg: "#FF6B6B18", label: "Critical" },
  high:     { color: "#FF9F43", bg: "#FF9F4318", label: "High"     },
  medium:   { color: "#FAD02C", bg: "#FAD02C18", label: "Medium"   },
};

export default function RiskTable({
  customers, total,
  riskFilter, onRiskFilter,
  searchQuery, onSearch,
  sortKey, sortDir, onSort,
  onSelectCustomer,
}) {
  return (
    <div style={{
      background: "#111827",
      border: "1px solid #1E2535",
      borderRadius: 12,
      overflow: "hidden",
    }}>
      {/* Table header bar */}
      <div style={{
        padding: "18px 24px",
        borderBottom: "1px solid #1E2535",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#F1F5F9" }}>
            At-Risk Customers
          </div>
          <div style={{ fontSize: 12, color: "#64748B", marginTop: 2 }}>
            Showing {customers.length} of {total} customers
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* Search */}
          <input
            value={searchQuery}
            onChange={e => onSearch(e.target.value)}
            placeholder="Search customer ID..."
            style={{
              padding: "7px 12px",
              background: "#1E2535",
              border: "1px solid #2A3045",
              borderRadius: 8,
              color: "#E2E8F0",
              fontSize: 13,
              outline: "none",
              width: 200,
            }}
          />

          {/* Risk filter pills */}
          <div style={{ display: "flex", gap: 6 }}>
            {["all", "critical", "high", "medium"].map(tier => {
              const cfg = RISK_CONFIG[tier] ?? { color: "#6C63FF", bg: "#6C63FF18", label: "All" };
              const active = riskFilter === tier;
              return (
                <button
                  key={tier}
                  onClick={() => onRiskFilter(tier)}
                  style={{
                    padding: "5px 12px",
                    borderRadius: 20,
                    border: `1px solid ${active ? cfg.color : "#2A3045"}`,
                    background: active ? cfg.bg : "transparent",
                    color: active ? cfg.color : "#64748B",
                    fontSize: 12,
                    fontWeight: 500,
                    cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                >
                  {tier === "all" ? "All" : RISK_CONFIG[tier].label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 13,
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #1E2535" }}>
              {[
                { key: "rank",       label: "#",           width: 50  },
                { key: "customer_id",label: "Customer ID", width: 140 },
                { key: "churn_prob", label: "Churn Prob",  width: 140 },
                { key: "risk_tier",  label: "Risk Tier",   width: 120 },
                { key: "drivers",    label: "Top Driver",  width: null },
                { key: "action",     label: "",            width: 80  },
              ].map(col => (
                <th
                  key={col.key}
                  onClick={col.key !== "drivers" && col.key !== "action" ? () => onSort(col.key) : undefined}
                  style={{
                    padding: "12px 20px",
                    textAlign: "left",
                    color: "#64748B",
                    fontWeight: 500,
                    letterSpacing: "0.03em",
                    fontSize: 11,
                    textTransform: "uppercase",
                    cursor: (col.key !== "drivers" && col.key !== "action") ? "pointer" : "default",
                    width: col.width ?? undefined,
                    whiteSpace: "nowrap",
                    userSelect: "none",
                  }}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span style={{ marginLeft: 4, color: "#6C63FF" }}>
                      {sortDir === "desc" ? "↓" : "↑"}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {customers.length === 0 ? (
              <tr>
                <td colSpan={6} style={{
                  padding: "48px 20px",
                  textAlign: "center",
                  color: "#475569",
                  fontSize: 14,
                }}>
                  No customers match the current filter.
                </td>
              </tr>
            ) : (
              customers.map((customer, idx) => (
                <TableRow
                  key={customer.customer_id}
                  customer={customer}
                  isEven={idx % 2 === 0}
                  onClick={() => onSelectCustomer(customer)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TableRow({ customer, isEven, onClick }) {
  const cfg = RISK_CONFIG[customer.risk_tier] ?? RISK_CONFIG.medium;
  const topDriver = customer.top_drivers?.[0];
  const prob = (customer.churn_prob * 100).toFixed(1);

  return (
    <tr
      onClick={onClick}
      style={{
        background: isEven ? "transparent" : "#0D1320",
        cursor: "pointer",
        transition: "background 0.1s",
        borderBottom: "1px solid #1A2035",
      }}
      onMouseEnter={e => e.currentTarget.style.background = "#162030"}
      onMouseLeave={e => e.currentTarget.style.background = isEven ? "transparent" : "#0D1320"}
    >
      {/* Rank */}
      <td style={{ padding: "14px 20px", color: "#475569", fontWeight: 500 }}>
        {customer.rank}
      </td>

      {/* Customer ID */}
      <td style={{ padding: "14px 20px", color: "#CBD5E1", fontFamily: "monospace" }}>
        {customer.customer_id}
      </td>

      {/* Churn probability */}
      <td style={{ padding: "14px 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 80, height: 6,
            background: "#1E2535",
            borderRadius: 3,
            overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width: `${prob}%`,
              background: cfg.color,
              borderRadius: 3,
            }} />
          </div>
          <span style={{ color: cfg.color, fontWeight: 700, fontSize: 14 }}>
            {prob}%
          </span>
        </div>
      </td>

      {/* Risk tier badge */}
      <td style={{ padding: "14px 20px" }}>
        <span style={{
          padding: "3px 10px",
          borderRadius: 20,
          fontSize: 11,
          fontWeight: 600,
          background: cfg.bg,
          color: cfg.color,
          border: `1px solid ${cfg.color}40`,
        }}>
          {cfg.label}
        </span>
      </td>

      {/* Top driver */}
      <td style={{ padding: "14px 20px", color: "#94A3B8" }}>
        {topDriver ? (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: topDriver.direction === "increases_churn" ? "#FF6B6B" : "#00E5A0",
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 13 }}>
              {cleanName(topDriver.feature)}
            </span>
            <span style={{
              fontSize: 11, color: "#475569",
              fontFamily: "monospace",
            }}>
              {topDriver.shap_value > 0 ? "+" : ""}{topDriver.shap_value.toFixed(3)}
            </span>
          </div>
        ) : "—"}
      </td>

      {/* Action */}
      <td style={{ padding: "14px 20px" }}>
        <span style={{
          fontSize: 12,
          color: "#6C63FF",
          fontWeight: 500,
        }}>
          View →
        </span>
      </td>
    </tr>
  );
}

function cleanName(name) {
  return name.replace(/^cat_|^num_/, "").replace(/_/g, " ");
}
