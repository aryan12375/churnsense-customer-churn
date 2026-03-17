/**
 * CustomerModal.jsx
 * Detailed per-customer risk breakdown with SHAP force plot visualization.
 * Shows churn probability, all top drivers with direction and magnitude.
 */

import { useEffect } from "react";

const RISK_CONFIG = {
  critical: { color: "#FF6B6B", label: "Critical Risk" },
  high:     { color: "#FF9F43", label: "High Risk"     },
  medium:   { color: "#FAD02C", label: "Medium Risk"   },
};

export default function CustomerModal({ customer, onClose }) {
  const cfg  = RISK_CONFIG[customer.risk_tier] ?? RISK_CONFIG.medium;
  const prob = (customer.churn_prob * 100).toFixed(1);

  // Close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Separate positive (churn) and negative (retain) drivers
  const pushDrivers = (customer.top_drivers ?? []).filter(d => d.direction === "increases_churn");
  const pullDrivers = (customer.top_drivers ?? []).filter(d => d.direction === "reduces_churn");
  const allDrivers  = customer.top_drivers ?? [];

  const maxAbsShap = allDrivers.length > 0
    ? Math.max(...allDrivers.map(d => Math.abs(d.shap_value)))
    : 1;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.65)",
        backdropFilter: "blur(4px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: "#111827",
          border: "1px solid #1E2535",
          borderRadius: 16,
          padding: 32,
          width: "100%",
          maxWidth: 680,
          maxHeight: "90vh",
          overflowY: "auto",
          position: "relative",
        }}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 20, right: 20,
            width: 32, height: 32,
            background: "#1E2535",
            border: "none",
            borderRadius: 8,
            color: "#94A3B8",
            cursor: "pointer",
            fontSize: 16,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          ×
        </button>

        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
            <div style={{
              fontSize: 11, color: "#64748B", textTransform: "uppercase",
              letterSpacing: "0.08em", fontWeight: 500,
            }}>
              Customer
            </div>
            <div style={{
              padding: "2px 10px",
              background: cfg.color + "18",
              border: `1px solid ${cfg.color}40`,
              borderRadius: 20,
              fontSize: 11, color: cfg.color, fontWeight: 600,
            }}>
              {cfg.label}
            </div>
          </div>

          <div style={{
            fontSize: 24, fontWeight: 700, color: "#F1F5F9",
            letterSpacing: "-0.3px", fontFamily: "monospace",
            marginBottom: 4,
          }}>
            {customer.customer_id}
          </div>

          {customer.actual_churn !== null && (
            <div style={{ fontSize: 13, color: "#475569" }}>
              Actual outcome:{" "}
              <span style={{
                color: customer.actual_churn ? "#FF6B6B" : "#00E5A0",
                fontWeight: 600,
              }}>
                {customer.actual_churn ? "Churned ✕" : "Retained ✓"}
              </span>
            </div>
          )}
        </div>

        {/* Probability gauge */}
        <div style={{
          background: "#0D1320",
          borderRadius: 12,
          padding: "20px 24px",
          marginBottom: 24,
        }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            marginBottom: 12,
          }}>
            <div>
              <div style={{ fontSize: 12, color: "#64748B", marginBottom: 4 }}>
                Churn Probability
              </div>
              <div style={{
                fontSize: 40, fontWeight: 800, color: cfg.color,
                letterSpacing: "-1px", lineHeight: 1,
              }}>
                {prob}%
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 12, color: "#64748B" }}>Rank</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#94A3B8" }}>
                #{customer.rank}
              </div>
            </div>
          </div>

          {/* Full-width probability bar */}
          <div style={{
            height: 10,
            background: "#1E2535",
            borderRadius: 5,
            overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width: `${prob}%`,
              background: `linear-gradient(90deg, ${cfg.color}88 0%, ${cfg.color} 100%)`,
              borderRadius: 5,
              transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
            }} />
          </div>
        </div>

        {/* SHAP force-plot style waterfall */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#F1F5F9", marginBottom: 16 }}>
            Why is this customer at risk?
          </div>
          <div style={{ fontSize: 12, color: "#64748B", marginBottom: 14 }}>
            Bars show each feature's contribution toward or away from churn.
          </div>

          {allDrivers.map((driver, idx) => {
            const isChurn = driver.direction === "increases_churn";
            const color   = isChurn ? "#FF6B6B" : "#00E5A0";
            const barPct  = Math.abs(driver.shap_value / maxAbsShap) * 100;

            return (
              <div key={idx} style={{ marginBottom: 12 }}>
                <div style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 5,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: color, flexShrink: 0,
                    }} />
                    <span style={{ fontSize: 13, color: "#CBD5E1" }}>
                      {cleanName(driver.feature)}
                    </span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace" }}>
                      val={driver.feature_value.toFixed(2)}
                    </span>
                    <span style={{
                      fontSize: 12, fontWeight: 700, color: color,
                      fontFamily: "monospace", minWidth: 60, textAlign: "right",
                    }}>
                      {driver.shap_value > 0 ? "+" : ""}{driver.shap_value.toFixed(4)}
                    </span>
                  </div>
                </div>

                {/* Dual-direction bar */}
                <div style={{
                  height: 8,
                  background: "#1E2535",
                  borderRadius: 4,
                  overflow: "hidden",
                  position: "relative",
                }}>
                  {isChurn ? (
                    <div style={{
                      height: "100%",
                      width: `${barPct}%`,
                      background: color,
                      borderRadius: 4,
                    }} />
                  ) : (
                    <div style={{
                      height: "100%",
                      width: `${barPct}%`,
                      background: color,
                      borderRadius: 4,
                      float: "right",
                    }} />
                  )}
                </div>

                <div style={{ fontSize: 11, color: "#475569", marginTop: 3 }}>
                  {isChurn
                    ? `↑ Increases churn probability`
                    : `↓ Reduces churn probability`}
                </div>
              </div>
            );
          })}
        </div>

        {/* Recommended action */}
        <div style={{
          background: "#0D1320",
          border: "1px solid #1E2535",
          borderRadius: 12,
          padding: "16px 20px",
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#F1F5F9", marginBottom: 8 }}>
            Suggested Retention Action
          </div>
          <div style={{ fontSize: 13, color: "#94A3B8", lineHeight: 1.6 }}>
            {getSuggestion(customer)}
          </div>
        </div>
      </div>
    </div>
  );
}

function cleanName(name) {
  return name.replace(/^cat_|^num_/, "").replace(/_/g, " ");
}

function getSuggestion(customer) {
  const topDrivers = customer.top_drivers ?? [];
  const topFeature = topDrivers[0]?.feature ?? "";

  if (topFeature.toLowerCase().includes("contract")) {
    return "Offer a discounted annual or two-year contract upgrade. Month-to-month customers have significantly higher churn risk — converting them to longer contracts is the single most effective retention lever.";
  }
  if (topFeature.toLowerCase().includes("tenure") || topFeature.toLowerCase().includes("group")) {
    return "This is a new customer (< 12 months). Trigger an onboarding check-in call and offer a loyalty discount at the 6-month mark. Early-tenure intervention reduces long-term churn significantly.";
  }
  if (topFeature.toLowerCase().includes("charge")) {
    return "Customer's monthly charges are high relative to perceived value. Consider offering a service bundle discount or reviewing their plan for cost optimisation.";
  }
  if (topFeature.toLowerCase().includes("tech") || topFeature.toLowerCase().includes("support")) {
    return "Customer doesn't have tech support. A complimentary 3-month tech support trial could significantly improve satisfaction and reduce churn likelihood.";
  }
  return `Based on ${topDrivers.length} identified risk factors, a personalised outreach from the retention team is recommended. Focus on understanding the customer's primary pain point before offering an incentive.`;
}
