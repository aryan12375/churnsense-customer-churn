/**
 * MetricCards.jsx
 * Top KPI row: 5 cards showing key risk metrics at a glance.
 */

export default function MetricCards({ metrics }) {
  if (!metrics) return null;

  const cards = [
    {
      label: "Customers Scored",
      value: metrics.total?.toLocaleString() ?? "—",
      sub: "this run",
      color: "#6C63FF",
      icon: "◉",
    },
    {
      label: "Critical Risk",
      value: metrics.critical,
      sub: "≥ 80% churn prob",
      color: "#FF6B6B",
      icon: "▲",
    },
    {
      label: "High Risk",
      value: metrics.high,
      sub: "65–79% churn prob",
      color: "#FF9F43",
      icon: "◆",
    },
    {
      label: "Medium Risk",
      value: metrics.medium,
      sub: "50–64% churn prob",
      color: "#FAD02C",
      icon: "●",
    },
    {
      label: "Avg. Churn Probability",
      value: `${(metrics.avgProb * 100).toFixed(1)}%`,
      sub: "across all at-risk",
      color: "#00E5A0",
      icon: "~",
    },
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(5, 1fr)",
      gap: 16,
    }}>
      {cards.map((card) => (
        <div
          key={card.label}
          style={{
            background: "#111827",
            border: `1px solid ${card.color}25`,
            borderRadius: 12,
            padding: "20px 20px 18px",
            position: "relative",
            overflow: "hidden",
            transition: "border-color 0.2s",
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = card.color + "70"}
          onMouseLeave={e => e.currentTarget.style.borderColor = card.color + "25"}
        >
          {/* Background glow */}
          <div style={{
            position: "absolute",
            top: -20, right: -20,
            width: 80, height: 80,
            background: card.color,
            borderRadius: "50%",
            opacity: 0.06,
            filter: "blur(20px)",
          }} />

          <div style={{
            fontSize: 22,
            color: card.color,
            marginBottom: 4,
            fontWeight: 300,
            lineHeight: 1,
          }}>
            {card.icon}
          </div>

          <div style={{
            fontSize: 28,
            fontWeight: 700,
            color: "#F1F5F9",
            letterSpacing: "-0.5px",
            marginBottom: 4,
          }}>
            {card.value}
          </div>

          <div style={{ fontSize: 13, color: "#94A3B8", fontWeight: 500 }}>
            {card.label}
          </div>
          <div style={{ fontSize: 11, color: "#475569", marginTop: 2 }}>
            {card.sub}
          </div>
        </div>
      ))}
    </div>
  );
}
