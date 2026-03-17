/**
 * ChurnSense Dashboard — App.jsx
 * Root component: layout, state, and data loading.
 */

import { useState, useEffect, useCallback } from "react";
import MetricCards     from "./components/MetricCards";
import RiskTable       from "./components/RiskTable";
import RiskDistribution from "./components/RiskDistribution";
import ShapChart       from "./components/ShapChart";
import CustomerModal   from "./components/CustomerModal";
import { mockData }    from "./utils/mockData";

export default function App() {
  const [data, setData]               = useState(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [riskFilter, setRiskFilter]   = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey]         = useState("churn_prob");
  const [sortDir, setSortDir]         = useState("desc");

  // Load report data — tries /reports/high_risk_customers.json, falls back to mock
  useEffect(() => {
    const loadData = async () => {
      try {
        const res = await fetch("/reports/high_risk_customers.json");
        if (!res.ok) throw new Error("Using mock data");
        const json = await res.json();
        setData(json);
      } catch {
        // Development fallback — rich mock dataset
        setData(mockData);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const handleSort = useCallback((key) => {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }, [sortKey]);

  const handleExportCSV = useCallback(() => {
    if (!data) return;
    const headers = ["Rank", "Customer ID", "Churn Probability", "Risk Tier", "Top Driver"];
    const rows = data.customers.map(c => [
      c.rank,
      c.customer_id,
      `${(c.churn_prob * 100).toFixed(1)}%`,
      c.risk_tier,
      c.top_drivers?.[0]?.feature ?? "—",
    ]);
    const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
    const blob = new URL("data:text/csv," + encodeURIComponent(csv));
    const a = document.createElement("a");
    a.href = blob;
    a.download = "churnsense_risk_report.csv";
    a.click();
  }, [data]);

  // ── Derived / filtered customers ────────────────────────────────────────────
  const customers = data?.customers ?? [];

  const filtered = customers
    .filter(c => {
      if (riskFilter !== "all" && c.risk_tier !== riskFilter) return false;
      if (searchQuery && !c.customer_id.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      return sortDir === "desc" ? vb - va : va - vb;
    });

  const metrics = data ? {
    total:    data.total_customers_scored,
    critical: customers.filter(c => c.risk_tier === "critical").length,
    high:     customers.filter(c => c.risk_tier === "high").length,
    medium:   customers.filter(c => c.risk_tier === "medium").length,
    avgProb:  customers.length
      ? (customers.reduce((s, c) => s + c.churn_prob, 0) / customers.length)
      : 0,
  } : null;

  // ── Render ───────────────────────────────────────────────────────────────────
  if (loading) return <LoadingScreen />;
  if (error)   return <ErrorScreen message={error} />;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0A0D14",
      color: "#E2E8F0",
      fontFamily: "'DM Sans', 'Inter', system-ui, sans-serif",
    }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header style={{
        padding: "20px 32px",
        borderBottom: "1px solid #1E2535",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(10,13,20,0.95)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36,
            background: "linear-gradient(135deg, #6C63FF 0%, #00E5A0 100%)",
            borderRadius: 10,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: 700, fontSize: 16, color: "#fff",
          }}>C</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.3px" }}>
              ChurnSense
            </div>
            <div style={{ fontSize: 11, color: "#64748B", marginTop: 1 }}>
              Customer retention intelligence
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <StatusPill label={`${data?.total_customers_scored?.toLocaleString() ?? 0} scored`} color="#6C63FF" />
          <StatusPill label={`Updated ${formatTime(data?.generated_at)}`} color="#00E5A0" />
          <button
            onClick={handleExportCSV}
            style={{
              padding: "8px 16px",
              background: "transparent",
              border: "1px solid #2A3045",
              borderRadius: 8,
              color: "#94A3B8",
              fontSize: 13,
              cursor: "pointer",
              display: "flex", alignItems: "center", gap: 6,
            }}
          >
            ↓ Export CSV
          </button>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────────────────────── */}
      <main style={{ padding: "28px 32px", maxWidth: 1440, margin: "0 auto" }}>

        {/* Metric cards row */}
        <MetricCards metrics={metrics} />

        {/* Charts row */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.6fr",
          gap: 20,
          marginTop: 24,
        }}>
          <RiskDistribution customers={customers} />
          <ShapChart customers={customers} />
        </div>

        {/* Risk table */}
        <div style={{ marginTop: 24 }}>
          <RiskTable
            customers={filtered}
            total={customers.length}
            riskFilter={riskFilter}
            onRiskFilter={setRiskFilter}
            searchQuery={searchQuery}
            onSearch={setSearchQuery}
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={handleSort}
            onSelectCustomer={setSelectedCustomer}
          />
        </div>

      </main>

      {/* Customer detail modal */}
      {selectedCustomer && (
        <CustomerModal
          customer={selectedCustomer}
          onClose={() => setSelectedCustomer(null)}
        />
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatusPill({ label, color }) {
  return (
    <div style={{
      padding: "5px 12px",
      background: color + "18",
      border: `1px solid ${color}40`,
      borderRadius: 20,
      fontSize: 12,
      color: color,
      fontWeight: 500,
    }}>
      {label}
    </div>
  );
}

function LoadingScreen() {
  return (
    <div style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      background: "#0A0D14", color: "#6C63FF", gap: 16,
    }}>
      <div style={{
        width: 48, height: 48,
        border: "3px solid #1E2535",
        borderTop: "3px solid #6C63FF",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
      }} />
      <div style={{ color: "#64748B", fontSize: 14 }}>Loading customer risk data...</div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function ErrorScreen({ message }) {
  return (
    <div style={{
      minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
      background: "#0A0D14", color: "#FF6B6B",
    }}>
      Error: {message}
    </div>
  );
}

function formatTime(isoString) {
  if (!isoString) return "now";
  try {
    return new Date(isoString).toLocaleDateString("en-IN", {
      day: "numeric", month: "short", year: "numeric",
    });
  } catch {
    return "recently";
  }
}
