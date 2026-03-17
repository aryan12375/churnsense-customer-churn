/**
 * utils/mockData.js
 * Realistic mock dataset for dashboard development.
 * Mirrors the schema produced by src/shap_explain.py
 */

const DRIVER_POOL = [
  { feature: "cat_Contract_Month-to-month", positive_shap: 0.82 },
  { feature: "cat_tenure_group_0-12",       positive_shap: 0.64 },
  { feature: "cat_InternetService_Fiber_optic", positive_shap: 0.55 },
  { feature: "num_TotalCharges",            positive_shap: -0.48 },
  { feature: "cat_TechSupport_No",          positive_shap: 0.42 },
  { feature: "cat_OnlineSecurity_No",       positive_shap: 0.38 },
  { feature: "cat_PaymentMethod_Electronic_check", positive_shap: 0.35 },
  { feature: "cat_OnlineBackup_No",         positive_shap: 0.29 },
  { feature: "cat_Contract_Two_year",       positive_shap: -0.44 },
  { feature: "num_MonthlyCharges",          positive_shap: 0.22 },
  { feature: "cat_PaperlessBilling_Yes",    positive_shap: 0.18 },
  { feature: "cat_Dependents_No",           positive_shap: 0.15 },
  { feature: "remainder_SeniorCitizen",     positive_shap: 0.12 },
];

function randomDrivers(baseProb) {
  const shuffled = [...DRIVER_POOL].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, 5).map(d => {
    const shap  = d.positive_shap * (0.7 + Math.random() * 0.6);
    const noise = (Math.random() - 0.5) * 0.2;
    return {
      feature:        d.feature,
      shap_value:     Math.round((shap + noise) * 10000) / 10000,
      feature_value:  Math.round(Math.random() * 10000) / 10000,
      direction:      shap > 0 ? "increases_churn" : "reduces_churn",
    };
  }).sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
}

function makeTier(prob) {
  if (prob >= 0.80) return "critical";
  if (prob >= 0.65) return "high";
  return "medium";
}

// Generate 25 mock customers
const customers = Array.from({ length: 25 }, (_, i) => {
  const prob = Math.round((0.55 + Math.random() * 0.44) * 10000) / 10000;
  return {
    rank:          i + 1,
    customer_id:   `CUST_${String(i + 1).padStart(4, "0")}`,
    churn_prob:    prob,
    actual_churn:  Math.random() < prob ? 1 : 0,
    risk_tier:     makeTier(prob),
    top_drivers:   randomDrivers(prob),
  };
}).sort((a, b) => b.churn_prob - a.churn_prob)
  .map((c, i) => ({ ...c, rank: i + 1 }));

export const mockData = {
  generated_at:           new Date().toISOString(),
  total_customers_scored: 1406,
  high_risk_threshold:    0.65,
  customers,
};
