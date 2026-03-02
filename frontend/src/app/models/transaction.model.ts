export interface TransactionInput {
  // Numerical
  amount_ngn: number;
  time_since_last_transaction: number;
  spending_deviation_score: number;
  velocity_score: number;
  geo_anomaly_score: number;

  // Categorical
  transaction_type: string;
  merchant_category: string;
  location: string;
  device_used: string;
  payment_channel: string;
  sender_persona: string;

  // Boolean flags
  bvn_linked: boolean;
  new_device_transaction: boolean;

  // Optional
  timestamp?: string;
}
