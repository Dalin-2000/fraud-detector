export interface PredictionResult {
  is_fraud: boolean;
  fraud_probability: number;
  message: string;
}
