import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { TransactionInput } from '../models/transaction.model';
import { PredictionResult } from '../models/prediction.model';

@Injectable({
  providedIn: 'root',
})
export class FraudService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  predict(transaction: TransactionInput): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.apiUrl}/api/predict`, transaction);
  }
}
