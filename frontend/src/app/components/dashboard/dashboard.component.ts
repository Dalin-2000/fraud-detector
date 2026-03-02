import { Component, ElementRef, ViewChild } from '@angular/core';
import { FraudService } from '../../services/fraud.service';
import { TransactionInput } from '../../models/transaction.model';
import { PredictionResult } from '../../models/prediction.model';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  @ViewChild('resultPanel') resultPanel!: ElementRef<HTMLElement>;

  result: PredictionResult | null = null;
  loading = false;
  error: string | null = null;

  constructor(private fraudService: FraudService) {}

  onTransactionSubmit(transaction: TransactionInput): void {
    this.loading = true;
    this.error = null;
    this.result = null;

    // On mobile (stacked layout) scroll result panel into view immediately
    if (window.innerWidth <= 860) {
      setTimeout(() => {
        this.resultPanel?.nativeElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    }

    this.fraudService.predict(transaction).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to get prediction. Make sure the backend is running.';
        this.loading = false;
        console.error(err);
      },
    });
  }
}
