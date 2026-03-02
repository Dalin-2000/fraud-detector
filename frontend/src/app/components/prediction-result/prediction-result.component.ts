import { Component, Input } from '@angular/core';
import { PredictionResult } from '../../models/prediction.model';

@Component({
  selector: 'app-prediction-result',
  templateUrl: './prediction-result.component.html',
  styleUrls: ['./prediction-result.component.scss'],
})
export class PredictionResultComponent {
  @Input() result: PredictionResult | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;
}
