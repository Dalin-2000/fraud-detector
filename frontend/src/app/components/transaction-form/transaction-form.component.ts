import { Component, EventEmitter, Output } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { TransactionInput } from '../../models/transaction.model';

@Component({
  selector: 'app-transaction-form',
  templateUrl: './transaction-form.component.html',
  styleUrls: ['./transaction-form.component.scss'],
})
export class TransactionFormComponent {
  @Output() submitted = new EventEmitter<TransactionInput>();

  transactionTypes    = ['Transfer', 'Payment', 'Withdrawal', 'Deposit', 'POS'];
  merchantCategories  = ['Retail', 'Food & Dining', 'Utilities', 'Healthcare', 'Education', 'Travel', 'Entertainment', 'Fuel', 'Other'];
  locations           = ['Lagos', 'Abuja', 'Kano', 'Port Harcourt', 'Enugu', 'Ibadan', 'Kaduna', 'Benin City', 'Owerri', 'Uyo'];
  devicesUsed         = ['Mobile App', 'Web Browser', 'POS Terminal', 'ATM', 'USSD'];
  paymentChannels     = ['Bank Transfer', 'Card Payment', 'Mobile Banking', 'USSD', 'Internet Banking'];
  senderPersonas      = ['Regular', 'High-Value', 'Occasional', 'New Customer'];

  form: FormGroup;

  // Pill picker options for risk signal fields
  readonly timeSinceOpts = [
    { label: 'Just now',    value: 0    },
    { label: '< 30 min',   value: 15   },
    { label: '1 hour',     value: 60   },
    { label: '3 hours',    value: 180  },
    { label: 'Half day',   value: 720  },
    { label: '1 day',      value: 1440 },
  ];

  readonly spendDeviationOpts = [
    { label: 'Normal (0)',       value: 0   },
    { label: 'Slightly off (2)', value: 2   },
    { label: 'Unusual (5)',      value: 5   },
    { label: 'Very unusual (8)', value: 8   },
    { label: 'Extreme (10)',     value: 10  },
  ];

  readonly velocityOpts = [
    { label: 'None (0)',      value: 0  },
    { label: 'Low (1–3)',     value: 2  },
    { label: 'Medium (4–9)', value: 6  },
    { label: 'High (10–15)', value: 12 },
    { label: 'Very high (16+)', value: 18 },
  ];

  readonly geoOpts = [
    { label: 'Normal location',   value: 0   },
    { label: 'Slightly off',      value: 0.3 },
    { label: 'Unusual location',  value: 0.6 },
    { label: 'Impossible travel', value: 1.0 },
  ];

  setField(field: string, value: number): void {
    this.form.get(field)?.setValue(value);
  }

  constructor(private fb: FormBuilder) {
    this.form = this.fb.group({
      // Numerical
      amount_ngn:                   [null, [Validators.required, Validators.min(0.01)]],
      time_since_last_transaction:  [60,   [Validators.required, Validators.min(0)]],
      spending_deviation_score:     [0,    Validators.required],
      velocity_score:               [2,    [Validators.required, Validators.min(0)]],
      geo_anomaly_score:            [0,    [Validators.required, Validators.min(0)]],
      // Categorical
      transaction_type:   ['', Validators.required],
      merchant_category:  ['', Validators.required],
      location:           ['', Validators.required],
      device_used:        ['', Validators.required],
      payment_channel:    ['', Validators.required],
      sender_persona:     ['', Validators.required],
      // Boolean flags
      bvn_linked:             [true,  Validators.required],
      new_device_transaction: [false, Validators.required],
    });
  }

  onSubmit(): void {
    if (this.form.valid) {
      this.submitted.emit(this.form.value as TransactionInput);
    }
  }

  reset(): void {
    this.form.reset({
      bvn_linked: true,
      new_device_transaction: false,
    });
  }
}
