import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from src.database import init_db, save_prediction, save_actual
from src.predict import predict_for_date, fetch_actual_for_date

LATITUDE  = 6.5244
LONGITUDE = 3.3792


def backfill_days(days: int = 14):
    init_db()
    print(f"🔄 Backfilling last {days} days of predictions and actuals...\n")

    filled_preds   = 0
    filled_actuals = 0
    failed_preds   = 0
    failed_actuals = 0

    for i in range(days, 0, -1):
        target_date = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"📅 Processing {target_date}...")

        # --- Prediction ---
        try:
            result = predict_for_date(target_date, LATITUDE, LONGITUDE)
            save_prediction(
                prediction_date=target_date,
                predicted_rainfall_mm=result["predicted_rainfall_mm"],
                based_on_data_up_to=result["based_on_data_up_to"],
                model_version=result["model"],
            )
            print(f"   ✅ Prediction: {result['predicted_rainfall_mm']}mm")
            filled_preds += 1
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            failed_preds += 1

        # --- Actual ---
        actual = fetch_actual_for_date(target_date, LATITUDE, LONGITUDE)
        if actual is not None:
            save_actual(target_date, actual)
            print(f"   ✅ Actual:     {actual}mm")
            filled_actuals += 1
        else:
            print(f"   ❌ Actual fetch failed")
            failed_actuals += 1

        print()

    print("=" * 48)
    print(f"  Backfill complete!")
    print(f"  Predictions → Saved: {filled_preds} | Failed: {failed_preds}")
    print(f"  Actuals     → Saved: {filled_actuals} | Failed: {failed_actuals}")
    print("=" * 48)


if __name__ == "__main__":
    backfill_days(days=14)