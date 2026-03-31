import pandas as pd

def submission(test_ids, test_preds):
    submission = pd.DataFrame({"TransactionID": test_ids, "isFraud": test_preds})
    submission.to_csv("submissions/submission.csv", index=False)
