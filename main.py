import mlflow
from src.load_data import load_data, split_train_data, prepare_final_test
from src.feature_engineer import categorical_encoding, combine_features, aggregate_user_stats, amt_email_features, frequency_encoding, time_features, remove_zero_importance
from src.models import lgb_model, xgb_model, train_model, optimize_lgb, best_optimize_lgb
from src.evaluation import evaluation, final_evaluation, plot_importance
from src.predict import submission


mlflow.set_experiment("fraud_detection")

with mlflow.start_run(run_name="lgb_best_params"):
        
    train, test =  load_data()
    print('Data loaded successfully!')
    categorical_encoding(train, test)
    combine_features(train, test)
    aggregate_user_stats(train, test)
    time_features(train, test)
    remove_zero_importance(train, test)

    train = train.copy()  # desfragmenta a meio
    test = test.copy()

    amt_email_features(train, test)
    frequency_encoding(train, test)
    print('Features engineered successfully!')

    x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_train_data(train)
    x_final_test, test_ids = prepare_final_test(test)

    #model = best_optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, 50)
    model = lgb_model(scale_pos_weight)
    train_model(model, x_train, y_train, x_cv, y_cv)

    evaluation(model, x_cv, y_cv, x_test, y_test)
    final_preds = final_evaluation(model, x_final_test)

    submission(test_ids, final_preds)
