# Prophet project solution guidance

The reference disables irrelevant seasonalities, removes posterior sampling for a fast deterministic smoke test, constrains changepoint flexibility, and verifies predictions occur strictly after training. Real work must use rolling-origin evaluation, naive baselines, calendar knowledge available at forecast time, interval assessment, and sensitivity to changepoint/seasonality priors.
