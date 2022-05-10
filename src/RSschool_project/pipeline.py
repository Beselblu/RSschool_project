from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

def create_pipeline(
    use_feature_selection: bool, use_standart_scaler: bool, 
    use_minmax_scaler: bool, max_iter:int,
    logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_feature_selection:
        pipeline_steps.append(('feature_selection', VarianceThreshold()))
    if use_standart_scaler:
        pipeline_steps.append(('scaler', StandardScaler()))
    if use_minmax_scaler:
        pipeline_steps.append(('scaler', MinMaxScaler()))
    pipeline_steps.append(
        (
            'classifier',
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            )
        )
    )
    return Pipeline(steps=pipeline_steps)