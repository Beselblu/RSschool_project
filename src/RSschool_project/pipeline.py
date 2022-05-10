from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

def create_pipeline(
    use_standart_scaller: bool, use_minmax_scaller: bool, max_iter:int,
    logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_standart_scaller:
        pipeline_steps.append(('scaller', StandardScaler()))
    if use_minmax_scaller:
        pipeline_steps.append(('scaller', MinMaxScaler()))
    pipeline_steps.append(
        (
            'classifier',
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            )
        )
    )
    return Pipeline(steps=pipeline_steps)