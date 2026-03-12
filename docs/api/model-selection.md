# Model Selection

## Base classes

::: agentopt.model_selection.base.BaseModelSelector
    options:
      members:
        - __init__
        - select_best

::: agentopt.model_selection.base.ModelResult

::: agentopt.model_selection.base.SelectionResults
    options:
      members:
        - get_best
        - get_by_attribute
        - to_csv
        - print_summary

## Selectors

::: agentopt.model_selection.BruteForceModelSelector
    options:
      show_bases: false
      members:
        - select_best

::: agentopt.model_selection.RandomSearchModelSelector
    options:
      show_bases: false
      members:
        - __init__
        - select_best

::: agentopt.model_selection.HillClimbingModelSelector
    options:
      show_bases: false
      members:
        - select_best

::: agentopt.model_selection.ArmEliminationModelSelector
    options:
      show_bases: false
      members:
        - select_best

::: agentopt.model_selection.HyperbandModelSelector
    options:
      show_bases: false
      members:
        - __init__
        - select_best

::: agentopt.model_selection.BayesianOptimizationModelSelector
    options:
      show_bases: false
      members:
        - select_best
