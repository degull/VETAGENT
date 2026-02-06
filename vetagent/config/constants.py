# 고정 액션 순서 (절대 바꾸지 않기)
ACTIONS_ORDER = ["drop", "snow", "rain", "blur", "haze"]

KEY_TO_ID = {k: i for i, k in enumerate(ACTIONS_ORDER)}
ID_TO_KEY = {i: k for k, i in KEY_TO_ID.items()}

def assert_valid_action_key(k: str) -> None:
    if k not in KEY_TO_ID:
        raise ValueError(f"Unknown action key: {k}. Must be one of {ACTIONS_ORDER}")
