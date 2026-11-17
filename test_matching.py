import numpy as np
import pandas as pd

from matching import flag, normalize_text, validate_input


def test_normalize_text():
    # Capitalization
    assert normalize_text("LAPTOP") == "laptop"

    # Punctuation
    assert normalize_text("Laptop, computer!?#$@") == "laptop computer"

    # Excess whitespace
    assert normalize_text("  laptop   computer  ") == "laptop computer"

    # Empty description
    assert normalize_text("") == ""


def test_flag():
    procurement_cos = np.array([
        [0.00, 0.00, 0.00],
        [0.05, 0.02, 0.01],
        [0.10, 0.05, 0.03],
        [0.20, 0.10, 0.05]
    ])

    # Default threshold
    expected = np.array([True, True, True, False])
    assert np.array_equal(flag(procurement_cos), expected)

    # Custom threshold
    expected = np.array([True, True, False, False])
    assert np.array_equal(flag(procurement_cos, threshold=0.05), expected)


def test_validate_input():
    # Valid input
    df = pd.DataFrame({
        "purchase_id": ["P01"],
        "description": ["Laptop"]
    })

    validate_input(df, ["purchase_id", "description"])

    # Missing column
    try:
        validate_input(df, ["purchase_id", "invalid_column"])
    except ValueError:
        pass
    else:
        raise AssertionError("The invalid column was not detected")

    # Empty dataset
    try:
        validate_input(df.iloc[:0], ["purchase_id", "description"])
    except ValueError:
        pass
    else:
        raise AssertionError("The empty dataset was not detected")


if __name__ == "__main__":
    test_normalize_text()
    test_flag()
    test_validate_input()
    print("All tests passed.")