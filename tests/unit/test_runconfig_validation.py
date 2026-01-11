from pathlib import Path

from infrastructure.config.models import DataColumnsConfig, RunConfig


def test_include_subcategory_in_icl_demo_is_ignored_when_include_icl_demo_false() -> None:
    cfg = RunConfig(
        model="dummy-model",
        test_file_path=Path("data/test.csv"),
        columns=DataColumnsConfig(
            test_question_col="question",
            test_category_col=None,
            test_subcategory_col=None,
            icl_demo_question_col=None,
            icl_demo_category_col=None,
            icl_demo_subcategory_col=None,
        ),
        include_icl_demo=False,
        include_subcategory_in_icl_demo=True,  # should be forced off by validator
        label_subcategories=False,  # avoids requiring test_subcategory_col
    )

    assert cfg.include_subcategory_in_icl_demo is False
