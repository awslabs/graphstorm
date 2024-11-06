import pytest
from common_parser import parse_unknown_gs_args


def test_basic_parsing():
    args = ["--num-epochs", "1", "--use-graphbolt", "true"]
    expected = {"num-epochs": "1", "use-graphbolt": "true"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_multiple_values():
    args = [
        "--target-etype",
        "query,clicks,asin",
        "query,search,asin",
        "--feat-name",
        "ntype0:feat0",
        "ntype1:feat1",
    ]
    expected = {
        "target-etype": "query,clicks,asin query,search,asin",
        "feat-name": "ntype0:feat0 ntype1:feat1",
    }
    assert dict(parse_unknown_gs_args(args)) == expected


def test_empty_value():
    args = ["--empty-arg", "--next-arg", "value"]
    expected = {"empty-arg": "", "next-arg": "value"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_no_args():
    args = []
    result = parse_unknown_gs_args(args)
    assert len(result) == 0


def test_only_flags():
    args = ["--flag1", "--flag2", "--flag3"]
    expected = {"flag1": "", "flag2": "", "flag3": ""}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_mixed_args():
    args = ["--arg1", "value1", "--flag", "--arg2", "value2a", "value2b"]
    expected = {"arg1": "value1", "flag": "", "arg2": "value2a value2b"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_quoted_args():
    args = ["--arg", '"quoted value"', "--another-arg", "'single quoted'"]
    expected = {"arg": '"quoted value"', "another-arg": "'single quoted'"}
    assert dict(parse_unknown_gs_args(args)) == expected


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (["--special-chars", "!@#$%^&*()"], {"special-chars": "!@#$%^&*()"}),
        (["--unicode", "こんにちは", "world"], {"unicode": "こんにちは world"}),
    ],
)
def test_parse_unknown_gs_args_parametrized(input_args, expected_output):
    assert dict(parse_unknown_gs_args(input_args)) == expected_output


def test_parse_single_string():
    """Happens when GS args are passed in quoted in bash:
    ``python launch_*.py --launch-arg 1 '--gs-arg1 2 --gs-arg2 3'``
    """
    args = ["--feat-name ntype0:feat0 ntype1:feat1"]
    expected = {"feat-name": "ntype0:feat0 ntype1:feat1"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_parse_complex_string_with_quotes():
    args = [
        (
            "--target-etype query,clicks,asin query,search,asin "
            "--feat-name ntype0:feat0 ntype1:feat1"
        )
    ]
    expected = {
        "target-etype": "query,clicks,asin query,search,asin",
        "feat-name": "ntype0:feat0 ntype1:feat1",
    }
    assert dict(parse_unknown_gs_args(args)) == expected
