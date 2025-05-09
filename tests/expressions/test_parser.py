#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
from decimal import Decimal

import pytest
from pyparsing import ParseException

import pyiceberg.expressions.parser as parser
from pyiceberg.expressions import (
    AlwaysFalse,
    AlwaysTrue,
    And,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNaN,
    IsNull,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqualTo,
    NotIn,
    NotNaN,
    NotNull,
    NotStartsWith,
    Or,
    StartsWith,
)
from pyiceberg.expressions.literals import DecimalLiteral


def test_always_true() -> None:
    assert AlwaysTrue() == parser.parse("true")


def test_always_false() -> None:
    assert AlwaysFalse() == parser.parse("false")


def test_quoted_column() -> None:
    assert EqualTo("foo", True) == parser.parse('"foo" = TRUE')


def test_leading_underscore() -> None:
    assert EqualTo("_foo", True) == parser.parse("_foo = true")


def test_equals_true() -> None:
    assert EqualTo("foo", True) == parser.parse("foo = true")
    assert EqualTo("foo", True) == parser.parse("foo == TRUE")


def test_equals_false() -> None:
    assert EqualTo("foo", False) == parser.parse("foo = false")
    assert EqualTo("foo", False) == parser.parse("foo == FALSE")


def test_is_null() -> None:
    assert IsNull("foo") == parser.parse("foo is null")
    assert IsNull("foo") == parser.parse("foo IS NULL")


def test_not_null() -> None:
    assert NotNull("foo") == parser.parse("foo is not null")
    assert NotNull("foo") == parser.parse("foo IS NOT NULL")


def test_is_nan() -> None:
    assert IsNaN("foo") == parser.parse("foo is nan")
    assert IsNaN("foo") == parser.parse("foo IS NAN")


def test_not_nan() -> None:
    assert NotNaN("foo") == parser.parse("foo is not nan")
    assert NotNaN("foo") == parser.parse("foo IS NOT NaN")


def test_less_than() -> None:
    assert LessThan("foo", 5) == parser.parse("foo < 5")
    assert LessThan("foo", "a") == parser.parse("'a' > foo")


def test_less_than_or_equal() -> None:
    assert LessThanOrEqual("foo", 5) == parser.parse("foo <= 5")
    assert LessThanOrEqual("foo", "a") == parser.parse("'a' >= foo")


def test_greater_than() -> None:
    assert GreaterThan("foo", 5) == parser.parse("foo > 5")
    assert GreaterThan("foo", "a") == parser.parse("'a' < foo")


def test_greater_than_or_equal() -> None:
    assert GreaterThanOrEqual("foo", 5) == parser.parse("foo >= 5")
    assert GreaterThanOrEqual("foo", "a") == parser.parse("'a' <= foo")


def test_equal_to() -> None:
    assert EqualTo("foo", 5) == parser.parse("foo = 5")
    assert EqualTo("foo", "a") == parser.parse("'a' = foo")
    assert EqualTo("foo", "a") == parser.parse("foo == 'a'")
    assert EqualTo("foo", 5) == parser.parse("5 == foo")


def test_not_equal_to() -> None:
    assert NotEqualTo("foo", 5) == parser.parse("foo != 5")
    assert NotEqualTo("foo", "a") == parser.parse("'a' != foo")
    assert NotEqualTo("foo", "a") == parser.parse("foo <> 'a'")
    assert NotEqualTo("foo", 5) == parser.parse("5 <> foo")


def test_in() -> None:
    assert In("foo", {5, 6, 7}) == parser.parse("foo in (5, 6, 7)")
    assert In("foo", {"a", "b", "c"}) == parser.parse("foo IN ('a', 'b', 'c')")


def test_in_different_types() -> None:
    with pytest.raises(ParseException):
        parser.parse("foo in (5, 'a')")


def test_not_in() -> None:
    assert NotIn("foo", {5, 6, 7}) == parser.parse("foo not in (5, 6, 7)")
    assert NotIn("foo", {"a", "b", "c"}) == parser.parse("foo NOT IN ('a', 'b', 'c')")


def test_not_in_different_types() -> None:
    with pytest.raises(ParseException):
        parser.parse("foo not in (5, 'a')")


def test_simple_and() -> None:
    assert And(GreaterThanOrEqual("foo", 5), LessThan("foo", 10)) == parser.parse("5 <= foo and foo < 10")


def test_and_with_not() -> None:
    assert And(Not(GreaterThanOrEqual("foo", 5)), LessThan("foo", 10)) == parser.parse("not 5 <= foo and foo < 10")
    assert And(GreaterThanOrEqual("foo", 5), Not(LessThan("foo", 10))) == parser.parse("5 <= foo and not foo < 10")


def test_or_with_not() -> None:
    assert Or(Not(LessThan("foo", 5)), GreaterThan("foo", 10)) == parser.parse("not foo < 5 or 10 < foo")
    assert Or(LessThan("foo", 5), Not(GreaterThan("foo", 10))) == parser.parse("foo < 5 or not 10 < foo")


def test_simple_or() -> None:
    assert Or(LessThan("foo", 5), GreaterThan("foo", 10)) == parser.parse("foo < 5 or 10 < foo")


def test_and_or_without_parens() -> None:
    assert Or(And(NotNull("foo"), LessThan("foo", 5)), GreaterThan("foo", 10)) == parser.parse(
        "foo is not null and foo < 5 or 10 < foo"
    )
    assert Or(IsNull("foo"), And(GreaterThanOrEqual("foo", 5), LessThan("foo", 10))) == parser.parse(
        "foo is null or 5 <= foo and foo < 10"
    )


def test_and_or_with_parens() -> None:
    assert And(NotNull("foo"), Or(LessThan("foo", 5), GreaterThan("foo", 10))) == parser.parse(
        "foo is not null and (foo < 5 or 10 < foo)"
    )
    assert Or(IsNull("foo"), And(GreaterThanOrEqual("foo", 5), Not(LessThan("foo", 10)))) == parser.parse(
        "(foo is null) or (5 <= foo) and not(foo < 10)"
    )


def test_multiple_and_or() -> None:
    assert And(EqualTo("foo", 1), EqualTo("bar", 2), EqualTo("baz", 3)) == parser.parse("foo = 1 and bar = 2 and baz = 3")
    assert Or(EqualTo("foo", 1), EqualTo("foo", 2), EqualTo("foo", 3)) == parser.parse("foo = 1 or foo = 2 or foo = 3")
    assert Or(
        And(NotNull("foo"), LessThan("foo", 5)), And(GreaterThan("foo", 10), LessThan("foo", 100), IsNull("bar"))
    ) == parser.parse("foo is not null and foo < 5 or (foo > 10 and foo < 100 and bar is null)")


def test_like_equality() -> None:
    assert EqualTo("foo", "data") == parser.parse("foo LIKE 'data'")
    assert EqualTo("foo", "data%") == parser.parse("foo LIKE 'data\\%'")


def test_starts_with() -> None:
    assert StartsWith("foo", "data") == parser.parse("foo LIKE 'data%'")
    assert StartsWith("foo", "some % data") == parser.parse("foo LIKE 'some \\% data%'")
    assert StartsWith("foo", "some data%") == parser.parse("foo LIKE 'some data\\%%'")


def test_invalid_likes() -> None:
    invalid_statements = ["foo LIKE '%data%'", "foo LIKE 'da%ta'", "foo LIKE '%data'"]

    for statement in invalid_statements:
        with pytest.raises(ValueError) as exc_info:
            parser.parse(statement)

        assert "LIKE expressions only supports wildcard, '%', at the end of a string" in str(exc_info)


def test_not_starts_with() -> None:
    assert NotEqualTo("foo", "data") == parser.parse("foo NOT LIKE 'data'")
    assert NotStartsWith("foo", "data") == parser.parse("foo NOT LIKE 'data%'")


def test_with_function() -> None:
    with pytest.raises(ParseException) as exc_info:
        parser.parse("foo = 1 and lower(bar) = '2'")

    assert "Expected end of text, found 'and'" in str(exc_info)


def test_nested_fields() -> None:
    assert EqualTo("foo.bar", "data") == parser.parse("foo.bar = 'data'")
    assert LessThan("location.x", DecimalLiteral(Decimal(52.00))) == parser.parse("location.x < 52.00")


def test_quoted_column_with_dots() -> None:
    with pytest.raises(ParseException) as exc_info:
        parser.parse("\"foo.bar\".baz = 'data'")

    with pytest.raises(ParseException) as exc_info:
        parser.parse("'foo.bar'.baz = 'data'")

    assert "Expected <= | <> | < | >= | > | == | = | !=, found '.'" in str(exc_info.value)


def test_quoted_column_with_spaces() -> None:
    assert EqualTo("Foo Bar", "data") == parser.parse("\"Foo Bar\" = 'data'")
