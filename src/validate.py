import great_expectations as ge

df = ge.read_csv("data/raw/transactions.csv")

df.expect_column_to_exist("amount")
df.expect_column_values_to_not_be_null("label")
df.expect_column_values_to_be_between("amount", 0.01, 50000)
df.expect_column_values_to_be_in_set("label", [0, 1])
df.expect_column_proportion_of_unique_values_to_be_between(
    "merchant_id", 0.01, 1.0)

result = df.validate()
print("PASSED" if result.success else "FAILED")
assert result.success, "Data validation failed — check GE report"