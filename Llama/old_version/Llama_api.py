import replicate

# プロンプト工夫の余地あり
output = replicate.run(
    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"prompt": '氏名というラベルは、整数、文字列、単一選択、複数選択のうち、どれにあたる？'}
)

# The replicate/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
output_list = list(output)
output_str = "".join(output_list)
print(output_str)