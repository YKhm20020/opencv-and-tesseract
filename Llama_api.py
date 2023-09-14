import replicate

# 〇〇というラベルは、日付、整数、文字列、単一選択、複数選択のうち、どれにあたる？
output = replicate.run(
    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"prompt": 'Which of the following is the label of "年齢" in Japanese ?',
           "system_prompt": 'Answer only in date, integer, string, single selection or multiple selection'}
)

# The replicate/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
output_list = list(output)
output_str = "".join(output_list)
print(output_str)

labels = ['date', 'int', 'str', 'single', 'multi']

for i in range (len(labels)):
    if labels[i] in output_str:
        label = labels[i]
        print(f'The label of input is {label}')
        break
        