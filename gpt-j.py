import nlpcloud
client = nlpcloud.Client("gpt-j", "your_token", gpu=True)
generation = client.generation("""Message: Support has been terrible for 2 weeks...
            Sentiment: Negative
            ###
            Message: I love your API, it is simple and so fast!
            Sentiment: Positive
            ###
            Message: GPT-J has been released 2 months ago.
            Sentiment: Neutral
            ###
            Message: The reactivity of your team has been amazing, thanks!
            Sentiment:""",
    min_length=1,
    max_length=1,
    length_no_input=True,
    end_sequence="###",
    remove_end_sequence=True,
    remove_input=True)
print(generation["generated_text"])