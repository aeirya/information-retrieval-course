import replicate

# The mistralai/mistral-7b-instruct-v0.2 model can stream output as it's running.
for event in replicate.stream(
    "mistralai/mistral-7b-instruct-v0.2",
    input={"prompt": "how are you doing today?"},
):
    print(str(event), end="")